"""GigaAM-v3 (MIT, Salute Devices) behind the `transcribe()` of faster-whisper.

The server talks to its model through `WhisperModel.transcribe(audio, **options)`, which
returns segments with words and an info object. This class answers the same call, so the
queue, the workers, the channel split and every response format work with either engine.

The model takes up to 25 s of audio at once. Silero VAD finds speech, `plan_chunks` merges
it into chunks of at most `max_chunk` seconds cut at pauses, the chunks go through the
model in batches straight from memory, and each chunk is then split into phrases at the
pauses between its words: a segment is a phrase and not 20 s of speech.

GigaAM has no text input: `hotwords`, `initial_prompt` and the decoding options of Whisper
are accepted and ignored. Russian is the only language.

The packages are an optional extra (`uv sync --extra gigaam`); nothing here is imported
until the engine is built.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
from typing import Any, Iterator

import numpy as np

from .audio import SAMPLE_RATE
from .chunking import MAX_CHUNK, Span, plan_chunks

logger = logging.getLogger("whisper-api.gigaam")

PAUSE = 0.5
"""Seconds of silence between two words that end a phrase."""
SENTENCE_PAUSE = 0.25
"""The same after a word that ends a sentence."""
LEVEL_WINDOW_S = 0.5
"""The stretch of sound brought to one loudness for the second look of the speech detector."""
QUIET_THRESHOLD = 0.3
"""Speech probability that counts as speech in that second look (the detector's own is 0.5)."""
LANGUAGE = "ru"
# A dialogue dash comes out of the model as a word of its own.
NOT_A_WORD = "—–- "


@dataclass(frozen=True, slots=True)
class Word:
    """One word in the shape faster-whisper gives it: the text starts with a space."""

    word: str
    start: float
    end: float
    probability: float | None = None


@dataclass(frozen=True, slots=True)
class Segment:
    start: float
    end: float
    text: str
    words: list[Word]
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None


@dataclass(frozen=True, slots=True)
class Info:
    duration: float
    language: str = LANGUAGE
    language_probability: float = 1.0


def split_at_pauses(words: list[Word]) -> list[list[Word]]:
    """Phrases of `words`: a new one starts after a pause, a shorter pause ends a sentence."""

    phrases: list[list[Word]] = []
    for word in words:
        if phrases:
            previous = phrases[-1][-1]
            sentence_end = previous.word.endswith((".", "?", "!"))
            if word.start - previous.end < (SENTENCE_PAUSE if sentence_end else PAUSE):
                phrases[-1].append(word)
                continue
        phrases.append([word])
    return phrases


def phrases_of_chunk(words: list[tuple[str, float, float]], *, offset: float) -> list[Segment]:
    """Segments of one recognised chunk; `words` are (text, start, end) inside the chunk."""

    spoken = [
        Word(word=" " + text.strip(), start=offset + start, end=offset + end)
        for text, start, end in words
        if text.strip(NOT_A_WORD)
    ]
    return [
        Segment(start=phrase[0].start, end=phrase[-1].end, text="".join(w.word for w in phrase), words=phrase)
        for phrase in split_at_pauses(spoken)
    ]


def batches(chunks: list[tuple[float, float]], size: int) -> list[list[tuple[float, float]]]:
    """Chunks of similar length side by side: a batch is padded to its longest member."""

    by_length = sorted(chunks, key=lambda chunk: chunk[1] - chunk[0], reverse=True)
    return [by_length[i : i + size] for i in range(0, len(by_length), max(1, size))]


def levelled(
    waveform: np.ndarray, *, window: float = LEVEL_WINDOW_S, target: float = 0.1, ceiling: float = 20.0
) -> np.ndarray:
    """The recording with every `window` seconds brought to the same loudness.

    The gain is capped at `ceiling`, so silence stays silence, and smoothed over three
    windows, so a word is not cut in the middle by a jump of the level.
    """

    size = int(window * SAMPLE_RATE)
    if size <= 0 or not len(waveform):
        return waveform
    frames = np.pad(waveform, (0, (-len(waveform)) % size)).reshape(-1, size)
    rms = np.sqrt((frames**2).mean(axis=1)) + 1e-6
    gain = np.convolve(np.minimum(target / rms, ceiling), np.ones(3) / 3, mode="same")
    return np.clip(frames * gain[:, None], -1.0, 1.0).reshape(-1)[: len(waveform)].astype(np.float32)


def unheard(segment: Segment, heard: list[Span], *, margin: float = 0.15) -> list[Segment]:
    """Phrases of the words of `segment` that no `heard` word overlaps in time."""

    fresh = [
        word
        for word in segment.words
        if not any(start - margin < word.end and word.start < end + margin for start, end in heard)
    ]
    return [
        Segment(start=phrase[0].start, end=phrase[-1].end, text="".join(w.word for w in phrase), words=phrase)
        for phrase in split_at_pauses(fresh)
    ]


class GigaAmModel:
    """A loaded GigaAM model that several threads may call at once."""

    def __init__(
        self,
        name: str,
        device: str,
        *,
        download_root: str | None = None,
        batch_size: int = 16,
        max_chunk: float = MAX_CHUNK,
        quiet_speech: bool = True,
    ) -> None:
        try:
            import gigaam
            import torch
        except ImportError as error:
            raise RuntimeError("ENGINE=gigaam needs the optional packages: uv sync --extra gigaam") from error
        _drop_flash_import_error()
        if max_chunk >= 25.0:
            raise ValueError("max_chunk must stay below 25 s, the limit of the model")
        self._torch = torch
        self._device = torch.device(device)
        self._batch_size = max(1, batch_size)
        self._max_chunk = max_chunk
        self._quiet_speech = quiet_speech
        root = Path(download_root).expanduser() if download_root else Path.home() / ".cache" / "gigaam"
        root.mkdir(parents=True, exist_ok=True)
        self._asr: Any = gigaam.load_model(name, device=self._device, download_root=str(root))
        # Silero VAD keeps a recurrent state between calls, so every thread has its own.
        self._vads = threading.local()

    def transcribe(self, audio: str | np.ndarray, **_options: Any) -> tuple[Iterator[Segment], Info]:
        waveform = _load(audio)
        duration = len(waveform) / SAMPLE_RATE
        return iter(self._segments(waveform, duration)), Info(duration=duration)

    def _vad(self) -> Any:
        if not hasattr(self._vads, "model"):
            from silero_vad import load_silero_vad

            self._vads.model = load_silero_vad()
        return self._vads.model

    def _segments(self, waveform: np.ndarray, duration: float) -> list[Segment]:
        from silero_vad import get_speech_timestamps

        if not len(waveform):
            return []
        spans = self._speech(waveform, get_speech_timestamps, threshold=None)
        chunks = plan_chunks(spans, duration=duration, max_chunk=self._max_chunk)
        segments = self._phrases(waveform, chunks)
        if self._quiet_speech:
            # A second look at the same recording with every moment brought to one loudness:
            # the detector misses a quiet voice next to a loud one (an operator answering
            # right after the music on hold), and the model skips it inside a chunk with a
            # loud voice. The speech of the second look is recognised as chunks of its own,
            # and only the words that fall where the first look left silence are kept:
            # nothing of the first look changes or doubles.
            quiet = self._speech(levelled(waveform), get_speech_timestamps, threshold=QUIET_THRESHOLD)
            extra = plan_chunks(quiet, duration=duration, max_chunk=self._max_chunk)
            heard = [(word.start, word.end) for segment in segments for word in segment.words]
            for segment in self._phrases(waveform, extra):
                segments.extend(unheard(segment, heard))
        return sorted(segments, key=lambda segment: (segment.start, segment.end))

    def _phrases(self, waveform: np.ndarray, chunks: list[Span]) -> list[Segment]:
        segments: list[Segment] = []
        for batch in batches(chunks, self._batch_size):
            for (start, _end), words in zip(batch, self._recognise(waveform, batch), strict=True):
                segments.extend(phrases_of_chunk(words, offset=start))
        return segments

    def _speech(self, waveform: np.ndarray, detect: Any, *, threshold: float | None) -> list[Span]:
        options = {} if threshold is None else {"threshold": threshold}
        found = detect(
            self._torch.from_numpy(waveform), self._vad(), sampling_rate=SAMPLE_RATE, return_seconds=True, **options
        )
        return [(float(s["start"]), float(s["end"])) for s in found]

    def _recognise(
        self, waveform: np.ndarray, batch: list[tuple[float, float]]
    ) -> list[list[tuple[str, float, float]]]:
        """Words of every chunk of the batch, timed from the start of the chunk."""

        torch = self._torch
        pieces = [waveform[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)] for start, end in batch]
        lengths = torch.tensor([len(piece) for piece in pieces], device=self._device)
        padded = np.zeros((len(pieces), max(len(piece) for piece in pieces)), dtype=np.float32)
        for row, piece in enumerate(pieces):
            padded[row, : len(piece)] = piece
        with torch.inference_mode():
            wav = torch.from_numpy(padded).to(self._device).to(self._asr._dtype)
            encoded, encoded_len = self._asr.forward(wav, lengths)
            decoded = self._asr._decode(encoded, encoded_len, lengths, True)
        return [[(str(w.text), float(w.start), float(w.end)) for w in (words or [])] for _text, words in decoded]


def _drop_flash_import_error() -> None:
    """Keep `gigaam.encoder` from pinning the first model in memory forever.

    Without flash_attn the module keeps the ImportError of its import in `IMPORT_FLASH_ERR`.
    The traceback of that error holds the frames of the import, and through `f_back` every
    frame below it: when the module is first imported while a model is being built, that is
    the `__init__` of the model, so unloading it never freed its weights (447 MiB of VRAM
    on the GPU). Importing the module here and dropping the traceback breaks the chain.
    """

    import gigaam.encoder

    error = getattr(gigaam.encoder, "IMPORT_FLASH_ERR", None)
    if isinstance(error, BaseException):
        error.__traceback__ = None


def _load(audio: str | np.ndarray) -> np.ndarray:
    """Mono float32 at 16 kHz: a file is decoded and downmixed, an array is one channel already."""

    if isinstance(audio, np.ndarray):
        return np.ascontiguousarray(audio, dtype=np.float32)
    from faster_whisper.audio import decode_audio

    return np.ascontiguousarray(decode_audio(audio, sampling_rate=SAMPLE_RATE), dtype=np.float32)
