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
from .chunking import MAX_CHUNK, plan_chunks

logger = logging.getLogger("whisper-api.gigaam")

PAUSE = 0.5
"""Seconds of silence between two words that end a phrase."""
SENTENCE_PAUSE = 0.25
"""The same after a word that ends a sentence."""
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
    ) -> None:
        try:
            import gigaam
            import torch
        except ImportError as error:
            raise RuntimeError("ENGINE=gigaam needs the optional packages: uv sync --extra gigaam") from error
        if max_chunk >= 25.0:
            raise ValueError("max_chunk must stay below 25 s, the limit of the model")
        self._torch = torch
        self._device = torch.device(device)
        self._batch_size = max(1, batch_size)
        self._max_chunk = max_chunk
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

        torch = self._torch
        if not len(waveform):
            return []
        speech = get_speech_timestamps(
            torch.from_numpy(waveform), self._vad(), sampling_rate=SAMPLE_RATE, return_seconds=True
        )
        chunks = plan_chunks(
            [(float(s["start"]), float(s["end"])) for s in speech], duration=duration, max_chunk=self._max_chunk
        )
        segments: list[Segment] = []
        for batch in batches(chunks, self._batch_size):
            for (start, _end), words in zip(batch, self._recognise(waveform, batch), strict=True):
                segments.extend(phrases_of_chunk(words, offset=start))
        return sorted(segments, key=lambda segment: (segment.start, segment.end))

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


def _load(audio: str | np.ndarray) -> np.ndarray:
    """Mono float32 at 16 kHz: a file is decoded and downmixed, an array is one channel already."""

    if isinstance(audio, np.ndarray):
        return np.ascontiguousarray(audio, dtype=np.float32)
    from faster_whisper.audio import decode_audio

    return np.ascontiguousarray(decode_audio(audio, sampling_rate=SAMPLE_RATE), dtype=np.float32)
