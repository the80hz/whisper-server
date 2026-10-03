"""Audio inspection and per-channel decoding built on PyAV (a faster-whisper dependency)."""

from __future__ import annotations

from dataclasses import dataclass

import av
import numpy as np
from av.audio.resampler import AudioResampler

SAMPLE_RATE = 16_000


@dataclass(frozen=True, slots=True)
class AudioInfo:
    """What the container header says; either field is None when it is not there."""

    duration: float | None
    channels: int | None


def probe_audio(path: str) -> AudioInfo:
    """Read duration and channel count from the header without decoding."""

    duration: float | None = None
    channels: int | None = None
    with av.open(path) as container:
        stream = container.streams.audio[0]
        if container.duration is not None:
            duration = float(container.duration) / av.time_base
        elif stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        channels = int(stream.codec_context.channels) or None
    return AudioInfo(duration=duration, channels=channels)


def decode_channels(path: str) -> list[np.ndarray]:
    """Decode `path` to one float32 16 kHz array per channel.

    Mirrors faster-whisper's own decoding but keeps the channels apart instead of
    downmixing them, so any channel count works.
    """

    resampler = AudioResampler(format="s16p", rate=SAMPLE_RATE)
    chunks: list[list[np.ndarray]] = []

    def collect(frames: list[av.AudioFrame]) -> None:
        for frame in frames:
            planes = frame.to_ndarray()  # (channels, samples) for planar formats
            while len(chunks) < planes.shape[0]:
                chunks.append([])
            for index, plane in enumerate(planes):
                chunks[index].append(plane)

    with av.open(path) as container:
        for frame in container.decode(audio=0):
            collect(resampler.resample(frame))
        collect(resampler.resample(None))

    # int16 until the last moment: a 2 h channel is 230 MB as int16, 460 MB as float32.
    return [
        np.concatenate(parts).astype(np.float32) / 32768.0 if parts else np.zeros(0, dtype=np.float32)
        for parts in chunks
    ]
