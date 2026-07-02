"""Local faster-whisper benchmark for CPU-only hosts.

The default options approximate older VDS hardware by forcing AVX and limiting
CPU threads. Accuracy is hardware-dependent, but the resulting RTF is useful for
capacity planning.

Usage:
    python bench.py <model> <audio> [cpu_threads]

Example:
    python bench.py small test_for_bratishkabot.mp3 2
"""

from __future__ import annotations

import os
import sys
import time

# Set ISA before importing ctranslate2/faster_whisper.
os.environ.setdefault("CT2_FORCE_CPU_ISA", "AVX")

from faster_whisper import WhisperModel  # noqa: E402

model_name = sys.argv[1] if len(sys.argv) > 1 else "small"
audio_path = sys.argv[2] if len(sys.argv) > 2 else "test_for_bratishkabot.mp3"
cpu_threads = int(sys.argv[3]) if len(sys.argv) > 3 else 2

print(
    f"ISA={os.environ['CT2_FORCE_CPU_ISA']} model={model_name} "
    f"threads={cpu_threads} compute=int8 file={audio_path}"
)

load_start = time.monotonic()
model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=cpu_threads)
print(f"Model load: {time.monotonic() - load_start:.2f}s")

start = time.monotonic()
segments_iter, info = model.transcribe(audio_path, vad_filter=True)
segments = list(segments_iter)
elapsed = time.monotonic() - start

text = "".join(segment.text for segment in segments)
rtf = elapsed / info.duration if info.duration else float("nan")

print("-" * 60)
print(f"Language:      {info.language} (p={info.language_probability:.2f})")
print(f"Duration:      {info.duration:.2f}s")
print(f"Processing:    {elapsed:.2f}s")
print(f"RTF:           {rtf:.2f}  (>1 = slower than real time)")
print(f"Segments:      {len(segments)}")
print("-" * 60)
print("Text:")
print(text.strip())
