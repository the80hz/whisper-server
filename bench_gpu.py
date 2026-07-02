"""GPU benchmark for faster-whisper.

Runs several model/compute_type combinations and prints RTF plus rough VRAM
usage from nvidia-smi.

Usage:
    python bench_gpu.py <audio_file>

Example:
    python bench_gpu.py test_for_bratishkabot.mp3
"""

from __future__ import annotations

import subprocess
import sys
import time

from faster_whisper import WhisperModel

audio_path = sys.argv[1] if len(sys.argv) > 1 else "test_for_bratishkabot.mp3"

configs = [
    ("small", "float16"),
    ("medium", "float16"),
    ("large-v3-turbo", "float16"),
    ("large-v3-turbo", "int8_float16"),
]


def vram_total() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout.strip().replace(", ", " / ") + " MiB"
    return "n/a"


print(f"Audio: {audio_path}")
print(
    f"{'Model':<20} {'compute':<16} {'Load':>10} "
    f"{'Processing':>12} {'RTF':>7} {'VRAM (nvidia-smi)':>20}"
)
print("-" * 90)

for model_name, compute_type in configs:
    try:
        load_start = time.monotonic()
        model = WhisperModel(model_name, device="cuda", compute_type=compute_type)
        load_time = time.monotonic() - load_start

        transcribe_start = time.monotonic()
        segments_iter, info = model.transcribe(audio_path, vad_filter=True)
        segments = list(segments_iter)
        processing_time = time.monotonic() - transcribe_start
        rtf = processing_time / info.duration if info.duration else float("nan")

        text = "".join(segment.text for segment in segments).strip()
        print(
            f"{model_name:<20} {compute_type:<16} {load_time:>9.2f}s "
            f"{processing_time:>11.2f}s {rtf:>7.2f}  {vram_total():>20}"
        )
        print(f"  Text: {text}")

        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    except Exception as exc:  # noqa: BLE001
        print(f"{model_name:<20} {compute_type:<16} {'ERROR:':>10} {exc}")

print("-" * 90)
