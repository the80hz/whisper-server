"""Cutting a long recording into pieces an ASR model can take, at pauses found by a VAD."""

from __future__ import annotations

import math

Span = tuple[float, float]

MAX_CHUNK = 24.0
"""Seconds; GigaAM takes up to 25 s at once."""


def plan_chunks(speech: list[Span], *, duration: float, max_chunk: float = MAX_CHUNK, pad: float = 0.3) -> list[Span]:
    """Chunks of at most `max_chunk` seconds, padding included, that cover all `speech` spans.

    Neighbouring speech spans are merged while the chunk stays short enough, and the cut
    goes into the longest pause of the second half of that window: a cut in a short pause
    (a comma) clips the last sound of a word. A single span longer than a chunk is cut into
    equal pieces: there is no pause to use. Each chunk is then widened by up to `pad` on
    both sides, never by more than half of the gap to its neighbour and never outside the
    recording.
    """

    limit = max_chunk - 2 * pad
    if limit <= 0:
        raise ValueError("max_chunk must be longer than twice the pad")
    pieces: list[Span] = []
    for start, end in sorted(speech):
        end = min(end, duration)
        if end <= start:
            continue
        parts = math.ceil((end - start) / limit)
        size = (end - start) / parts
        pieces.extend((start + i * size, start + (i + 1) * size) for i in range(parts))

    chunks: list[Span] = []
    first = 0
    while first < len(pieces):
        start = pieces[first][0]
        last = first
        while last + 1 < len(pieces) and pieces[last + 1][1] - start <= limit:
            last += 1
        if last + 1 < len(pieces):
            # More speech follows: cut at the longest pause late enough in the window.
            late = [i for i in range(first, last + 1) if pieces[i][1] - start >= limit / 2]
            last = max(late or [last], key=lambda i: (pieces[i + 1][0] - pieces[i][1], i))
        chunks.append((start, pieces[last][1]))
        first = last + 1

    padded: list[Span] = []
    for index, (start, end) in enumerate(chunks):
        before = start - chunks[index - 1][1] if index else start
        after = chunks[index + 1][0] - end if index + 1 < len(chunks) else duration - end
        padded.append((start - min(pad, before / 2), end + min(pad, after / 2)))
    return padded
