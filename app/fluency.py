"""Delivery metrics for one spoken answer: pauses from the PCM, fillers/repetitions from a verbatim transcript.

Descriptive only (how the answer was delivered), never a judgement of what was said. Field names
`speech_duration_ms`, `pause_count`, `total_silence_ms` and `words_per_minute` match the core's
`candidate_answer.vad_metrics`, so the dict can be forwarded unchanged.

Heuristics, tuned for Vietnamese with English terms:
- Speech/silence: 30 ms frames, threshold 12 dB above the clip's noise floor (10th percentile
  frame energy) and never below -50 dBFS. Silences between speech of >= 300 ms are pauses,
  >= 1 s long pauses. Leading silence (key pressed, not yet talking) is reported separately.
- Fillers: standalone hesitation sounds (ừm, ờ, ơ, um, uh…). "à"/"ừ" also serve as particles or
  "yes", so they count only when set off by punctuation or at the start of a clause.
- Repetitions: a 1–3 word phrase said again immediately ("em là em là"), ignoring single
  syllables said twice, which are often Vietnamese reduplicatives ("từ từ", "dần dần").
"""
from __future__ import annotations

import re

import numpy as np

FRAME_MS = 30
PAUSE_MS = 300
LONG_PAUSE_MS = 1000
_FILLERS = {"ừm", "ừmm", "ưm", "ờ", "ờm", "ơ", "ơm", "ầm", "ừa", "hừm", "hmm", "hm", "mm", "um", "umm",
            "uh", "uhm", "erm", "ah", "eh"}
_CLAUSE_FILLERS = {"à", "ừ"}  # also particles / "yes": only when isolated by punctuation
_WORD = re.compile(r"[\wÀ-ỹ]+(?:['’][\wÀ-ỹ]+)?", re.UNICODE)


def analyze(pcm: bytes, verbatim: str, sample_rate: int = 16_000) -> tuple[str, dict]:
    """Return (answer text without fillers, metrics) for one turn of PCM16 LE mono audio."""
    timing = _timing(pcm, sample_rate)
    fillers = _fillers(verbatim)
    clean = _strip_fillers(verbatim)
    words = len(_WORD.findall(clean))
    span_ms = timing.pop("_span_ms")
    wpm = round(words / (span_ms / 60_000)) if span_ms >= 1000 and words else 0
    repetitions = _repetitions(clean)
    filler_count = sum(fillers.values())
    per_100 = round(filler_count * 100 / max(words, 1), 1)
    # Rates over at least 30 s, so one long pause in a short answer does not read as "high".
    long_per_min = timing["long_pause_count"] / max(span_ms / 60_000, 0.5)
    level = ("high" if per_100 >= 8 or long_per_min >= 4 else
             "medium" if per_100 >= 3 or long_per_min >= 2 or repetitions >= 3 else "low")
    return clean, {
        **timing,
        "words_per_minute": wpm,
        "word_count": words,
        "filler_count": filler_count,
        "fillers": dict(sorted(fillers.items(), key=lambda kv: -kv[1])),
        "fillers_per_100_words": per_100,
        "repetition_count": repetitions,
        "hesitation_level": level,
    }


def _timing(pcm: bytes, rate: int) -> dict:
    samples = np.frombuffer(pcm[:len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float32) / 32768.0
    frame = max(1, rate * FRAME_MS // 1000)
    count = samples.size // frame
    empty = {"audio_duration_ms": round(samples.size * 1000 / rate), "speech_duration_ms": 0, "pause_count": 0,
             "long_pause_count": 0, "total_silence_ms": 0, "longest_pause_ms": 0, "leading_silence_ms": 0,
             "_span_ms": 0}
    if count == 0:
        return empty
    rms = np.sqrt(np.mean(samples[:count * frame].reshape(count, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    threshold = max(float(np.percentile(db, 10)) + 12, -50.0)
    voiced = db > threshold
    if not voiced.any():
        return empty
    first, last = int(np.argmax(voiced)), int(count - 1 - np.argmax(voiced[::-1]))
    pauses, run = [], 0
    for is_voiced in voiced[first:last + 1]:
        if is_voiced:
            if run * FRAME_MS >= PAUSE_MS:
                pauses.append(run * FRAME_MS)
            run = 0
        else:
            run += 1
    span_ms = (last - first + 1) * FRAME_MS
    silence = sum(pauses)
    return {
        "audio_duration_ms": round(samples.size * 1000 / rate),
        "speech_duration_ms": span_ms - silence,
        "pause_count": len(pauses),
        "long_pause_count": sum(p >= LONG_PAUSE_MS for p in pauses),
        "total_silence_ms": silence,
        "longest_pause_ms": max(pauses, default=0),
        "leading_silence_ms": first * FRAME_MS,
        "_span_ms": span_ms,
    }


def _tokens(text: str) -> list[tuple[str, bool]]:
    """(lowercased word, isolated) pairs; isolated = at clause start or followed by punctuation."""
    out = []
    for match in _WORD.finditer(text):
        before = text[:match.start()].rstrip()
        after = text[match.end():match.end() + 1]
        isolated = not before or before[-1] in ",.;:!?…-—" or after in {",", ".", "…", "-", "—", ";"}
        out.append((match.group().lower(), isolated))
    return out


def _is_filler(word: str, isolated: bool) -> bool:
    return word in _FILLERS or (isolated and word in _CLAUSE_FILLERS)


def _fillers(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for word, isolated in _tokens(text):
        if _is_filler(word, isolated):
            counts[word] = counts.get(word, 0) + 1
    return counts


def _strip_fillers(text: str) -> str:
    def replace(match: re.Match) -> str:
        word = match.group()
        before = text[:match.start()].rstrip()
        after = text[match.end():match.end() + 1]
        isolated = not before or before[-1] in ",.;:!?…-—" or after in {",", ".", "…", "-", "—", ";"}
        return "" if _is_filler(word.lower(), isolated) else word

    cleaned = _WORD.sub(replace, text)
    cleaned = re.sub(r"\s*([,…]\s*)+(?=[,.!?…])", "", cleaned)  # "ừm, ..." leftovers
    cleaned = re.sub(r"(^|[.!?]\s+)(?:[,…]|\.{2,})+\s*", r"\1", cleaned)  # sentence left starting with ", " / "..."
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,")
    # Re-capitalize sentences whose first word was a filler (not after a mid-sentence "...").
    return re.sub(r"(^|(?<!\.)[.!?]\s+)(\w)", lambda m: m[1] + m[2].upper(), cleaned)


def _repetitions(text: str) -> int:
    words = [match.group().lower() for match in _WORD.finditer(text)]
    count, i = 0, 0
    while i < len(words):
        for n in (3, 2):
            if i + 2 * n <= len(words) and words[i:i + n] == words[i + n:i + 2 * n]:
                count += 1
                i += n
                break
        else:
            if i + 2 < len(words) and words[i] == words[i + 1] == words[i + 2]:  # "em em em"
                count += 1
                i += 2
            i += 1
    return count
