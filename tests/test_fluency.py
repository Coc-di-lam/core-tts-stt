import numpy as np
import pytest

from app import fluency
from tests.test_stt_live import DONE, VAD_END, VAD_START, FakeLive, make_client, receive_until, segment

RATE = 16_000


def clip(*parts):
    """parts: ("speech" | "silence", seconds) -> PCM16 bytes; speech is a 200 Hz tone, silence faint noise."""
    rng = np.random.default_rng(0)
    chunks = []
    for kind, seconds in parts:
        n = int(RATE * seconds)
        if kind == "speech":
            chunks.append(0.3 * np.sin(2 * np.pi * 200 * np.arange(n) / RATE))
        else:
            chunks.append(rng.normal(0, 0.001, n))
    return (np.concatenate(chunks) * 32767).astype("<i2").tobytes()


def test_timing_counts_pauses_between_speech_only():
    pcm = clip(("silence", 0.6), ("speech", 2.0), ("silence", 0.5), ("speech", 1.0), ("silence", 1.2),
               ("speech", 1.0), ("silence", 0.15), ("speech", 0.5), ("silence", 0.4))
    _, m = fluency.analyze(pcm, "Em làm backend bằng FastAPI và Redis.")
    assert m["leading_silence_ms"] == pytest.approx(600, abs=60)
    assert m["pause_count"] == 2  # 0.5 s and 1.2 s; 0.15 s is too short, trailing silence is ignored
    assert m["long_pause_count"] == 1
    assert m["longest_pause_ms"] == pytest.approx(1200, abs=60)
    assert m["total_silence_ms"] == pytest.approx(1700, abs=90)
    assert m["speech_duration_ms"] == pytest.approx(4650, abs=120)  # the 0.15 s gap counts as speech
    assert m["words_per_minute"] > 0


def test_fillers_are_counted_and_stripped_but_particles_kept():
    text = "Ừm, em nghĩ là... ờ, hệ thống dùng Redis à? Dạ, ừm, đúng vậy ạ."
    clean, m = fluency.analyze(clip(("speech", 3.0)), text)
    assert m["fillers"] == {"ừm": 2, "ờ": 1}
    assert m["filler_count"] == 3
    assert "ừm" not in clean.lower() and "ờ" not in clean.split()
    assert "Redis à?" in clean and "vậy ạ" in clean  # particles, not fillers
    assert clean.startswith("Em nghĩ là")


def test_repetitions_ignore_reduplicatives():
    _, m = fluency.analyze(clip(("speech", 3.0)), "Em là em là người làm từ từ, dần dần học, tôi tôi tôi nghĩ vậy.")
    assert m["repetition_count"] == 2  # "em là em là" and "tôi tôi tôi"; "từ từ" / "dần dần" are words


def test_silent_clip_reports_zero_without_errors():
    clean, m = fluency.analyze(clip(("silence", 1.0)), "")
    assert clean == "" and m["speech_duration_ms"] == 0 and m["words_per_minute"] == 0
    assert m["hesitation_level"] == "low"


def test_live_final_carries_fluency_and_forces_verbatim():
    fake = FakeLive(on_audio=[VAD_START], on_end=[segment("Ừm, em dùng Redis."), DONE, VAD_END])
    client, captured = make_client(fake)
    with client.websocket_connect("/v1/stt/stream") as ws:
        ws.send_json({"type": "start", "mode": "smart", "fluency": True})
        assert ws.receive_json()["fluency"] is True
        ws.send_bytes(clip(("silence", 0.3), ("speech", 1.5)))
        ws.send_json({"type": "end"})
        final = receive_until(ws, "final")[-1]
    assert captured["mode"] == "verbatim"
    assert final["text"] == "Em dùng Redis." and final["verbatim"] == "Ừm, em dùng Redis."
    assert final["fluency"]["filler_count"] == 1
    assert final["fluency"]["speech_duration_ms"] == pytest.approx(1500, abs=90)


def test_clean_text_keeps_spaces_between_sentences():
    clean, _ = fluency.analyze(clip(("speech", 2.0)), "Dạ, em là An. Ừm, em làm backend. Ờ... hai năm.")
    assert clean == "Dạ, em là An. Em làm backend. Hai năm."


def test_one_long_pause_in_a_short_answer_is_not_high():
    pcm = clip(("speech", 5.0), ("silence", 1.9), ("speech", 5.0))
    _, m = fluency.analyze(pcm, "Em làm backend hai năm, dự án đặt vé, tối ưu PostgreSQL và Redis cache cho hệ thống.")
    assert m["long_pause_count"] == 1 and m["hesitation_level"] == "medium"
