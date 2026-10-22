import io
import json

import httpx
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.stt import ElevenLabsTranscriber, GeminiTranscriber
from app.tts import (LEADING_PAD_SEC, TRAILING_PAD_SEC, ElevenLabsSynthesizer, GeminiSynthesizer,
                     VieneuSynthesizer, create_synthesizer)


class FakeVieneu:
    sample_rate = 48_000
    backend = "onnx"

    def __init__(self):
        self.calls = 0
        self.stream_calls = 0

    def list_preset_voices(self):
        return [("⭐ Hải Đăng — Nam · Bắc", "Hải Đăng"), ("Mai Anh — Nữ · Bắc", "Mai Anh")]

    def infer(self, text, voice=None, **_):
        if voice not in {"Hải Đăng", "Mai Anh"}:
            raise ValueError(f"Voice '{voice}' not found.")
        self.calls += 1
        return np.full(self.sample_rate, 0.1, dtype=np.float32)  # 1 second

    def infer_stream(self, text, voice=None, **_):
        self.stream_calls += 1
        for _ in range(4):  # 4 x 0.25 s
            yield np.full(self.sample_rate // 4, 0.1, dtype=np.float32)


def elevenlabs_ok(text="Xin chào"):
    return {"text": text, "language_code": "vie"}


def make_client(handler, **overrides):
    settings = Settings(_env_file=None, elevenlabs_api_key="test-key", tts_preload=False, **overrides)
    transcriber = ElevenLabsTranscriber(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    fake = FakeVieneu()
    app = create_app(settings, transcriber, VieneuSynthesizer(settings, model=fake))
    return TestClient(app), fake


def test_stt_sends_multipart_to_elevenlabs_scribe():
    captured = {}

    def handler(request: httpx.Request):
        captured["key"] = request.headers.get("xi-api-key")
        captured["content_type"] = request.headers.get("content-type")
        content = request.content.decode("latin1")
        captured["has_file"] = "filename=\"audio.webm\"" in content
        captured["has_model"] = "scribe_v2" in content
        return httpx.Response(200, json=elevenlabs_ok("Tôi dùng Kubernetes"))

    client, _ = make_client(handler)
    response = client.post("/v1/stt", files={"file": ("a.webm", b"opus-bytes", "audio/webm;codecs=opus")},
                           data={"vocabulary": "Kubernetes, Redis"})

    assert response.status_code == 200
    assert response.json()["text"] == "Tôi dùng Kubernetes"
    assert captured["key"] == "test-key"
    assert captured["has_file"] is True
    assert captured["has_model"] is True


def test_stt_auto_language_and_verbatim_mode():
    def handler(request):
        return httpx.Response(200, json=elevenlabs_ok("xin chao"))

    client, _ = make_client(handler)
    response = client.post("/v1/stt", files={"file": ("a.wav", b"RIFF", "audio/wav")},
                           data={"language": "auto", "mode": "verbatim"})

    assert response.status_code == 200
    assert response.json()["mode"] == "verbatim"


@pytest.mark.parametrize("content_type,filename,status", [
    ("text/plain", "notes.txt", 415),
    ("audio/wav", "a.wav", 400),  # empty file
])
def test_stt_rejects_bad_input(content_type, filename, status):
    client, _ = make_client(lambda request: httpx.Response(200, json=elevenlabs_ok()))
    response = client.post("/v1/stt", files={"file": (filename, b"" if status == 400 else b"x", content_type)})
    assert response.status_code == status


def test_stt_maps_elevenlabs_errors():
    client, _ = make_client(lambda request: httpx.Response(503, json={"detail": "overloaded"}))
    response = client.post("/v1/stt", files={"file": ("a.wav", b"RIFF", "audio/wav")})
    assert response.status_code == 502
    assert "overloaded" in response.json()["detail"]


def test_stt_without_api_key_is_unavailable():
    settings = Settings(_env_file=None, elevenlabs_api_key="", tts_preload=False)
    app = create_app(settings, synthesizer=VieneuSynthesizer(settings, model=FakeVieneu()))
    client = TestClient(app)
    assert client.get("/health").json()["stt"]["configured"] is False
    assert client.post("/v1/stt", files={"file": ("a.wav", b"RIFF", "audio/wav")}).status_code == 503


def test_tts_returns_padded_wav_and_caches():
    client, fake = make_client(lambda request: httpx.Response(200))
    first = client.post("/v1/tts", json={"text": "Xin chào", "voice": "Mai Anh"})
    second = client.post("/v1/tts", json={"text": "Xin chào", "voice": "Mai Anh"})

    assert first.status_code == 200 and first.headers["content-type"] == "audio/wav"
    data, rate = sf.read(io.BytesIO(first.content))
    assert rate == 48_000
    assert len(data) == pytest.approx(rate * (1 + LEADING_PAD_SEC + TRAILING_PAD_SEC), abs=2)
    assert float(first.headers["X-Audio-Duration"]) == pytest.approx(1.0)
    assert second.headers["X-Cache-Hit"] == "true"
    assert float(second.headers["X-Audio-Duration"]) == pytest.approx(1.0, abs=0.01)
    assert fake.calls == 1


def test_tts_validation():
    client, _ = make_client(lambda request: httpx.Response(200), tts_max_chars=10)
    assert client.post("/v1/tts", json={"text": "   "}).status_code == 400
    assert client.post("/v1/tts", json={"text": "x" * 11}).status_code == 413
    assert client.post("/v1/tts", json={"text": "Chào", "voice": "Không có"}).status_code == 422


def test_voices_strip_star_marker():
    client, _ = make_client(lambda request: httpx.Response(200))
    body = client.get("/v1/tts/voices").json()
    assert body["voices"][0] == {"id": "Hải Đăng", "label": "Hải Đăng — Nam · Bắc"}


def test_service_token_guards_v1_but_not_health_or_page():
    client, _ = make_client(lambda request: httpx.Response(200, json=elevenlabs_ok()), service_token="s3cret")
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/v1/tts/voices").status_code == 401
    assert client.get("/v1/tts/voices", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/v1/tts/voices", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_tts_stream_yields_padded_pcm_then_serves_cache():
    client, fake = make_client(lambda request: httpx.Response(200))
    first = client.post("/v1/tts/stream", json={"text": "Xin chào", "voice": "Mai Anh"})
    second = client.post("/v1/tts/stream", json={"text": "Xin chào", "voice": "Mai Anh"})

    assert first.status_code == 200 and first.headers["content-type"] == "audio/pcm"
    assert first.headers["X-Sample-Rate"] == "48000" and first.headers["X-Sample-Format"] == "s16le"
    samples = len(first.content) // 2
    assert samples == pytest.approx(48_000 * (1 + LEADING_PAD_SEC + TRAILING_PAD_SEC), abs=4)
    assert second.content == first.content
    assert fake.stream_calls == 1
    assert client.post("/v1/tts", json={"text": "Xin chào", "voice": "Mai Anh"}).headers["X-Cache-Hit"] == "true"


def test_tts_stream_validates_before_streaming():
    client, _ = make_client(lambda request: httpx.Response(200), tts_max_chars=10)
    assert client.post("/v1/tts/stream", json={"text": "  "}).status_code == 400
    assert client.post("/v1/tts/stream", json={"text": "x" * 11}).status_code == 413
    assert client.post("/v1/tts/stream", json={"text": "Chào", "voice": "Không có"}).status_code == 422


@pytest.mark.parametrize("cuda,expected", [(True, "cuda"), (False, "cpu")])
def test_auto_device_follows_cuda_availability(monkeypatch, cuda, expected):
    import sys
    import types

    from app.tts import resolve_device

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert resolve_device("auto") == expected
    assert resolve_device("cpu") == "cpu"


def test_auto_device_without_torch_is_cpu(monkeypatch):
    import sys

    from app.tts import resolve_device

    monkeypatch.setitem(sys.modules, "torch", None)
    assert resolve_device("auto") == "cpu"


@pytest.mark.parametrize("device,expected", [
    ("cpu", {"mode": "v3turbo", "device": "cpu", "precision": "fp32", "threads": 0}),
    ("cuda", {"mode": "v3turbo", "device": "cuda", "dtype": "float32", "max_batch_size": 32, "max_streams": 16}),
])
def test_engine_options_per_device(monkeypatch, device, expected):
    import vieneu

    captured = {}
    monkeypatch.setattr(vieneu, "Vieneu", lambda **kwargs: captured.update(kwargs) or FakeVieneu())
    synthesizer = VieneuSynthesizer(Settings(_env_file=None, tts_device=device, tts_preload=False))
    synthesizer.load()
    assert captured == expected
    assert synthesizer.state == "ready" and synthesizer.default_voice == "Hải Đăng"


# ---- ElevenLabs TTS backend ----

def _pcm(seconds, rate=48_000):
    return (np.full(int(rate * seconds), 0.1, dtype=np.float32) * 32767).astype("<i2").tobytes()


def make_elevenlabs_client(handler, **overrides):
    settings = Settings(_env_file=None, elevenlabs_api_key="test-key", tts_preload=False, tts_backend="elevenlabs",
                        elevenlabs_tts_voice="cgSgspJ2msm6clMCkdW9", **overrides)
    synthesizer = ElevenLabsSynthesizer(settings, httpx.Client(transport=httpx.MockTransport(handler)))
    transcriber = ElevenLabsTranscriber(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return TestClient(create_app(settings, transcriber, synthesizer))


def test_elevenlabs_tts_whole_clip_request_and_cache():
    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        return httpx.Response(200, content=_pcm(1.0, 48_000), headers={"content-type": "audio/pcm"})

    client = make_elevenlabs_client(handler)
    first = client.post("/v1/tts", json={"text": "Xin chào", "voice": "cgSgspJ2msm6clMCkdW9"})
    second = client.post("/v1/tts", json={"text": "Xin chào", "voice": "cgSgspJ2msm6clMCkdW9"})

    assert first.status_code == 200
    data, rate = sf.read(io.BytesIO(first.content))
    assert rate == 48_000
    assert float(first.headers["X-Audio-Duration"]) == pytest.approx(1.0)
    assert second.headers["X-Cache-Hit"] == "true" and len(calls) == 1
    assert calls[0].headers["xi-api-key"] == "test-key"
    assert "text-to-speech/cgSgspJ2msm6clMCkdW9" in str(calls[0].url)


def test_elevenlabs_tts_stream():
    pcm = _pcm(0.5, 48_000)

    def handler(request: httpx.Request):
        return httpx.Response(200, content=pcm, headers={"content-type": "audio/pcm"})

    client = make_elevenlabs_client(handler)
    response = client.post("/v1/tts/stream", json={"text": "Xin chào"})

    assert response.status_code == 200 and response.headers["X-Sample-Rate"] == "48000"
    lead, trail = int(48_000 * LEADING_PAD_SEC) * 2, int(48_000 * TRAILING_PAD_SEC) * 2
    assert response.content[lead:len(response.content) - trail] == pcm
    assert response.headers["X-Voice"] == "cgSgspJ2msm6clMCkdW9"


def test_elevenlabs_tts_errors_surface_before_streaming():
    client = make_elevenlabs_client(lambda request: httpx.Response(400, json={"detail": {"message": "bad voice"}}))
    response = client.post("/v1/tts/stream", json={"text": "Xin chào", "voice": "Nope"})
    assert response.status_code == 400 and "bad voice" in response.json()["detail"]
    assert client.post("/v1/tts", json={"text": "Xin chào"}).status_code == 400


def test_elevenlabs_tts_health_and_voices():
    def handler(request: httpx.Request):
        if "voices" in str(request.url):
            return httpx.Response(200, json={"voices": [{"voice_id": "v1", "name": "Jessica"}]})
        return httpx.Response(200)

    client = make_elevenlabs_client(handler)
    tts = client.get("/health").json()["tts"]
    assert tts["provider"] == "elevenlabs" and tts["model"] == "eleven_v4_turbo" and tts["state"] == "ready"
    voices = client.get("/v1/tts/voices").json()
    assert voices["default"] == "cgSgspJ2msm6clMCkdW9"
    assert {"id": "v1", "label": "Jessica"} in voices["voices"]


@pytest.mark.parametrize("backend,cuda,expected", [
    ("auto", True, VieneuSynthesizer),
    ("auto", False, ElevenLabsSynthesizer),
    ("vieneu", False, VieneuSynthesizer),
    ("elevenlabs", True, ElevenLabsSynthesizer),
])
def test_backend_selection(monkeypatch, backend, cuda, expected):
    import app.tts as tts

    monkeypatch.setattr(tts, "resolve_device", lambda setting: "cuda" if cuda or setting == "cuda" else "cpu")
    synthesizer = create_synthesizer(Settings(_env_file=None, elevenlabs_api_key="k", tts_backend=backend,
                                              tts_device="cpu" if backend == "vieneu" else "auto"))
    assert type(synthesizer) is expected
    if backend == "auto" and cuda:
        assert synthesizer.device == "cuda"
