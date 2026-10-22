import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.tts import VieneuSynthesizer
from tests.test_api import FakeVieneu


class FakeLive:
    """Scripted ElevenLabs Scribe Live: replies are queued when audio arrives or the stream ends."""

    def __init__(self, on_audio=(), on_end=()):
        self.on_audio = list(on_audio)
        self.on_end = list(on_end)
        self.sent: list = []
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.closed = False

    async def send(self, message) -> None:
        self.sent.append(message)
        try:
            data = json.loads(message) if isinstance(message, str) else {}
        except Exception:
            data = {}
        replies = self.on_end if data.get("commit") else self.on_audio
        while replies:
            item = replies.pop(0)
            if item == "|":
                break
            await self.inbox.put(item)

    async def close(self) -> None:
        self.closed = True
        await self.inbox.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.inbox.get()
        if item is None:
            raise StopAsyncIteration
        return json.dumps(item)


def interim(text):
    return {"message_type": "partial_transcript", "text": text}


def segment(text):
    return {"message_type": "committed_transcript", "text": text}


DONE = {"message_type": "committed_transcript", "text": ""}
VAD_START = {"message_type": "vad_start"}
VAD_END = {"message_type": "vad_end"}


def make_client(fake, **overrides):
    settings = Settings(_env_file=None, elevenlabs_api_key="test-key", tts_preload=False, **overrides)
    captured = {}

    async def connector(settings_, languages, mode, vocabulary):
        captured.update(languages=languages, mode=mode, vocabulary=vocabulary)
        return fake

    app = create_app(settings, synthesizer=VieneuSynthesizer(settings, model=FakeVieneu()), live_connector=connector)
    return TestClient(app), captured


def receive_until(ws, kind):
    messages = []
    while True:
        message = ws.receive_json()
        messages.append(message)
        if message["type"] == kind:
            return messages


def test_live_turn_joins_segments_and_forwards_pcm():
    fake = FakeLive(on_audio=[VAD_START, interim("Em đã"), segment("Em đã triển khai FastAPI.")],
                    on_end=[segment("Chạy trên Kubernetes."), DONE, VAD_END])
    client, captured = make_client(fake)
    with client.websocket_connect("/v1/stt/stream") as ws:
        ws.send_json({"type": "start", "language": "vi-VN,en-US", "mode": "verbatim", "vocabulary": "FastAPI"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(b"\x01\x00" * 1600)
        messages = receive_until(ws, "interim")
        ws.send_json({"type": "end"})
        messages += receive_until(ws, "final")

    assert captured == {"languages": ["vi-VN", "en-US"], "mode": "verbatim", "vocabulary": ["FastAPI"]}
    # Audio is sent as base64 JSON chunk
    first_msg = json.loads(fake.sent[0])
    assert first_msg["message_type"] == "input_audio_chunk"
    assert "audio_base_64" in first_msg
    assert {"type": "vad", "event": "start"} in messages
    assert messages[-1]["text"] == "Em đã triển khai FastAPI. Chạy trên Kubernetes."
    assert fake.closed


def test_silence_only_turn_finalizes_immediately_with_empty_text():
    fake = FakeLive()
    client, _ = make_client(fake)
    with client.websocket_connect("/v1/stt/stream") as ws:
        ws.send_json({"type": "start"})
        ws.receive_json()
        ws.send_bytes(bytes(3200))
        ws.send_json({"type": "end"})
        final = ws.receive_json()
    assert final["type"] == "final" and final["text"] == ""


def test_final_falls_back_to_timeout_when_online_stays_silent():
    fake = FakeLive(on_audio=[VAD_START, interim("xin chào")])
    client, _ = make_client(fake, stt_live_final_timeout_sec=0.05)
    with client.websocket_connect("/v1/stt/stream") as ws:
        ws.send_json({"type": "start"})
        ws.receive_json()
        ws.send_bytes(bytes(3200))
        receive_until(ws, "interim")
        ws.send_json({"type": "end"})
        final = receive_until(ws, "final")[-1]
    assert final["text"] == "xin chào"


def test_session_carries_several_turns():
    fake = FakeLive(on_audio=[VAD_START, "|", VAD_START], on_end=[segment("Một."), DONE, "|", segment("Hai."), DONE])
    client, _ = make_client(fake)
    with client.websocket_connect("/v1/stt/stream") as ws:
        ws.send_json({"type": "start"})
        ws.receive_json()
        finals = []
        for _ in range(2):
            ws.send_bytes(bytes(3200))
            receive_until(ws, "vad")
            ws.send_json({"type": "end"})
            finals.append(receive_until(ws, "final")[-1]["text"])
    assert finals == ["Một.", "Hai."]


@pytest.mark.parametrize("first,overrides,detail", [
    ({"type": "end"}, {}, "start"),
    ({"type": "start"}, {"service_token": "s3cret"}, "token"),
    ({"type": "start", "token": "s3cret", "mode": "loud"}, {"service_token": "s3cret"}, "mode"),
])
def test_rejects_bad_start(first, overrides, detail):
    client, _ = make_client(FakeLive(), **overrides)
    with client.websocket_connect("/v1/stt/stream") as ws:
        ws.send_json(first)
        error = ws.receive_json()
    assert error["type"] == "error" and detail in error["detail"]


def test_rejects_odd_sized_pcm_frames():
    client, _ = make_client(FakeLive())
    with client.websocket_connect("/v1/stt/stream") as ws:
        ws.send_json({"type": "start"})
        ws.receive_json()
        ws.send_bytes(b"\x00\x00\x00")
        assert ws.receive_json()["type"] == "error"
