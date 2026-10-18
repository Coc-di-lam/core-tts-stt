"""Streaming speech-to-text: relay a client WebSocket to the Gemini Live transcription model.

Client protocol (`WS /v1/stt/stream`):
  -> {"type": "start", "language"?, "mode"?, "vocabulary"?, "token"?}   first message, required
  -> binary frames: raw PCM 16-bit little-endian, 16 kHz, mono (~100 ms per frame)
  -> {"type": "end"}    the speaker finished this turn; the service answers with one "final"
  -> {"type": "stop"}   close the session (or just close the socket)
  <- {"type": "ready", "model", "setup_ms"}
  <- {"type": "interim", "text"}   running text of the current turn, revised while speaking
  <- {"type": "vad", "event": "start" | "end"}   Gemini server-side voice activity
  <- {"type": "final", "text", "latency_ms"}   whole turn; latency counts from "end"
  <- {"type": "error", "detail"}
One session can carry several turns; Gemini caps a live session at ~10 minutes.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import hmac
import json
import logging
import time
from typing import Any, Awaitable, Callable, Protocol

from fastapi import WebSocket, WebSocketDisconnect

from app.config import Settings, split_csv

logger = logging.getLogger(__name__)

PCM_MIME_TYPE = "audio/pcm;rate=16000"
MAX_FRAME_BYTES = 64 * 1024


class Upstream(Protocol):
    async def send(self, message: str) -> None: ...
    async def close(self) -> None: ...
    def __aiter__(self): ...


Connector = Callable[[Settings, list[str], str, list[str]], Awaitable[Upstream]]


class LiveSetupError(Exception):
    pass


async def connect_gemini_live(settings: Settings, language_codes: list[str], mode: str,
                              vocabulary: list[str]) -> Upstream:
    from websockets.asyncio.client import connect

    transcription: dict[str, Any] = {"languageCodes": language_codes}
    if mode == "smart":
        transcription["mode"] = "SMART"  # Gemini's default is verbatim
    if vocabulary:
        transcription["customVocabulary"] = vocabulary[:1000]
    ws = await connect(settings.gemini_live_url, additional_headers={"x-goog-api-key": settings.gemini_api_key},
                       max_size=None, open_timeout=settings.stt_live_setup_timeout_sec)
    try:
        await ws.send(json.dumps({"setup": {
            "model": f"models/{settings.stt_live_model}",
            "generationConfig": {"responseModalities": ["TEXT"]},
            "inputAudioTranscription": transcription,
        }}))
        reply = json.loads(await asyncio.wait_for(ws.recv(), settings.stt_live_setup_timeout_sec))
    except BaseException:
        await ws.close()
        raise
    if "setupComplete" not in reply:
        await ws.close()
        raise LiveSetupError(str(reply.get("error") or reply)[:300])
    return ws


class _Turn:
    """Accumulates Gemini segments until the client ends the turn."""

    def __init__(self) -> None:
        self.segments: list[str] = []
        self.interim = ""
        self.speaking = False
        self.ended_at: float | None = None

    def text(self) -> str:
        return " ".join(part for part in [*self.segments, self.interim] if part).strip()

    def reset(self) -> None:
        self.__init__()


async def relay_stream(client: WebSocket, settings: Settings, connector: Connector) -> None:
    await client.accept()
    try:
        start = await asyncio.wait_for(client.receive_json(), settings.stt_live_setup_timeout_sec)
    except (asyncio.TimeoutError, ValueError, KeyError, WebSocketDisconnect):
        await _close(client, 1008, "Tin nhắn đầu tiên phải là {\"type\": \"start\"}.")
        return
    if not isinstance(start, dict) or start.get("type") != "start":
        await _close(client, 1008, "Tin nhắn đầu tiên phải là {\"type\": \"start\"}.")
        return
    if settings.service_token and not hmac.compare_digest(str(start.get("token") or ""), settings.service_token):
        await _close(client, 1008, "Thiếu hoặc sai service token.")
        return
    if not settings.gemini_api_key:
        await _close(client, 1011, "Chưa cấu hình GEMINI_API_KEY.")
        return
    language = start.get("language")
    languages = settings.language_codes if language is None else (
        [] if str(language).strip().lower() == "auto" else split_csv(str(language)))
    mode = str(start.get("mode") or settings.stt_mode).strip().lower()
    if mode not in {"smart", "verbatim"}:
        await _close(client, 1008, "mode phải là 'smart' hoặc 'verbatim'.")
        return

    started = time.perf_counter()
    try:
        upstream = await connector(settings, languages, mode, split_csv(start.get("vocabulary")))
    except Exception as exc:
        logger.exception("Gemini Live setup failed")
        await _close(client, 1011, f"Không mở được Gemini Live: {type(exc).__name__}: {str(exc)[:200]}")
        return
    await client.send_json({"type": "ready", "model": settings.stt_live_model,
                            "language_codes": languages, "mode": mode,
                            "setup_ms": round((time.perf_counter() - started) * 1000, 1)})

    turn = _Turn()
    lock = asyncio.Lock()  # serializes "final" between the upstream pump and the end-of-turn timer
    timer: asyncio.Task | None = None

    async def emit_final() -> None:
        nonlocal timer
        async with lock:
            if turn.ended_at is None:
                return
            latency = round((time.perf_counter() - turn.ended_at) * 1000, 1)
            await client.send_json({"type": "final", "text": turn.text(), "latency_ms": latency})
            turn.reset()
        if timer is not None and timer is not asyncio.current_task():
            timer.cancel()
        timer = None

    async def final_after_timeout() -> None:
        await asyncio.sleep(settings.stt_live_final_timeout_sec)
        await emit_final()

    async def pump_client() -> None:
        nonlocal timer
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                frame = message["bytes"]
                if len(frame) > MAX_FRAME_BYTES or len(frame) % 2:
                    await client.send_json({"type": "error", "detail": "Frame audio phải là PCM16 và ≤ 64 KB."})
                    continue
                if turn.ended_at is None and frame:
                    await upstream.send(json.dumps({"realtimeInput": {"audio": {
                        "data": base64.b64encode(frame).decode("ascii"), "mimeType": PCM_MIME_TYPE}}}))
                continue
            try:
                command = json.loads(message.get("text") or "{}").get("type")
            except (ValueError, AttributeError):
                command = None
            if command == "stop":
                return
            if command == "end" and turn.ended_at is None:
                turn.ended_at = time.perf_counter()
                await upstream.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
                if not turn.speaking and not turn.interim:
                    await emit_final()  # nothing pending: Gemini sends no reply for silence
                else:
                    timer = asyncio.create_task(final_after_timeout())
            elif command not in {"end"}:
                await client.send_json({"type": "error", "detail": f"Lệnh không hợp lệ: {command!r}"})

    async def pump_upstream() -> None:
        async for raw in upstream:
            data = json.loads(raw)
            content = data.get("serverContent") or {}
            if "interimInputTranscription" in content:
                turn.interim = (content["interimInputTranscription"].get("text") or "").strip()
                await client.send_json({"type": "interim", "text": turn.text()})
            if "inputTranscription" in content:
                segment = (content["inputTranscription"].get("text") or "").strip()
                turn.interim = ""
                if segment:
                    turn.segments.append(segment)
                await client.send_json({"type": "interim", "text": turn.text()})
            if content.get("generationComplete") and turn.ended_at is not None:
                await emit_final()
            activity = (data.get("voiceActivity") or {}).get("type")
            if activity in {"ACTIVITY_START", "ACTIVITY_END"}:
                turn.speaking = activity == "ACTIVITY_START"
                await client.send_json({"type": "vad", "event": "start" if turn.speaking else "end"})
            if "goAway" in data:
                await client.send_json({"type": "error", "detail": "Gemini sắp đóng phiên live (goAway)."})
            if "error" in data:
                await client.send_json({"type": "error", "detail": str(data["error"])[:300]})

    tasks = [asyncio.create_task(pump_client()), asyncio.create_task(pump_upstream())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.exception() and not isinstance(task.exception(), WebSocketDisconnect):
                logger.warning("Live STT relay ended: %r", task.exception())
                with contextlib.suppress(Exception):
                    await client.send_json({"type": "error", "detail": f"Phiên live lỗi: {type(task.exception()).__name__}"})
        if tasks[1] in done:
            with contextlib.suppress(Exception):
                await emit_final()  # Gemini closed the session; flush what we have
    finally:
        for task in [*tasks, timer]:
            if task is not None:
                task.cancel()
        with contextlib.suppress(Exception):
            await upstream.close()
        with contextlib.suppress(Exception):
            await client.close()


async def _close(client: WebSocket, code: int, detail: str) -> None:
    with contextlib.suppress(Exception):
        await client.send_json({"type": "error", "detail": detail})
        await client.close(code=code)
