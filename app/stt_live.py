"""Streaming speech-to-text: relay a client WebSocket to ElevenLabs Scribe v2 Realtime (or local Gipformer).

Client protocol (`WS /v1/stt/stream`):
  -> {"type": "start", "language"?, "mode"?, "vocabulary"?, "fluency"?, "token"?}   first message, required
  -> binary frames: raw PCM 16-bit little-endian, 16 kHz, mono (~100 ms per frame)
  -> {"type": "end"}    the speaker finished this turn; the service answers with one "final"
  -> {"type": "stop"}   close the session (or just close the socket)
  <- {"type": "ready", "model", "setup_ms"}
  <- {"type": "interim", "text"}   running text of the current turn, revised while speaking
  <- {"type": "vad", "event": "start" | "end"}   voice activity
  <- {"type": "final", "text", "latency_ms"}   whole turn; latency counts from "end"
     with "fluency": true also "verbatim" (raw, with fillers) and "fluency" (app/fluency.py metrics);
     mode runs verbatim and "text" is the transcript without fillers
  <- {"type": "error", "detail"}
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
from urllib.parse import urlencode

from fastapi import WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedOK

from app import fluency
from app.config import Settings, split_csv
from app.stt import _map_language_code

logger = logging.getLogger(__name__)

MAX_FRAME_BYTES = 64 * 1024
MAX_TURN_PCM_BYTES = 16_000 * 2 * 600  # keep up to 10 min of a turn for fluency analysis


class Upstream(Protocol):
    async def send(self, message: str | bytes) -> None: ...
    async def close(self) -> None: ...
    def __aiter__(self): ...


Connector = Callable[[Settings, list[str], str, list[str]], Awaitable[Upstream]]


class LiveSetupError(Exception):
    pass


async def connect_elevenlabs_live(settings: Settings, language_codes: list[str], mode: str,
                                  vocabulary: list[str]) -> Upstream:
    from websockets.asyncio.client import connect

    lang = _map_language_code(language_codes) or settings.stt_language_code
    params = {
        "model_id": settings.stt_live_model,
        "audio_format": "pcm_16000",
    }
    if lang:
        params["language_code"] = lang

    url = f"{settings.stt_live_ws_url}?{urlencode(params)}"
    headers = {"xi-api-key": settings.elevenlabs_api_key}

    ws = await connect(url, additional_headers=headers, max_size=None,
                       open_timeout=settings.stt_live_setup_timeout_sec)
    try:
        reply_raw = await asyncio.wait_for(ws.recv(), settings.stt_live_setup_timeout_sec)
        reply = json.loads(reply_raw)
    except BaseException:
        await ws.close()
        raise

    msg_type = reply.get("message_type")
    if msg_type != "session_started":
        await ws.close()
        raise LiveSetupError(f"ElevenLabs live setup failed: {reply.get('error') or reply}")

    return ws


# Backward-compatible alias
connect_gemini_live = connect_elevenlabs_live


class _Turn:
    """Accumulates transcription segments until the client ends the turn."""

    def __init__(self) -> None:
        self.segments: list[str] = []
        self.interim = ""
        self.speaking = False
        self.ended_at: float | None = None
        self.pcm = bytearray()
        self.frames_received = 0

    def text(self) -> str:
        return " ".join(part for part in [*self.segments, self.interim] if part).strip()

    def reset(self) -> None:
        self.__init__()


async def relay_stream(client: WebSocket, settings: Settings, connector: Connector | None = None) -> None:
    connector = connector or connect_elevenlabs_live
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

    backend = settings.stt_backend.strip().lower()
    if backend in ("gipformer", "auto"):
        try:
            from app.stt_gipformer import GipformerTranscriber
            gip = GipformerTranscriber(settings)
            if gip.is_available():
                await relay_stream_gipformer(client, settings, start, gip)
                return
            if backend == "gipformer":
                await _close(client, 1011, "Gipformer STT chưa sẵn sàng hoặc chưa nạp xong.")
                return
        except Exception as exc:
            if backend == "gipformer":
                await _close(client, 1011, f"Gipformer không khả dụng: {exc}")
                return
            logger.warning("Gipformer live stream failed (%s), falling back to ElevenLabs Scribe", exc)

    if not settings.elevenlabs_api_key:
        await _close(client, 1011, "Chưa cấu hình ELEVENLABS_API_KEY.")
        return

    language = start.get("language")
    languages = settings.language_codes if language is None else (
        [] if str(language).strip().lower() == "auto" else split_csv(str(language)))
    mode = str(start.get("mode") or settings.stt_mode).strip().lower()
    if mode not in {"smart", "verbatim"}:
        await _close(client, 1008, "mode phải là 'smart' hoặc 'verbatim'.")
        return
    measure = bool(start.get("fluency"))
    if measure:
        mode = "verbatim"

    started = time.perf_counter()
    try:
        upstream = await connector(settings, languages, mode, split_csv(start.get("vocabulary")))
    except Exception as exc:
        logger.exception("ElevenLabs Live setup failed")
        await _close(client, 1011, f"Không mở được ElevenLabs Live: {type(exc).__name__}: {str(exc)[:200]}")
        return

    await client.send_json({
        "type": "ready",
        "model": settings.stt_live_model,
        "language_codes": languages,
        "mode": mode,
        "fluency": measure,
        "setup_ms": round((time.perf_counter() - started) * 1000, 1),
    })

    turn = _Turn()
    lock = asyncio.Lock()
    timer: asyncio.Task | None = None

    async def emit_final() -> None:
        nonlocal timer
        async with lock:
            if turn.ended_at is None:
                return
            message = {"type": "final", "text": turn.text()}
            if measure:
                clean, metrics = fluency.analyze(bytes(turn.pcm), turn.text())
                message.update(text=clean, verbatim=turn.text(), fluency=metrics)
            message["latency_ms"] = round((time.perf_counter() - turn.ended_at) * 1000, 1)
            await client.send_json(message)
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
                    turn.frames_received += 1
                    if measure and len(turn.pcm) < MAX_TURN_PCM_BYTES:
                        turn.pcm.extend(frame)
                    b64 = base64.b64encode(frame).decode("ascii")
                    await upstream.send(json.dumps({"message_type": "input_audio_chunk", "audio_base_64": b64}))
                continue

            try:
                command = json.loads(message.get("text") or "{}").get("type")
            except (ValueError, AttributeError):
                command = None

            if command == "stop":
                return
            if command == "end" and turn.ended_at is None:
                turn.ended_at = time.perf_counter()
                with contextlib.suppress(Exception):
                    await upstream.send(json.dumps({"message_type": "input_audio_chunk", "audio_base_64": "", "commit": True}))
                if turn.frames_received == 0:
                    await emit_final()
                else:
                    timer = asyncio.create_task(final_after_timeout())
            elif command not in {"end"}:
                await client.send_json({"type": "error", "detail": f"Lệnh không hợp lệ: {command!r}"})

    async def pump_upstream() -> None:
        async for raw in upstream:
            try:
                data = json.loads(raw)
            except Exception:
                continue

            msg_type = data.get("message_type") or data.get("type")

            # Partial transcript: interim update while user speaks
            if msg_type in ("partial_transcript", "interim"):
                text = (data.get("text") or data.get("transcript") or "").strip()
                turn.interim = text
                if not turn.speaking and text:
                    turn.speaking = True
                    await client.send_json({"type": "vad", "event": "start"})
                await client.send_json({"type": "interim", "text": turn.text()})

            # Committed transcript: final segment
            elif msg_type in ("committed_transcript", "final_transcript", "final_segment"):
                segment = (data.get("text") or data.get("transcript") or "").strip()
                if segment:
                    turn.segments.append(segment)
                    turn.interim = ""
                elif not turn.segments and turn.interim:
                    turn.segments.append(turn.interim)
                    turn.interim = ""
                else:
                    turn.interim = ""
                await client.send_json({"type": "interim", "text": turn.text()})
                if turn.ended_at is not None:
                    await emit_final()

            # Voice activity detection (if model provides VAD events)
            elif msg_type in ("vad_start", "speech_started"):
                turn.speaking = True
                await client.send_json({"type": "vad", "event": "start"})
            elif msg_type in ("vad_end", "speech_ended"):
                turn.speaking = False
                await client.send_json({"type": "vad", "event": "end"})

            elif msg_type in ("error", "invalid_audio"):
                detail = str(data.get("message") or data.get("detail") or data)[:300]
                await client.send_json({"type": "error", "detail": detail})

    tasks = [asyncio.create_task(pump_client()), asyncio.create_task(pump_upstream())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            exc = task.exception()
            if exc and not isinstance(exc, (WebSocketDisconnect, ConnectionClosedOK)):
                logger.warning("Live STT relay ended: %r", exc)
                with contextlib.suppress(Exception):
                    await client.send_json({"type": "error", "detail": f"Phiên live lỗi: {type(exc).__name__}"})
        if tasks[1] in done:
            with contextlib.suppress(Exception):
                await emit_final()
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


async def relay_stream_gipformer(client: WebSocket, settings: Settings, start: dict, gipformer: Any) -> None:
    """Live streaming transcription relay using local Gipformer ASR."""
    languages = ["vi-VN"]
    mode = str(start.get("mode") or settings.stt_mode).strip().lower()
    measure = bool(start.get("fluency"))
    started = time.perf_counter()

    await client.send_json({
        "type": "ready",
        "model": gipformer.model_name,
        "language_codes": languages,
        "mode": mode,
        "fluency": measure,
        "setup_ms": round((time.perf_counter() - started) * 1000, 1),
    })

    pcm_buffer = bytearray()
    last_interim_len = 0
    interim_lock = asyncio.Lock()

    async def update_interim() -> None:
        nonlocal last_interim_len
        async with interim_lock:
            current_len = len(pcm_buffer)
            if current_len < 16000 or current_len - last_interim_len < 12800:
                return
            last_interim_len = current_len
            buf = bytes(pcm_buffer)
        try:
            partial_text = await asyncio.to_thread(gipformer.transcribe_pcm, buf)
            if partial_text:
                await client.send_json({"type": "interim", "text": partial_text})
        except Exception:
            pass

    interim_task: asyncio.Task | None = None

    try:
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                frame = message["bytes"]
                if len(frame) > MAX_FRAME_BYTES or len(frame) % 2:
                    await client.send_json({"type": "error", "detail": "Frame audio phải là PCM16 và ≤ 64 KB."})
                    continue
                pcm_buffer.extend(frame)
                if interim_task is None or interim_task.done():
                    interim_task = asyncio.create_task(update_interim())
                continue

            try:
                command = json.loads(message.get("text") or "{}").get("type")
            except (ValueError, AttributeError):
                command = None

            if command == "stop":
                return
            if command == "end":
                ended_at = time.perf_counter()
                if interim_task and not interim_task.done():
                    interim_task.cancel()
                buf = bytes(pcm_buffer)
                pcm_buffer.clear()
                last_interim_len = 0
                if not buf:
                    await client.send_json({"type": "final", "text": "", "latency_ms": 0.0})
                    continue

                verbatim = await asyncio.to_thread(gipformer.transcribe_pcm, buf)
                result = {"type": "final", "text": verbatim}
                if measure:
                    clean, metrics = fluency.analyze(buf, verbatim)
                    result.update(text=clean, verbatim=verbatim, fluency=metrics)
                result["latency_ms"] = round((time.perf_counter() - ended_at) * 1000, 1)
                await client.send_json(result)
            elif command not in {"end"}:
                await client.send_json({"type": "error", "detail": f"Lệnh không hợp lệ: {command!r}"})
    finally:
        if interim_task and not interim_task.done():
            interim_task.cancel()
        with contextlib.suppress(Exception):
            await client.close()
