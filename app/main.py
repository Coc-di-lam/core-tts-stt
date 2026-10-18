"""Speech service: STT (Gemini transcribe API) + TTS (local VieNeu), plus a small test page."""
from __future__ import annotations

import asyncio
import hmac
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings, split_csv
from app.stt import GeminiTranscriber, SttError, resolve_mime_type
from app.stt_live import Connector, connect_gemini_live, relay_stream
from app.tts import TtsError, VieneuSynthesizer

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


class TtsRequest(BaseModel):
    text: str = Field(min_length=1)
    voice: str | None = None


class SttResponse(BaseModel):
    text: str
    model: str
    language_codes: list[str]
    mode: str
    latency_ms: float


def create_app(settings: Settings | None = None, transcriber: GeminiTranscriber | None = None,
               synthesizer: VieneuSynthesizer | None = None, live_connector: Connector | None = None) -> FastAPI:
    settings = settings or get_settings()
    transcriber = transcriber or GeminiTranscriber(settings)
    synthesizer = synthesizer or VieneuSynthesizer(settings)
    live_connector = live_connector or connect_gemini_live

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.tts_preload and synthesizer.state == "idle":
            # Load in the background so /health answers while the model warms up.
            threading.Thread(target=_preload, args=(synthesizer,), daemon=True).start()
        yield
        await transcriber.aclose()

    app = FastAPI(title="Speech Service", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=split_csv(settings.cors_origins) or ["*"],
                       allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["X-Audio-Duration", "X-Latency-Ms", "X-Voice", "X-Cache-Hit",
                                      "X-Sample-Rate", "X-Channels", "X-Sample-Format"])

    def require_token(authorization: str | None = Header(default=None)) -> None:
        if not settings.service_token:
            return
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip(), settings.service_token):
            raise HTTPException(status_code=401, detail="Thiếu hoặc sai service token.")

    @app.exception_handler(SttError)
    @app.exception_handler(TtsError)
    async def speech_error(_: Request, exc: SttError | TtsError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.get("/", include_in_schema=False)
    async def test_page():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health():
        return {
            "status": "ok" if transcriber.configured and synthesizer.state != "error" else "degraded",
            "auth_required": bool(settings.service_token),
            "stt": {"provider": "gemini", "model": settings.stt_model, "live_model": settings.stt_live_model,
                    "configured": transcriber.configured,
                    "language_codes": settings.language_codes, "mode": settings.stt_mode},
            "tts": {"provider": "vieneu", "model": settings.tts_model, "device": synthesizer.device,
                    "backend": synthesizer.backend, "cpu_precision": settings.tts_cpu_precision,
                    "state": synthesizer.state, "error": synthesizer.error,
                    "default_voice": synthesizer.default_voice},
        }

    @app.post("/v1/stt", response_model=SttResponse, dependencies=[Depends(require_token)])
    async def speech_to_text(file: UploadFile = File(...), language: str | None = Form(default=None),
                             mode: str | None = Form(default=None), vocabulary: str | None = Form(default=None)):
        """Transcribe one utterance. `language`: comma-separated BCP-47 codes, "auto" to detect."""
        mime_type = resolve_mime_type(file.content_type, file.filename)
        audio = await file.read(settings.stt_max_bytes + 1)
        languages = None if language is None else ([] if language.strip().lower() == "auto" else split_csv(language))
        result = await transcriber.transcribe(audio, mime_type, languages, mode, split_csv(vocabulary))
        logger.info("STT %s %d bytes -> %d chars in %.0f ms", mime_type, len(audio), len(result.text), result.latency_ms)
        return SttResponse(**result.__dict__)

    @app.websocket("/v1/stt/stream")
    async def speech_to_text_stream(websocket: WebSocket):
        """Live transcription; protocol in app/stt_live.py. The token goes in the "start" message."""
        await relay_stream(websocket, settings, live_connector)

    @app.post("/v1/tts", dependencies=[Depends(require_token)],
              responses={200: {"content": {"audio/wav": {}}}})
    async def text_to_speech(request: TtsRequest):
        speech = await asyncio.to_thread(synthesizer.synthesize, request.text, request.voice)
        logger.info("TTS %d chars -> %.2fs audio in %.0f ms", len(request.text), speech.duration_sec, speech.latency_ms)
        return Response(content=speech.wav, media_type="audio/wav", headers={
            "X-Audio-Duration": str(speech.duration_sec), "X-Latency-Ms": str(speech.latency_ms),
            "X-Voice": quote(speech.voice), "X-Cache-Hit": str(speech.cache_hit).lower(),
        })

    @app.post("/v1/tts/stream", dependencies=[Depends(require_token)],
              responses={200: {"content": {"audio/pcm": {}}}})
    async def text_to_speech_stream(request: TtsRequest):
        """Raw PCM s16le mono chunks while VieNeu generates; sample rate in X-Sample-Rate."""
        cancel = threading.Event()
        voice = request.voice or synthesizer.default_voice
        chunks = _iterate_in_thread(lambda: synthesizer.stream(request.text, request.voice, cancel))
        first = await anext(chunks)  # validation/load errors surface here as HTTP errors

        async def body():
            try:
                yield first
                async for chunk in chunks:
                    yield chunk
            finally:
                cancel.set()

        return StreamingResponse(body(), media_type="audio/pcm", headers={
            "X-Sample-Rate": str(synthesizer.sample_rate), "X-Channels": "1", "X-Sample-Format": "s16le",
            "X-Voice": quote(voice), "Cache-Control": "no-store",
        })

    @app.get("/v1/tts/voices", dependencies=[Depends(require_token)])
    async def list_voices():
        voices = await asyncio.to_thread(synthesizer.voices)
        return {"default": synthesizer.default_voice, "voices": voices}

    return app


async def _iterate_in_thread(make_iterator):
    """Drive a blocking iterator in a worker thread and yield its items asynchronously."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    def worker():
        try:
            for item in make_iterator():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except BaseException as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    threading.Thread(target=worker, daemon=True).start()
    while (item := await queue.get()) is not done:
        if isinstance(item, BaseException):
            raise item
        yield item


def _preload(synthesizer: VieneuSynthesizer) -> None:
    try:
        synthesizer.load()
    except TtsError:
        pass  # state/error are recorded on the synthesizer and surfaced by /health


app = create_app()
