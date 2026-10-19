"""Service settings, read from the environment and the repo-root .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # STT backend: "auto" = Gipformer local (if available), fallback to Gemini | "gipformer" | "gemini"
    stt_backend: str = "auto"

    # STT: local Gipformer (g-group-ai-lab/gipformer1.5-68M-rnnt, Zipformer RNN-T)
    gipformer_version: str = "1.5"  # "1.5" (latest 68M) | "1"
    gipformer_quantize: str = "fp32"  # "fp32" (max setting, full precision) | "int8"
    gipformer_decoding_method: str = "modified_beam_search"  # "modified_beam_search" (max setting) | "greedy_search"
    gipformer_model_dir: str = ""  # empty = HuggingFace cache
    gipformer_num_threads: int = 4
    gipformer_preload: bool = True
    stt_fallback_gemini: bool = True  # Automatically fallback to Gemini STT if Gipformer is unavailable/fails

    # STT (batch): Gemini transcription API
    gemini_api_key: str = ""
    gemini_api_url: str = "https://generativelanguage.googleapis.com/v1beta/interactions"
    stt_model: str = "gemini-3.5-transcribe"
    stt_language_codes: str = "vi-VN"  # comma-separated BCP-47; empty = auto-detect
    stt_mode: str = "smart"  # "smart" | "verbatim"
    stt_timeout_sec: float = 60.0
    stt_max_bytes: int = 15 * 1024 * 1024  # inline audio; base64 adds ~33% to the request

    # Streaming STT: Gemini Live API
    gemini_live_url: str = ("wss://generativelanguage.googleapis.com/ws/"
                            "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")
    stt_live_model: str = "gemini-3.5-transcribe-live"
    stt_live_setup_timeout_sec: float = 10.0
    stt_live_final_timeout_sec: float = 3.0  # wait this long after "end" for Gemini's last segment

    # TTS backend: "auto" = VieNeu on an NVIDIA GPU (max settings), else Gemini TTS | "vieneu" | "gemini"
    tts_backend: str = "auto"
    tts_fallback_gemini: bool = True  # Automatically fallback to Gemini TTS if VieNeu fails

    # TTS: local VieNeu (ONNX Runtime on CPU, PyTorch on CUDA)
    tts_model: str = "vieneu-v3-turbo"  # or "vieneu-v3-nano" for weak CPUs (24 kHz, own voice list)
    tts_voice: str = "Trúc Ly"  # young female, natural Northern style (also in vieneu-v3-nano)
    tts_device: str = "auto"  # auto = cuda when PyTorch sees a GPU, else cpu
    tts_cpu_precision: str = "fp32"  # "fp32" (best quality) | "int8" (~1.5x faster, needs a VNNI CPU)
    tts_cpu_threads: int = 0  # 0 = engine default (~physical cores, max 8)
    tts_gpu_max_batch_size: int = 32
    tts_gpu_max_streams: int = 16
    tts_gpu_dtype: str = "float32"  # "float32" (best quality) | "bfloat16" | "float16"
    tts_preload: bool = True
    tts_max_chars: int = 2000
    tts_temperature: float = 0.8
    tts_top_k: int = 25
    tts_top_p: float = 0.95
    tts_repetition_penalty: float = 1.2
    tts_apply_watermark: bool = False  # False = purest uncompressed audio without acoustic watermark

    # TTS: Gemini (cloud), same GEMINI_API_KEY and Interactions endpoint as STT
    gemini_tts_model: str = "gemini-3.8-flash-lite-tts"
    gemini_tts_voice: str = "Leda"  # youthful female
    gemini_tts_style: str = "young, warm and friendly female interviewer"  # delivery instruction; empty = none
    gemini_tts_timeout_sec: float = 60.0

    # Service
    service_token: str = ""  # when set, /v1/* requires "Authorization: Bearer <token>"
    cors_origins: str = "*"

    @property
    def language_codes(self) -> list[str]:
        return split_csv(self.stt_language_codes)


def split_csv(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
