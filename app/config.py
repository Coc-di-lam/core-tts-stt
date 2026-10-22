"""Service settings, read from the environment and the repo-root .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # ElevenLabs credentials
    elevenlabs_api_key: str = ""
    elevenlabs_api_url: str = "https://api.elevenlabs.io"

    # STT backend: "auto" = Gipformer local (if available), fallback to ElevenLabs | "gipformer" | "elevenlabs"
    stt_backend: str = "elevenlabs"
    stt_fallback_online: bool = True  # Automatically fallback to ElevenLabs STT if Gipformer is unavailable/fails

    # STT: local Gipformer (g-group-ai-lab/gipformer1.5-68M-rnnt, Zipformer RNN-T)
    gipformer_version: str = "1.5"  # "1.5" (latest 68M) | "1"
    gipformer_quantize: str = "fp32"  # "fp32" (max setting, full precision) | "int8"
    gipformer_decoding_method: str = "modified_beam_search"  # "modified_beam_search" (max setting) | "greedy_search"
    gipformer_model_dir: str = ""  # empty = HuggingFace cache
    gipformer_num_threads: int = 4
    gipformer_preload: bool = True

    # STT (batch): ElevenLabs Scribe v2 API
    stt_model: str = "scribe_v2"
    stt_language_code: str = "vie"  # ISO-639-3 or ISO-639-1: "vie" or empty for auto-detect
    stt_language_codes: str = "vi-VN"  # comma-separated BCP-47 for backward compatibility
    stt_mode: str = "smart"  # "smart" | "verbatim"
    stt_timeout_sec: float = 60.0
    stt_max_bytes: int = 25 * 1024 * 1024

    # Streaming STT: ElevenLabs Scribe v2 Realtime WebSocket API
    stt_live_ws_url: str = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
    stt_live_model: str = "scribe_v2_realtime"
    stt_live_setup_timeout_sec: float = 10.0
    stt_live_final_timeout_sec: float = 3.0  # wait this long after "end" for last segment

    # TTS backend: "auto" = VieNeu on an NVIDIA GPU, else ElevenLabs v4 Turbo | "vieneu" | "elevenlabs"
    tts_backend: str = "elevenlabs"
    tts_fallback_online: bool = True  # Automatically fallback to ElevenLabs TTS if VieNeu fails

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
    tts_apply_watermark: bool = False

    # TTS: ElevenLabs v4 Turbo (cloud)
    elevenlabs_tts_model: str = "eleven_v4_turbo"
    elevenlabs_tts_voice: str = "A5w1fw5x0uXded1LDvZp"  # Nhu - Calm and Confident (professional female interviewer)
    elevenlabs_tts_output_format: str = "pcm_48000"  # pcm_48000 (48 kHz) | pcm_24000
    elevenlabs_tts_sample_rate: int = 48000
    elevenlabs_tts_timeout_sec: float = 60.0

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
