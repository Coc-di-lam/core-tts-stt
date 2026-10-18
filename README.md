# Speech Service — STT & TTS tiếng Việt

Service rời chỉ làm hai việc: **lời nói → chữ** và **chữ → lời nói**. Không có LLM hội thoại, không có
session, không có DB. Core AI (`dpl_pvan`) chỉ xử lý chữ; frontend hoặc gateway gọi service này
trước và sau core.

| Chức năng | Model | Chạy ở đâu |
|---|---|---|
| STT streaming | `gemini-3.5-transcribe-live` ([docs](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe)) | Google Live API, service làm relay WebSocket (API key không lộ ra client) |
| STT batch | `gemini-3.5-transcribe` ([docs](https://ai.google.dev/gemini-api/docs/transcribe)) | Google API, audio gửi inline |
| TTS | VieNeu-TTS v3 Turbo, 48 kHz | Local, tự chọn thiết bị: có GPU NVIDIA → PyTorch/CUDA; không có → ONNX Runtime int8 trên CPU |

```
Browser (mic, PCM 16 kHz) ──WS /v1/stt/stream──▶ interim… final ──▶ Browser/Gateway ──candidate_answer──▶ core (WS)
core ──response_token──▶ Gateway (cắt theo câu) ──▶ POST /v1/tts/stream ──PCM từng đoạn──▶ Browser
```

## Chạy local

```powershell
copy .env.example .env      # điền GEMINI_API_KEY
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8010
```

Mở `http://127.0.0.1:8010/` để dùng trang test: bấm nói → STT live (hiện chữ tạm trong lúc nói) → TTS →
phát lại, kèm thời gian từng bước; tùy chọn tự dừng khi Gemini thấy hết nói. Có form riêng cho
`/v1/stt` (upload file) và `/v1/tts`. Lần chạy đầu tải weight VieNeu
từ Hugging Face; `/health` báo `tts.state` = `loading` cho tới khi sẵn sàng.

Docker: `docker compose up --build` (cổng `127.0.0.1:8010`, weight cache trong volume `models`). Image
mặc định là bản CPU (không có torch). Muốn dùng GPU trong container thì cần image có `--extra cuda`, CUDA
runtime và `gpus: all`; image hiện tại chưa làm phần này.

Trên Windows không có GPU, nên chạy TTS trực tiếp (`uv run …`) thay vì Docker Desktop: đo trên cùng máy,
trong container (VM WSL2, 8 vCPU dùng chung với các container khác) TTS chậm hơn ~3–4 lần (âm đầu
~0,6–0,8 s, RTF ~1,4). Trên server Linux chạy Docker trực tiếp thì không bị khoản chậm này.

## API

`/health` và `/` luôn mở. Khi đặt `SERVICE_TOKEN`, mọi `/v1/*` cần `Authorization: Bearer <token>`.

### `WS /v1/stt/stream` — STT streaming (khuyên dùng cho hội thoại)

Client gửi:

| Tin nhắn | Mô tả |
|---|---|
| `{"type": "start", "language"?, "mode"?, "vocabulary"?, "token"?}` | Bắt buộc, tin đầu tiên. Ý nghĩa field như `/v1/stt`; `token` khi có `SERVICE_TOKEN` (WebSocket trình duyệt không gửi được header) |
| frame nhị phân | PCM 16-bit little-endian, 16 kHz, mono; nên ~100 ms (3200 byte) mỗi frame, tối đa 64 KB |
| `{"type": "end"}` | Người nói xong lượt này → service trả đúng một `final` |
| `{"type": "stop"}` | Đóng phiên (hoặc đóng socket) |

Service gửi: `ready` (`setup_ms`), `interim` (`text` = toàn bộ chữ của lượt hiện tại, cập nhật liên tục),
`vad` (`event`: `start`/`end`, VAD phía Gemini), `final` (`text`, `latency_ms` tính từ lúc nhận `end`),
`error` (`detail`).

- Một phiên chạy được nhiều lượt; Gemini giới hạn phiên live ~10 phút, hết thì mở phiên mới.
- Gemini tự cắt câu khi người nói ngừng giữa chừng; service gộp các đoạn lại tới khi nhận `end`.
- Lượt chỉ có im lặng trả `final` với `text` rỗng ngay lập tức. Nếu Gemini không trả đoạn cuối trong
  `STT_LIVE_FINAL_TIMEOUT_SEC`, service chốt bằng phần chữ đang có.
- Muốn rảnh tay: chờ `vad` `end` rồi gửi `end` (trang test có tùy chọn này).

### `POST /v1/stt` — STT batch, multipart

| Field | Bắt buộc | Mô tả |
|---|---|---|
| `file` | có | WAV, MP3, OGG, FLAC, M4A, AAC, AIFF, Opus, WebM (MediaRecorder của trình duyệt). Tối đa 15 MB |
| `language` | không | BCP-47, phân tách bằng dấu phẩy (`vi-VN,en-US`); `auto` để tự nhận diện. Mặc định `STT_LANGUAGE_CODES` |
| `mode` | không | `smart` (bỏ ậm ừ, tự định dạng) hoặc `verbatim`. Mặc định `STT_MODE` |
| `vocabulary` | không | Thuật ngữ cần nhận đúng, phân tách bằng dấu phẩy (`Kubernetes,FastAPI`) |

```json
{"text": "Em đã triển khai hệ thống bằng FastAPI và Redis.", "model": "gemini-3.5-transcribe",
 "language_codes": ["vi-VN"], "mode": "smart", "latency_ms": 3684.3}
```

### `POST /v1/tts` — JSON `{"text": "...", "voice": "Mai Anh"}`

Trả `audio/wav` (48 kHz, PCM 16-bit, có 150 ms im lặng đầu và 80 ms cuối để loa không nuốt âm đầu).
Header: `X-Audio-Duration` (giây, không tính phần đệm), `X-Latency-Ms`, `X-Voice` (URL-encoded),
`X-Cache-Hit`. Tối đa `TTS_MAX_CHARS` ký tự; nên gửi từng câu để giảm độ trễ.

### `POST /v1/tts/stream` — JSON như `/v1/tts` (khuyên dùng cho hội thoại)

Trả PCM thô `audio/pcm` (s16le, mono) theo từng đoạn ngay khi VieNeu sinh ra; header `X-Sample-Rate`
(48000 với Turbo, 24000 với Nano), `X-Channels`, `X-Sample-Format`, `X-Voice`. Có cùng phần đệm im lặng
như `/v1/tts`. Lỗi kiểm tra đầu vào (400/413/422/503) trả về trước khi stream bắt đầu. Câu đã sinh được
cache, lần sau trả ngay. Client phát bằng Web Audio, xếp lịch từng đoạn nối tiếp nhau (xem `ttsStream`
trong `app/static/index.html`).

### `GET /v1/tts/voices`

Danh sách giọng preset của VieNeu (`id` dùng cho trường `voice`).

### Mã lỗi

`400` audio/văn bản rỗng · `401` sai token · `413` vượt giới hạn · `415` định dạng audio không hỗ trợ ·
`422` giọng/mode không hợp lệ · `502`/`504` Gemini lỗi hoặc quá thời gian · `503` thiếu
`GEMINI_API_KEY` hoặc VieNeu không tải được.

## TTS trên CPU và GPU

`TTS_DEVICE=auto` (mặc định): nếu cài PyTorch có CUDA (`uv sync --extra cuda`) và thấy GPU NVIDIA thì
chạy như trước (PyTorch, batching, stream scheduler cho nhiều luồng cùng lúc); nếu không thì chạy ONNX
Runtime trên CPU với các tối ưu:

- **int8** (`TTS_CPU_PRECISION`): nhanh hơn fp32 ~1,5 lần, chất lượng như nhau. Cần CPU có VNNI
  (Intel Ice Lake/Alder Lake trở lên, AMD Zen 4 trở lên); CPU cũ nghe méo thì đổi sang `fp32`.
- **Streaming** (`/v1/tts/stream`): âm thanh đầu tiên sẵn sau ~0,2–0,35 s thay vì chờ cả câu.
- **Warm-up** sau khi tải model, nên request đầu không chậm hơn các request sau.
- **Cache** các câu đã sinh (128 câu).
- CPU yếu: `TTS_MODEL=vieneu-v3-nano` sinh cả câu nhanh hơn (RTF 0,19) nhưng âm đầu tới muộn
  (~0,9 s, không stream thật), 24 kHz, bộ giọng riêng.

Đo trên Ryzen 7 8845HS (8 nhân, không GPU), 3 câu phỏng vấn ~12–14 s audio. WER là tỉ lệ sai từ khi cho
Gemini transcribe lại audio:

| Cấu hình | Thời gian / độ dài audio (RTF) | Âm đầu tiên (stream) | WER |
|---|---|---|---|
| Turbo fp32 | 0,53 | ~420 ms | 1,8% |
| **Turbo int8 (mặc định)** | **0,34** | **~180–190 ms** | 1,8% |
| Turbo int8, 4 luồng | 0,34 | ~185 ms | 1,8% |
| Nano | 0,19 | ~900 ms | 1,8% |

## Hiệu năng đo được (CPU, không GPU)

- TTS: tải model + warm-up ~10 s (lần đầu lâu hơn do tải weight); stream qua HTTP: âm đầu ~250–360 ms,
  cả câu 4–6 s audio xong trong ~1,6–2,8 s.
- STT live: mở phiên ~0,5–0,6 s; `final` về **~250–300 ms** sau `end` (hoặc gần như ngay khi chờ `vad` `end`).
- STT batch: ~3,7–4,5 s mỗi lượt cho câu ~4 s audio, phần lớn là thời gian gọi Gemini.
- Vòng đầy đủ trên trang test (dừng nói → nghe tiếng TTS): **~0,24 s** với TTS streaming (~1,7 s nếu chờ cả WAV).

## Kiểm thử

```powershell
uv run pytest
```

Test dùng Gemini giả (`httpx.MockTransport`, Live API giả) và VieNeu giả, không cần mạng hay weight.

## Cấu hình

Xem `.env.example`. Các biến chính: `GEMINI_API_KEY`, `STT_LIVE_MODEL`, `STT_LIVE_FINAL_TIMEOUT_SEC`, `STT_MODEL`, `STT_LANGUAGE_CODES`, `STT_MODE`,
`TTS_MODEL`, `TTS_VOICE`, `TTS_DEVICE` (`auto` | `cpu` | `cuda`), `TTS_CPU_PRECISION`, `TTS_CPU_THREADS`, `TTS_PRELOAD`,
`SERVICE_TOKEN`, `CORS_ORIGINS`.
