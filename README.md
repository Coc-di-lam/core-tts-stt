# Speech Service — STT & TTS tiếng Việt

Service rời chỉ làm hai việc: **lời nói → chữ** và **chữ → lời nói**. Không có LLM hội thoại, không có
session, không có DB. Core AI (`dpl_pvan`) chỉ xử lý chữ; frontend hoặc gateway gọi service này
trước và sau core.

| Chức năng | Model | Chạy ở đâu |
|---|---|---|
| STT streaming | `scribe_v2_realtime (ElevenLabs Scribe v2)` ([docs](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe)) | Google Live API, service làm relay WebSocket (API key không lộ ra client) |
| STT batch | `scribe_v2 (ElevenLabs Scribe v2)` ([docs](https://ai.google.dev/gemini-api/docs/transcribe)) | Google API, audio gửi inline |
| TTS | VieNeu-TTS v3 Turbo, 48 kHz **hoặc** `gemini-3.8-flash-lite-tts`, 24 kHz ([docs](https://ai.google.dev/gemini-api/docs/speech-generation)) | `TTS_BACKEND=auto`: có GPU NVIDIA → VieNeu local (PyTorch/CUDA, fp32); không có → Gemini TTS. Xem [Chọn backend TTS](#chọn-backend-tts) |

```
Browser (mic, PCM 16 kHz) ──WS /v1/stt/stream──▶ interim… final ──▶ Browser/Gateway ──candidate_answer──▶ core (WS)
core ──response_token──▶ Gateway (cắt theo câu) ──▶ POST /v1/tts/stream ──PCM từng đoạn──▶ Browser
```

## Chạy local

```powershell
copy .env.example .env      # điền ELEVENLABS_API_KEY
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8010
```

Mở `http://127.0.0.1:8010/` để dùng trang test: bấm nói → STT live (hiện chữ tạm trong lúc nói) → TTS →
phát lại, kèm thời gian từng bước; tùy chọn tự dừng khi Gemini thấy hết nói. Có form riêng cho
`/v1/stt` (upload file) và `/v1/tts`. Lần chạy đầu tải weight VieNeu
từ Hugging Face; `/health` báo `tts.state` = `loading` cho tới khi sẵn sàng.

Docker: `docker compose up --build` (cổng `127.0.0.1:8010`, weight cache trong volume `models`).
Image hỗ trợ GPU NVIDIA (PyTorch CUDA runtime và GPU reservation trong `compose.yaml`).
Khuyến nghị chạy build với giới hạn song song:
```bash
docker compose --parallel 1 up -d --build
```
do thư viện CUDA rất lớn (~3.5GB), tránh nghẽn RAM và I/O khi build song song cùng các container khác.

Trên Windows không có GPU, nên chạy TTS trực tiếp (`uv run …`) thay vì Docker Desktop: đo trên cùng máy,
trong container (VM WSL2, 8 vCPU dùng chung với các container khác) TTS chậm hơn ~3–4 lần (âm đầu
~0,6–0,8 s, RTF ~1,4). Trên server Linux chạy Docker trực tiếp thì không bị khoản chậm này.

## API

`/health` và `/` luôn mở. Khi đặt `SERVICE_TOKEN`, mọi `/v1/*` cần `Authorization: Bearer <token>`.

### `WS /v1/stt/stream` — STT streaming (khuyên dùng cho hội thoại)

Client gửi:

| Tin nhắn | Mô tả |
|---|---|
| `{"type": "start", "language"?, "mode"?, "vocabulary"?, "fluency"?, "token"?}` | Bắt buộc, tin đầu tiên. Ý nghĩa field như `/v1/stt`; `fluency: true` để đo độ ấp úng (xem dưới); `token` khi có `SERVICE_TOKEN` (WebSocket trình duyệt không gửi được header) |
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

**Đo độ ấp úng (`fluency: true`).** Gemini chạy ở chế độ verbatim (giữ "ừm", "ờ", lặp từ) và service giữ
PCM của lượt. `final` khi đó có thêm `verbatim` (nguyên văn), `fluency` (số liệu), còn `text` là bản đã bỏ
từ đệm. Số liệu tính ngay khi chốt lượt (không gọi API thêm, không thêm độ trễ), trong `app/fluency.py`:

| Field | Ý nghĩa |
|---|---|
| `speech_duration_ms`, `total_silence_ms`, `pause_count`, `words_per_minute` | Cùng tên với `vad_metrics` của core, chuyển thẳng được |
| `long_pause_count`, `longest_pause_ms` | Ngắt ≥ 1 s; ngắt dài nhất |
| `leading_silence_ms` | Im lặng từ lúc bắt đầu thu tới khi nói |
| `filler_count`, `fillers`, `fillers_per_100_words` | Từ đệm (ừm, ờ, ơ, um, uh…; "à"/"ừ" chỉ khi đứng riêng) |
| `repetition_count` | Lặp cụm từ liền nhau ("em là em là"), không tính từ láy ("từ từ") |
| `hesitation_level` | `low` / `medium` / `high`, gộp từ đệm, ngắt dài và lặp từ |

Đây là heuristic mô tả cách nói, không phải đánh giá năng lực. Ngắt quãng đo bằng năng lượng tín hiệu
nên tiếng ồn nền lớn làm số liệu kém chính xác.

### `POST /v1/stt` — STT batch, multipart

| Field | Bắt buộc | Mô tả |
|---|---|---|
| `file` | có | WAV, MP3, OGG, FLAC, M4A, AAC, AIFF, Opus, WebM (MediaRecorder của trình duyệt). Tối đa 15 MB |
| `language` | không | BCP-47, phân tách bằng dấu phẩy (`vi-VN,en-US`); `auto` để tự nhận diện. Mặc định `STT_LANGUAGE_CODES` |
| `mode` | không | `smart` (bỏ ậm ừ, tự định dạng) hoặc `verbatim`. Mặc định `STT_MODE` |
| `vocabulary` | không | Thuật ngữ cần nhận đúng, phân tách bằng dấu phẩy (`Kubernetes,FastAPI`) |

```json
{"text": "Em đã triển khai hệ thống bằng FastAPI và Redis.", "model": "scribe_v2 (ElevenLabs Scribe v2)",
 "language_codes": ["vi-VN"], "mode": "smart", "latency_ms": 3684.3}
```

### `POST /v1/tts` — JSON `{"text": "...", "voice": "Mai Anh"}`

Trả `audio/wav` (48 kHz, PCM 16-bit, có 150 ms im lặng đầu và 80 ms cuối để loa không nuốt âm đầu).
Header: `X-Audio-Duration` (giây, không tính phần đệm), `X-Latency-Ms`, `X-Voice` (URL-encoded),
`X-Cache-Hit`. Tối đa `TTS_MAX_CHARS` ký tự; nên gửi từng câu để giảm độ trễ.

### `POST /v1/tts/stream` — JSON như `/v1/tts` (khuyên dùng cho hội thoại)

Trả PCM thô `audio/pcm` (s16le, mono) theo từng đoạn ngay khi VieNeu sinh ra; header `X-Sample-Rate`
(48000 với VieNeu Turbo, 24000 với Nano và Gemini TTS), `X-Channels`, `X-Sample-Format`, `X-Voice`. Có cùng phần đệm im lặng
như `/v1/tts`. Lỗi kiểm tra đầu vào (400/413/422/503) trả về trước khi stream bắt đầu. Câu đã sinh được
cache, lần sau trả ngay. Client phát bằng Web Audio, xếp lịch từng đoạn nối tiếp nhau (xem `ttsStream`
trong `app/static/index.html`).

### `GET /v1/tts/voices`

Danh sách giọng preset của backend đang chạy (`id` dùng cho trường `voice`).

### Mã lỗi

`400` audio/văn bản rỗng · `401` sai token · `413` vượt giới hạn · `415` định dạng audio không hỗ trợ ·
`422` giọng/mode không hợp lệ · `502`/`504` Gemini lỗi hoặc quá thời gian · `503` thiếu
`ELEVENLABS_API_KEY` hoặc VieNeu không tải được.

## Chọn backend TTS

Đặt `TTS_BACKEND` trong `.env`:

| Giá trị | Kết quả |
|---|---|
| `auto` (mặc định, hoặc bỏ trống) | PyTorch thấy GPU NVIDIA → VieNeu trên CUDA với cấu hình cao nhất (Turbo 48 kHz, `TTS_GPU_DTYPE=float32`, batch 32, 16 luồng stream). Không có GPU → Gemini TTS |
| `vieneu` | Luôn dùng VieNeu local; `TTS_DEVICE` (`auto`/`cpu`/`cuda`) chọn thiết bị, như mục dưới |
| `gemini` | Luôn dùng Gemini TTS (`GEMINI_TTS_MODEL`, `GEMINI_TTS_VOICE`, `GEMINI_TTS_STYLE`), cần `ELEVENLABS_API_KEY` |

"Có GPU" nghĩa là đã `uv sync --extra cuda` và `torch.cuda.is_available()` là `True`. Máy có GPU NVIDIA
nhưng chưa cài torch CUDA thì service vẫn chọn Gemini và ghi cảnh báo vào log. `/health` → `tts.provider`
cho biết backend đang chạy.

Giọng mặc định là nữ trẻ (người phỏng vấn Linh): VieNeu `Trúc Ly` (nữ, Bắc, tự nhiên; có cả ở v3-turbo và
v3-nano), Gemini `Leda` (youthful) kèm `GEMINI_TTS_STYLE="young, warm and friendly female interviewer"`.
Đổi bằng `TTS_VOICE` / `GEMINI_TTS_VOICE` / `GEMINI_TTS_STYLE`.

Gemini TTS: giọng là bộ preset của Gemini (`Kore`, `Puck`, …; `GET /v1/tts/voices`), không phải giọng
VieNeu. `/v1/tts/stream` trả PCM 24 kHz (`X-Sample-Rate: 24000`), cùng phần đệm im lặng và cache như VieNeu.
Đo một lần trên máy này (flash-lite): âm đầu tiên ~2,8 s, câu ~3 s audio xong trong ~4,8 s, chậm hơn nhiều
so với VieNeu local (~0,2 s âm đầu).

## TTS trên CPU và GPU (VieNeu)

Khi dùng VieNeu với `TTS_DEVICE=auto`: nếu cài PyTorch có CUDA (`uv sync --extra cuda`) và thấy GPU NVIDIA thì
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

Xem `.env.example`. Các biến chính: `ELEVENLABS_API_KEY`, `STT_LIVE_MODEL`, `STT_LIVE_FINAL_TIMEOUT_SEC`, `STT_MODEL`, `STT_LANGUAGE_CODES`, `STT_MODE`,
`TTS_BACKEND` (`auto` | `vieneu` | `gemini`), `GEMINI_TTS_MODEL`, `GEMINI_TTS_VOICE`, `GEMINI_TTS_STYLE`,
`TTS_MODEL`, `TTS_VOICE`, `TTS_DEVICE` (`auto` | `cpu` | `cuda`), `TTS_CPU_PRECISION`, `TTS_CPU_THREADS`, `TTS_GPU_DTYPE`, `TTS_PRELOAD`,
`SERVICE_TOKEN`, `CORS_ORIGINS`.


hachimi
