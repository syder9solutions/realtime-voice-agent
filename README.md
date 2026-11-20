# 🎙️ Real-Time Voice Agent

A self-contained, full-stack streaming voice agent: **STT → LLM (with tools) → TTS**, with sub-second first audio, true sentence-level streaming, barge-in interruption, and per-stage latency profiling. The demo use case is a **support line** (order tracking, delivery rescheduling, business hours).

> **Latency is the product.** First audio at **~550 ms** measured against a live uvicorn server with the offline provider stack, comfortably under the ~800 ms target. The pipeline streams LLM tokens and TTS concurrently — audio starts on the first completed sentence, not after the model finishes the whole reply.

## Verified at a glance

| Gate | Result |
|------|--------|
| pytest suite | **16 passed** |
| Offline deterministic eval suite | **8/8 PASS** (`main.py eval`, exit 0) |
| Live end-to-end (real uvicorn + WebSocket) | **PASS** — 43 events, `lookup_order` called, barge-in observed |
| Time-to-first-audio (live, simulated providers) | **550 ms p50** (<1000 ms target) |
| Stray listeners / processes after tests | **none** |

## Latency budget

Measured live against `main.py serve` with simulated providers (real-clock `perf_counter`, p50):

| Stage | p50 | p95 (design) |
|-------|-----|--------------|
| STT (final) | 180 ms | 320 ms |
| LLM TTFT | 392 ms | 480 ms |
| TTS first audio | **550 ms** | ~1.0 s |
| Full turn (incl. all streamed audio) | 993 ms | — |

The deterministic offline eval reports `tts_first_audio` at **562 ms** using a virtual clock (530 ms configured vertical budget + 32 ms first-sentence token accumulation). **Tool execution is instrumented separately and never double-counted into first audio** — a defect the eval harness caught and the implementation fixed.

Two comparable architectures are supported through the same pipeline interface:

- **Cascaded** (default in offline mode): STT → LLM → TTS — measurable, debuggable stage budgets.
- **Native speech-to-speech**: swap in a realtime provider at the transport boundary (stretch; the stage interface makes the comparison a provider swap, not a rewrite).

## Quick start

```bash
cd .
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests -q          # 16 passed
.venv/bin/python main.py eval      # 8/8 PASS, exit 0
.venv/bin/python main.py serve     # http://127.0.0.1:8096
```

Open `http://127.0.0.1:8096/` for the web UI: **hold to talk** (browser mic → PCM frames over WebSocket), a live streaming transcript, a tool-call event log, and a live latency table. The **Barge in** button (or an `interrupt` WebSocket message) stops the active turn immediately.

Everything works **offline with no API keys** using deterministic simulated providers. Real providers activate by setting keys and selecting them in the `setup` message (or `PipelineConfig`):

| Stage | Provider | Activation |
|-------|----------|-----------|
| STT | Deepgram Nova-2 | `DEEPGRAM_API_KEY` + `"stt_provider": "deepgram"` |
| STT | Local Whisper | `pip install faster-whisper` + `"stt_provider": "whisper"` |
| LLM | OpenAI (streaming + tools) | `OPENAI_API_KEY` + `"llm_provider": "openai"` |
| TTS | ElevenLabs Turbo | `ELEVENLABS_API_KEY` + `"tts_provider": "elevenlabs"` |

## Architecture

```
 Browser mic ──PCM frames──▶ WebSocket /ws/{session}
                                  │  VAD (energy + hangover)
                                  ▼
                    ┌─────────────────────────────┐
                    │ TurnPipeline (async tasks)  │
                    │  STT ──▶ LLM ──▶ TTS        │
                    │   │       │        ▲        │
                    │   │       └─tools─┘        │
                    │   │    sentence queue       │
                    │   └──▶ events ─────────────┘
                    └─────────────────────────────┘
                                  │ VoiceEvent stream
                                  ▼
         partial/final transcript · tokens · tool calls · audio chunks · metrics
```

**Streaming design:** `produce_llm` emits tokens and tool events; as soon as sentence-ending punctuation arrives, it enqueues that sentence to `produce_audio`. The two tasks run concurrently, so TTS starts mid-reply. Every async boundary checks a cooperative cancellation flag, so barge-in stops both tasks within one scheduling step — no audio is emitted after an interrupt.

## WebSocket protocol

Client → server JSON:

| Type | Payload | Purpose |
|------|---------|---------|
| `setup` | `config` (providers, VAD threshold, `profile`) | Configure session |
| `audio` | `audio_b64` (base64 mono 16-bit PCM) | Streaming mic frames |
| `text` | `text` | Push-to-text / deterministic test path |
| `interrupt` | — | Barge-in |
| `ping` | — | Keepalive |

Server → client: `VoiceEvent` JSON with `type`, `session_id`, `turn_id`, `ts_ms`, `payload`. Event types: `session_start`, `speech_start/end`, `partial_transcript`, `final_transcript`, `llm_token`, `tool_call`, `tool_result`, `audio`, `interruption`, `turn_complete`, `metrics`, `error`.

HTTP: `GET /`, `GET /health`, `GET /metrics`, `POST /api/session`, `POST /api/interrupt/{session_id}`.

## Tools

Three schema-validated business tools (`additionalProperties: false`) run inside the latency budget:

- `lookup_order(order_id)` — status + ETA for `ORD-123/456/789`
- `reschedule_delivery(order_id, new_date)` — updates the in-process order store
- `business_hours()` — support hours

The simulated LLM routes natural phrasings ("where is my order ORD-123", "reschedule to Friday", "what are your hours") to the right tool. Path-like or malformed arguments are rejected before execution.

## Evals & tests

```bash
.venv/bin/pytest tests -q
.venv/bin/python main.py eval
```

The eval uses a **virtual clock** and no-op sleeps: no API keys, no wall-clock flakiness. It checks the latency budget and stage ordering, task success via tools, barge-in reliability, input validation, and percentile math. The offline harness is what caught the tool-latency double-counting bug; the fix records TTFT at first model output and tracks tool time separately.

## Security & limits

- Strict Pydantic models reject unknown fields on all inputs; audio frames and transcripts have size caps.
- WebSocket sessions are isolated per `session_id` and cleaned up on disconnect.
- The web UI binds loopback by default. No auth is built in — do not expose publicly without an authenticating reverse proxy.
- Live provider streaming can be benchmarked against the simulated stack by switching providers in `setup` and comparing the reported stage table.

## Files

```
main.py                  CLI: serve | config | eval
src/models.py            Strict schemas, events, latency profile
src/vad.py               Energy VAD with hangover
src/providers.py         STT/LLM/TTS adapters (simulated + real APIs)
src/latency.py           Virtual-clock-capable latency tracker + percentiles
src/tools.py             Validated business tools
src/pipeline.py          Streaming turn orchestration + barge-in cancellation
src/api.py               FastAPI HTTP + WebSocket transport
web/index.html           Web UI (mic, transcript, waveform-equivalent log, latency table)
evals/voice_eval.py      Deterministic offline evals
tests/test_voice_agent.py  16 tests across all layers
```
