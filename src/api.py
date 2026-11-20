"""ASGI app: WebSocket audio streaming, HTTP health/metrics, static web UI."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.models import EventType, PipelineConfig, VoiceEvent
from src.pipeline import TurnPipeline
from src.vad import VoiceActivityDetector

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Real-Time Voice Agent", version="1.0.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

pipelines: dict[str, TurnPipeline] = {}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "realtime-voice-agent",
        "sessions": len(pipelines),
        "providers": ["simulated", "deepgram", "openai", "elevenlabs"],
    }


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    reports = {sid: pipe.tracker.report().model_dump(mode="json") for sid, pipe in pipelines.items()}
    return {"sessions": reports}


@app.post("/api/session")
async def create_session() -> dict[str, str]:
    pipe = TurnPipeline()
    pipelines[pipe.session.id] = pipe
    return {"session_id": pipe.session.id}


@app.post("/api/interrupt/{session_id}")
async def interrupt(session_id: str) -> JSONResponse:
    pipe = pipelines.get(session_id)
    if pipe is None:
        return JSONResponse({"error": "unknown session"}, status_code=404)
    pipe.interrupt()
    return JSONResponse({"interrupted": True, "session_id": session_id})


async def _send(ws: WebSocket, event: VoiceEvent) -> None:
    await ws.send_text(event.model_dump_json())


def _config_from_setup(setup: dict[str, Any]) -> PipelineConfig:
    allowed = {"stt_provider", "llm_provider", "tts_provider", "profile", "vad_rms_threshold"}
    clean = {key: value for key, value in setup.items() if key in allowed}
    return PipelineConfig.model_validate(clean)


@app.websocket("/ws/{session_id}")
async def voice_socket(ws: WebSocket, session_id: str) -> None:
    await ws.accept()
    pipe = pipelines.get(session_id)
    if pipe is None:
        await _send(ws, TurnPipeline._event(EventType.ERROR, session_id, 0, message="unknown session"))
        await ws.close()
        return
    vad = VoiceActivityDetector(pipe.config)
    audio_buffer = bytearray()
    active_task: asyncio.Task[None] | None = None

    async def consume(pcm: bytes, forced_text: str | None = None) -> None:
        nonlocal active_task
        async for event in pipe.run_turn(bytes(pcm), forced_text=forced_text):
            await _send(ws, event)
        active_task = None

    try:
        while True:
            raw = await ws.receive_text()
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "setup":
                pipe.config = _config_from_setup(message.get("config", {}))
                vad = VoiceActivityDetector(pipe.config)
                await _send(
                    ws,
                    TurnPipeline._event(
                        EventType.SESSION_START,
                        session_id,
                        0,
                        providers={
                            "stt": pipe.config.stt_provider,
                            "llm": pipe.config.llm_provider,
                            "tts": pipe.config.tts_provider,
                        },
                    ),
                )
            elif kind == "audio":
                chunk = base64.b64decode(message.get("audio_b64", ""))
                if len(chunk) > pipe.config.max_audio_bytes:
                    await _send(
                        ws,
                        TurnPipeline._event(
                            EventType.ERROR, session_id, 0, message="audio frame too large"
                        ),
                    )
                    continue
                audio_buffer.extend(chunk)
                result = vad.process(chunk)
                if result.speech_started:
                    if active_task and not active_task.done():
                        pipe.interrupt()
                        await _send(
                            ws,
                            TurnPipeline._event(
                                EventType.INTERRUPTION, session_id, pipe.session.turn_id, stage="vad"
                            ),
                        )
                    audio_buffer = bytearray(chunk)
                    await _send(
                        ws,
                        TurnPipeline._event(
                            EventType.SPEECH_START, session_id, pipe.session.turn_id, rms=result.rms
                        ),
                    )
                if result.speech_ended and audio_buffer:
                    await _send(
                        ws,
                        TurnPipeline._event(
                            EventType.SPEECH_END, session_id, pipe.session.turn_id
                        ),
                    )
                    pcm = bytes(audio_buffer)
                    audio_buffer.clear()
                    active_task = asyncio.create_task(consume(pcm))
            elif kind == "text":
                # Push-to-talk path: bypass VAD, send a known transcript.
                text = str(message.get("text", ""))[: pipe.config.max_transcript_chars]
                if active_task and not active_task.done():
                    pipe.interrupt()
                active_task = asyncio.create_task(consume(b"", forced_text=text))
            elif kind == "interrupt":
                pipe.interrupt()
                await _send(
                    ws,
                    TurnPipeline._event(
                        EventType.INTERRUPTION, session_id, pipe.session.turn_id, stage="manual"
                    ),
                )
            elif kind == "ping":
                await _send(ws, TurnPipeline._event(EventType.SPEECH_END, session_id, 0, pong=True))
            else:
                await _send(
                    ws,
                    TurnPipeline._event(
                        EventType.ERROR, session_id, 0, message=f"unknown message type: {kind}"
                    ),
                )
    except WebSocketDisconnect:
        pass
    finally:
        if active_task and not active_task.done():
            active_task.cancel()
        pipelines.pop(session_id, None)
