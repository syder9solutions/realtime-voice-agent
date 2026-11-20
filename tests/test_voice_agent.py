"""Tests for the voice agent: models, VAD, timing, tools, pipeline, and HTTP/WS."""

from __future__ import annotations

import asyncio
import base64
import json
from array import array

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from src.api import app, pipelines
from src.latency import LatencyTracker, percentile
from src.models import EventType, PipelineConfig
from src.pipeline import TurnPipeline
from src.providers import SimulatedLLM, SimulatedSTT, SimulatedTTS
from src.tools import ORDERS, ToolError, ToolRegistry
from src.vad import VoiceActivityDetector, VadState

LOUD_PCM = b"".join(int(8000 * (i % 7) / 7).to_bytes(2, "little") for i in range(320))
SILENT_PCM = bytes(640)


def pcm_from(values: list[int]) -> bytes:
    return array("h", values).tobytes()


# --------------------------------------------------------------------------- models


def test_config_defaults_and_validation():
    config = PipelineConfig()
    assert config.sample_rate == 16_000
    assert config.vad_hangover_ms == 350
    with pytest.raises(ValidationError):
        PipelineConfig(sample_rate=1234)
    with pytest.raises(ValidationError):
        PipelineConfig(frame_ms=5)


def test_strict_model_rejects_extra_fields():
    with pytest.raises(ValidationError):
        PipelineConfig(evil_field="x")


# ------------------------------------------------------------------------------ VAD


def test_vad_detects_speech_and_hangover():
    vad = VoiceActivityDetector()
    assert vad.process(SILENT_PCM).state is VadState.SILENCE
    started = vad.process(LOUD_PCM)
    assert started.speech_started and started.state is VadState.SPEECH
    vad.config.vad_hangover_ms = 60
    assert vad.process(SILENT_PCM).state is VadState.SPEECH  # 20ms < 60ms hangover
    vad.process(SILENT_PCM)
    ended = vad.process(SILENT_PCM)
    assert ended.speech_ended and ended.state is VadState.SILENCE


def test_vad_rms_scales_with_amplitude():
    assert VoiceActivityDetector.rms(SILENT_PCM) == 0.0
    assert VoiceActivityDetector.rms(LOUD_PCM) > 0.025


# --------------------------------------------------------------------------- timing


def test_percentiles_and_tracker():
    assert percentile([], 0.5) == 0.0
    assert percentile([10, 20, 30, 40, 50], 0.5) == 30
    assert percentile([10, 20, 30, 40, 50], 0.95) == 50
    now = [0.0]
    tracker = LatencyTracker(clock=lambda: now[0])
    tracker.begin_turn(1)
    now[0] = 0.18
    tracker.mark("stt")
    now[0] = 0.7
    turn = tracker.end_turn()
    assert turn["stt"] == pytest.approx(180)
    assert tracker.report().stages["stt"].p50_ms == pytest.approx(180)


# ---------------------------------------------------------------------------- tools


def test_tools_validated_and_stateful():
    ORDERS["ORD-123"] = {"status": "in transit", "eta": "tomorrow 14:00", "item": "headphones"}
    registry = ToolRegistry()
    result = registry.call("lookup_order", {"order_id": "ORD-123"})
    assert "in transit" in result
    changed = registry.call("reschedule_delivery", {"order_id": "ORD-123", "new_date": "friday"})
    assert "friday" in changed
    assert ORDERS["ORD-123"]["eta"] == "friday"
    with pytest.raises(ToolError):
        registry.call("lookup_order", {"order_id": "../../etc/passwd"})
    with pytest.raises(ToolError):
        registry.call("lookup_order", {"order_id": "x", "extra": 1})
    with pytest.raises(ToolError):
        registry.call("nonexistent", {})


def test_tool_schema_description():
    schemas = {spec["name"]: spec for spec in ToolRegistry().describe()}
    assert "properties" in schemas["lookup_order"]
    assert schemas["lookup_order"]["additionalProperties"] is False


# ------------------------------------------------------------------------- pipeline


async def collect(pipe: TurnPipeline, pcm: bytes = LOUD_PCM, **kwargs):
    return [event async for event in pipe.run_turn(pcm, **kwargs)]


@pytest.mark.asyncio
async def test_full_turn_events_and_latency():
    pipe = TurnPipeline(simulate=False)
    events = await collect(pipe)
    types = [event.type for event in events]
    assert types[0] is EventType.PARTIAL_TRANSCRIPT
    assert EventType.FINAL_TRANSCRIPT in types
    assert EventType.TOOL_CALL in types
    assert EventType.TOOL_RESULT in types
    assert EventType.AUDIO in types
    assert types[-2] is EventType.TURN_COMPLETE
    assert types[-1] is EventType.METRICS
    report = pipe.tracker.report()
    assert report.completed_turns == 1
    assert report.stage("stt_final").p50_ms <= report.stage("llm_ttft").p50_ms
    assert report.stage("llm_ttft").p50_ms <= report.stage("tts_first_audio").p50_ms


@pytest.mark.asyncio
async def test_streaming_tts_starts_before_llm_finishes():
    """Deterministic proof: with a virtual clock, first audio precedes llm_complete.

    Uses sentence-level streaming with a slow token stream. The TTS provider
    starts on the first sentence instead of waiting for the whole reply.
    """
    from evals.voice_eval import VirtualClock, build_pipeline

    clock = VirtualClock()
    config = PipelineConfig()
    config.profile.token_ms = 40
    pipe = build_pipeline(clock)
    pipe.config = config
    seen: list[EventType] = []
    async for event in pipe.run_turn(LOUD_PCM):
        seen.append(event.type)
    marks = pipe.tracker.samples
    assert marks["tts_first_audio"][-1] < marks["llm_complete"][-1], (
        "TTS must start before the full LLM reply is produced"
    )
    assert EventType.AUDIO in seen


@pytest.mark.asyncio
async def test_forced_text_and_rejection():
    pipe = TurnPipeline(simulate=False)
    events = await collect(pipe, b"", forced_text="track order ORD-789")
    assert any(e.type is EventType.TOOL_CALL and e.payload["name"] == "lookup_order" for e in events)
    pipe2 = TurnPipeline(simulate=False)
    bad = await collect(pipe2, b"", forced_text="x" * 501)
    assert any(e.type is EventType.ERROR for e in bad)
    assert not any(e.type is EventType.AUDIO for e in bad)


@pytest.mark.asyncio
async def test_barge_in_stops_audio():
    pipe = TurnPipeline(simulate=False)
    seen: list[EventType] = []
    async for event in pipe.run_turn(LOUD_PCM):
        seen.append(event.type)
        if event.type is EventType.LLM_TOKEN:
            pipe.interrupt()
    assert EventType.INTERRUPTION in seen
    assert EventType.AUDIO not in seen
    assert pipe.session.barge_ins == 1


@pytest.mark.asyncio
async def test_multiple_turns_increment_ids():
    pipe = TurnPipeline(simulate=False)
    await collect(pipe)
    events = await collect(pipe, b"", forced_text="what are your support hours")
    assert events[0].turn_id == 2
    assert any(e.payload.get("name") == "business_hours" for e in events if e.type is EventType.TOOL_CALL)
    assert pipe.tracker.report().completed_turns == 2


@pytest.mark.asyncio
async def test_provider_error_becomes_event():
    config = PipelineConfig(llm_provider="openai")
    pipe = TurnPipeline(config=config, simulate=False)
    events = await collect(pipe)
    assert any(e.type is EventType.ERROR and "OPENAI_API_KEY" in e.payload["message"] for e in events)


# ----------------------------------------------------------------------- http + ws


@pytest.mark.asyncio
async def test_http_routes():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        health = (await client.get("/health")).json()
        assert health["status"] == "ok"
        assert (await client.get("/")).status_code == 200
        created = (await client.post("/api/session")).json()
        sid = created["session_id"]
        assert (await client.post(f"/api/interrupt/{sid}")).json()["interrupted"] is True
        assert (await client.post("/api/interrupt/missing")).status_code == 404
        assert "sessions" in (await client.get("/metrics")).json()


@pytest.mark.asyncio
async def test_websocket_streaming_roundtrip(live_server):
    async with AsyncClient(base_url=f"http://{live_server}") as client:
        sid = (await client.post("/api/session")).json()["session_id"]
    import websockets

    async with websockets.connect(f"ws://{live_server}/ws/{sid}") as ws:
        await ws.send(json.dumps({"type": "setup", "config": {"vad_rms_threshold": 0.01}}))
        hello = json.loads(await ws.recv())
        assert hello["type"] == "session_start"
        await ws.send(json.dumps({"type": "text", "text": "track order ORD-123"}))
        events = []
        for _ in range(120):
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            events.append(event)
            if event["type"] in ("metrics", "error"):
                break
        types = [e["type"] for e in events]
        assert "final_transcript" in types
        assert "audio" in types
        assert "metrics" in types


@pytest.mark.asyncio
async def test_websocket_barge_in_and_validation(live_server):
    async with AsyncClient(base_url=f"http://{live_server}") as client:
        sid = (await client.post("/api/session")).json()["session_id"]
    import websockets

    async with websockets.connect(f"ws://{live_server}/ws/{sid}") as ws:
        await ws.send(json.dumps({"type": "bogus"}))
        assert json.loads(await ws.recv())["type"] == "error"
        await ws.send(json.dumps({"type": "text", "text": "help"}))
        await ws.send(json.dumps({"type": "interrupt"}))
        events = []
        for _ in range(60):
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            events.append(event)
            if any(e["type"] == "interruption" for e in events):
                break
        assert any(e["type"] == "interruption" for e in events)
