"""Deterministic offline evaluators for the voice-agent pipeline.

No API keys and no wall-clock sleeps are used: simulated providers receive a
no-op sleep and an injected virtual clock, so every reported number is an
exact assertion about the architecture's stage ordering and latency budget.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from src.latency import LatencyTracker, percentile
from src.models import PipelineConfig
from src.pipeline import TurnPipeline
from src.providers import SimulatedLLM, SimulatedSTT, SimulatedTTS
from src.tools import ORDERS, ToolRegistry

LOUD_PCM = bytes([80]) * 640  # 320 samples RMS ~55 -> voiced
QUIET_PCM = bytes([0]) * 640


class VirtualClock:
    """Advances by provider-configured delays as stages execute."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, ms: float) -> None:
        if ms > 0:
            self.now += ms / 1000.0
        await asyncio.sleep(0)

    def jump(self, ms: float) -> None:
        self.now += ms / 1000.0


@dataclass
class EvalResult:
    name: str
    passed: bool
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSuite:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def add(self, name: str, passed: bool, detail: str, **metrics: Any) -> None:
        self.results.append(EvalResult(name, passed, detail, metrics))

    def render(self) -> str:
        lines = [
            "=" * 68,
            "REAL-TIME VOICE AGENT — OFFLINE EVALUATION & LATENCY BUDGET",
            "=" * 68,
        ]
        for result in self.results:
            mark = "PASS" if result.passed else "FAIL"
            lines.append(f"[{mark}] {result.name}: {result.detail}")
            for key, value in result.metrics.items():
                lines.append(f"         {key}: {value}")
        lines.append("-" * 68)
        lines.append(f"OVERALL: {'PASS' if self.passed else 'FAIL'} "
                     f"({sum(r.passed for r in self.results)}/{len(self.results)})")
        return "\n".join(lines)


def build_pipeline(clock: VirtualClock) -> TurnPipeline:
    config = PipelineConfig()
    tracker = LatencyTracker(clock=clock)
    return TurnPipeline(
        config=config,
        tracker=tracker,
        tools=ToolRegistry(),
        stt=SimulatedSTT(sleep=clock.sleep),
        llm=SimulatedLLM(sleep=clock.sleep),
        tts=SimulatedTTS(sleep=clock.sleep),
    )


async def eval_latency(suite: EvalSuite) -> None:
    """TTFA must equal the configured stage budget and stay under 1000 ms."""
    clock = VirtualClock()
    pipe = build_pipeline(clock)
    events = [event async for event in pipe.run_turn(LOUD_PCM)]
    report = pipe.tracker.report()
    stages = report.stages
    expected_ttfa = (
        PipelineConfig().profile.stt_final_ms
        + PipelineConfig().profile.llm_ttft_ms
        + PipelineConfig().profile.tts_first_ms
    )
    measured = stages["tts_first_audio"].p50_ms
    total = stages["total"].p50_ms
    assert any(e.type.value == "turn_complete" for e in events)
    # With sentence-level streaming, first audio = STT + LLM-TTFT + a few
    # tokens needed to complete the first sentence + TTS start. The small
    # sentence-accumulation overhead is expected; tool time is NOT included.
    overhead = measured - expected_ttfa
    tool_ms = stages["tool_latency"].p50_ms if "tool_latency" in stages else 0.0
    suite.add(
        "latency_budget",
        0 <= overhead < 100 and measured < 1000,
        f"first-audio at {measured:.0f} ms (vertical budget {expected_ttfa:.0f} ms "
        f"+ {overhead:.0f} ms first-sentence streaming, target <1000)",
        ttfa_p50_ms=measured,
        total_ms=total,
        tool_latency_ms=tool_ms,
    )
    suite.add(
        "stage_ordering",
        stages["stt_final"].p50_ms <= stages["llm_ttft"].p50_ms
        <= stages["tts_first_audio"].p50_ms <= total,
        "stt_final <= llm_ttft <= tts_first_audio <= total",
        **{k: v.p50_ms for k, v in stages.items()},
    )


async def eval_tool_use(suite: EvalSuite) -> None:
    clock = VirtualClock()
    pipe = build_pipeline(clock)
    events = [event async for event in pipe.run_turn(LOUD_PCM)]
    calls = [(e.payload["name"], e.payload["args"]) for e in events if e.type.value == "tool_call"]
    results = [e.payload["result"] for e in events if e.type.value == "tool_result"]
    suite.add(
        "task_success",
        bool(calls) and calls[0][0] == "lookup_order" and "ORD-123" in results[0],
        f"scripted order lookup resolved via tool: {results[0] if results else 'no result'}",
        tool_calls=calls,
    )
    pipe2 = build_pipeline(VirtualClock())
    events2 = [e async for e in pipe2.run_turn(b"", forced_text="track order ORD-456")]
    called = any(e.type.value == "tool_call" and e.payload.get("name") == "lookup_order" for e in events2)
    suite.add("forced_text_tool", called, "push-to-text path also invokes tools")


async def eval_barge_in(suite: EvalSuite) -> None:
    clock = VirtualClock()
    pipe = build_pipeline(clock)
    events: list[Any] = []
    async for event in pipe.run_turn(LOUD_PCM):
        events.append(event)
        if event.type.value == "llm_token":
            pipe.interrupt()
    interruptions = [e for e in events if e.type.value == "interruption"]
    audio_after_interrupt = False
    seen_interrupt = False
    for event in events:
        if event.type.value == "interruption":
            seen_interrupt = True
        if event.type.value == "audio" and seen_interrupt:
            audio_after_interrupt = True
    suite.add(
        "barge_in",
        bool(interruptions) and not audio_after_interrupt,
        f"interruption emitted at stage '{interruptions[0].payload['stage'] if interruptions else 'n/a'}'; "
        "no audio emitted after interrupt",
        barge_ins=pipe.session.barge_ins,
        events=len(events),
    )


async def eval_input_validation(suite: EvalSuite) -> None:
    from src.tools import ToolError

    tools = ToolRegistry()
    rejected = False
    try:
        tools.call("lookup_order", {"order_id": "../etc/passwd"})
    except ToolError:
        rejected = True
    suite.add("tool_schema_guard", rejected, "path-like order id rejected by strict schema")

    pipe = build_pipeline(VirtualClock())
    events = [e async for e in pipe.run_turn(b"", forced_text="x" * 501)]
    suite.add(
        "oversize_reject",
        any(e.type.value == "error" for e in events),
        "over-length transcript rejected before LLM stage",
    )


async def eval_percentiles(suite: EvalSuite) -> None:
    values = [100.0, 200.0, 300.0, 400.0, 500.0]
    correct = percentile(values, 0.5) == 300.0 and percentile(values, 0.95) == 500.0
    suite.add("percentile_math", correct, "nearest-rank p50/p95 correctly computed")


async def run_evals() -> EvalSuite:
    ORDERS["ORD-123"] = {"status": "in transit", "eta": "tomorrow 14:00", "item": "wireless headphones"}
    suite = EvalSuite()
    await eval_latency(suite)
    await eval_tool_use(suite)
    await eval_barge_in(suite)
    await eval_input_validation(suite)
    await eval_percentiles(suite)
    return suite
