"""Turn orchestration: streaming STT -> LLM (tools) -> TTS with barge-in.

A single ``TurnPipeline`` owns one conversation turn. Cancellation is a
cooperative flag checked at every async yield point, so an interruption stops
the LLM token stream and TTS chunks within one scheduling step.

True streaming: LLM tokens and TTS audio are produced concurrently via an
asyncio queue. TTS starts on the *first sentence* rather than waiting for the
model to finish the whole reply, which is what makes first-audio latency equal
to STT + LLM-TTFT + TTS-start instead of the full reply generation time.
"""

from __future__ import annotations

import asyncio
import base64
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.latency import LatencyTracker
from src.models import PipelineConfig, VoiceEvent
from src.providers import LLMResult, LLMProvider, STTProvider, TTSProvider, make_llm, make_stt, make_tts
from src.tools import ToolError, ToolRegistry

_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    turn_id: int = 0
    history: list[dict[str, str]] = field(default_factory=list)
    barge_ins: int = 0


class TurnPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        tracker: LatencyTracker | None = None,
        tools: ToolRegistry | None = None,
        stt: STTProvider | None = None,
        llm: LLMProvider | None = None,
        tts: TTSProvider | None = None,
        simulate: bool = True,
    ) -> None:
        self.config = config or PipelineConfig()
        self.tracker = tracker or LatencyTracker()
        self.tools = tools or ToolRegistry()
        self.stt = stt or make_stt(self.config, simulate=simulate)
        self.llm = llm or make_llm(self.config, simulate=simulate)
        self.tts = tts or make_tts(self.config, simulate=simulate)
        self.session = Session()
        self._cancelled = False

    @staticmethod
    def _event(event_type: Any, session_id: str, turn_id: int, **payload: Any) -> VoiceEvent:
        return VoiceEvent(
            type=event_type,
            session_id=session_id,
            turn_id=turn_id,
            ts_ms=round(time.perf_counter() * 1000, 3),
            payload=payload,
        )

    def interrupt(self) -> None:
        """Cancel the active turn (barge-in). Safe to call repeatedly."""
        self.tracker.cancel_turn()
        self._cancelled = True

    async def run_turn(self, pcm: bytes, forced_text: str | None = None):
        """Run one full turn, yielding events as each stage produces output."""
        self._cancelled = False
        self.session.turn_id += 1
        turn_id = self.session.turn_id
        sid = self.session.id
        self.tracker.begin_turn(turn_id)
        self.tracker.mark("vad_detected")

        # ---- STT: partials then a final transcript ---------------------------
        transcript = ""
        final = False
        if forced_text:
            await self._await_sim(self.config.profile.stt_final_ms)
            transcript = forced_text
            final = True
        else:
            async for result in self.stt.transcribe(pcm, self.config):
                if self._cancelled:
                    break
                transcript = result.text
                if not result.final:
                    yield self._event("partial_transcript", sid, turn_id, text=result.text)
                else:
                    final = True
        if self._cancelled:
            yield self._event("interruption", sid, turn_id, stage="stt")
            self.session.barge_ins += 1
            return
        if not final or not transcript.strip():
            yield self._event("error", sid, turn_id, message="empty transcript")
            return
        if len(transcript) > self.config.max_transcript_chars:
            yield self._event("error", sid, turn_id, message="transcript too long")
            return
        self.tracker.mark("stt_final")
        yield self._event("final_transcript", sid, turn_id, text=transcript)

        # ---- LLM stream + concurrent sentence-level TTS ----------------------
        # The producer pushes LLM tokens and tool events into a queue; the TTS
        # worker consumes sentences from a separate queue. Both run in parallel
        # so first audio does not wait for the full reply.
        events: asyncio.Queue[VoiceEvent | None] = asyncio.Queue()
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        reply_parts: list[str] = []

        async def produce_llm() -> None:
            try:
                ttft_marked = False
                buffer = ""
                async for item in self.llm.generate(transcript, self.tools, self.config):
                    if self._cancelled:
                        return
                    if isinstance(item, LLMResult):
                        reply_parts.append(item.text)
                        if not ttft_marked:
                            self.tracker.mark("llm_ttft")
                            ttft_marked = True
                        for call in item.tool_calls:
                            start = time.perf_counter()
                            try:
                                result = self.tools.call(call["name"], call["args"])
                            except ToolError as exc:
                                result = f"Tool failed: {exc}"
                            self.tracker.stage("tool_latency", (time.perf_counter() - start) * 1000)
                            await events.put(
                                self._event(
                                    "tool_call", sid, turn_id, name=call["name"], args=call["args"]
                                )
                            )
                            await events.put(
                                self._event(
                                    "tool_result", sid, turn_id, name=call["name"], result=result
                                )
                            )
                    elif item:
                        reply_parts.append(item)
                        buffer += item
                        await events.put(self._event("llm_token", sid, turn_id, text=item))
                        # Dispatch a sentence as soon as punctuation arrives.
                        while _SENTENCE_END.search(buffer):
                            match = _SENTENCE_END.search(buffer)
                            assert match
                            sentence = buffer[: match.end()].strip()
                            buffer = buffer[match.end() :]
                            if sentence:
                                await sentences.put(sentence)
                if buffer.strip():
                    await sentences.put(buffer.strip())
            except Exception as exc:
                await events.put(self._event("error", sid, turn_id, message=f"{type(exc).__name__}: {exc}"))
            finally:
                self.tracker.mark("llm_complete")
                await sentences.put(None)
                await events.put(None)

        async def produce_audio() -> None:
            first = True
            while not self._cancelled:
                sentence = await sentences.get()
                if sentence is None:
                    break
                async for chunk in self.tts.synthesize(sentence, self.config):
                    if self._cancelled:
                        return
                    if first:
                        self.tracker.mark("tts_first_audio")
                        first = False
                    await events.put(
                        self._event(
                            "audio", sid, turn_id, audio_b64=base64.b64encode(chunk).decode("ascii")
                        )
                    )
            await events.put(None)

        llm_task = asyncio.create_task(produce_llm())
        tts_task = asyncio.create_task(produce_audio())
        pending = 2
        try:
            while pending > 0 and not self._cancelled:
                event = await events.get()
                if event is None:
                    pending -= 1
                    continue
                yield event
        finally:
            for task in (llm_task, tts_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(llm_task, tts_task, return_exceptions=True)

        if self._cancelled:
            yield self._event("interruption", sid, turn_id, stage="tts")
            self.session.barge_ins += 1
            return
        reply = "".join(reply_parts)
        if not reply:
            yield self._event("error", sid, turn_id, message="empty llm reply")
            return
        self.session.history.extend(
            ({"role": "user", "content": transcript}, {"role": "assistant", "content": reply})
        )
        report = self.tracker.end_turn()
        yield self._event("turn_complete", sid, turn_id, **report)
        yield self._event(
            "metrics", sid, turn_id, latency=self.tracker.report().model_dump(mode="json")
        )

    async def _await_sim(self, ms: float) -> None:
        """Cooperative sleep used by the forced-text path."""
        if ms <= 0:
            await asyncio.sleep(0)
            return
        deadline = time.perf_counter() + ms / 1000.0
        while not self._cancelled and time.perf_counter() < deadline:
            await asyncio.sleep(min(0.005, deadline - time.perf_counter()))
