"""Streaming provider adapters.

Every interface is an async generator so the pipeline's stage boundaries stay
identical whether the backend is a deterministic simulator or a real API.
Real adapters require the corresponding API key; without one the app falls
back to the simulator and every command still runs offline.

Simulated providers use an injectable async sleep so the ASGI event loop is
never blocked. Real providers do their own network I/O, which supplies their
latency naturally (``simulate=False`` disables artificial sleeps).
"""

from __future__ import annotations

import json
import math
import os
import struct
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from src.models import PipelineConfig
from src.tools import ToolRegistry

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech"

_Sleep = Callable[[float], Awaitable[None]]


async def _async_sleep(ms: float) -> None:
    if ms > 0:
        import asyncio

        await asyncio.sleep(ms / 1000.0)


async def _no_sleep(_ms: float) -> None:
    return None


# --------------------------------------------------------------------------- STT


@dataclass
class Transcript:
    text: str
    final: bool
    confidence: float = 1.0


class STTProvider:
    name = "base"

    async def transcribe(self, pcm: bytes, config: PipelineConfig) -> AsyncIterator[Transcript]:
        if False:
            yield Transcript("", True)  # pragma: no cover - async generator marker
        raise NotImplementedError


class SimulatedSTT(STTProvider):
    """Maps audio amplitude to canned phrases so tests stay deterministic."""

    name = "simulated"

    PHRASES = (
        "where is my order ORD-123",
        "hello can you help me",
        "what are your support hours",
        "can you reschedule my delivery to friday",
    )

    def __init__(self, sleep: _Sleep | None = None, phrases: tuple[str, ...] | None = None) -> None:
        self.sleep = sleep or _async_sleep
        self.phrases = phrases or self.PHRASES

    async def transcribe(self, pcm: bytes, config: PipelineConfig) -> AsyncIterator[Transcript]:
        if not pcm:
            return
        from src.vad import VoiceActivityDetector

        level = VoiceActivityDetector.rms(pcm)
        phrase = self.phrases[0] if level >= config.vad_rms_threshold else self.phrases[1]
        await self.sleep(config.profile.partial_stt_ms)
        yield Transcript(phrase[:14], final=False, confidence=0.72)
        await self.sleep(config.profile.stt_final_ms - config.profile.partial_stt_ms)
        yield Transcript(phrase, final=True, confidence=0.98)


class DeepgramSTT(STTProvider):
    """Streaming Deepgram Nova-2 transcription over the REST endpoint."""

    name = "deepgram"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")

    async def transcribe(self, pcm: bytes, config: PipelineConfig) -> AsyncIterator[Transcript]:
        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is required for the deepgram STT provider")
        headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "audio/raw"}
        params = {
            "encoding": "linear16",
            "sample_rate": str(config.sample_rate),
            "channels": "1",
            "smart_format": "true",
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(DEEPGRAM_URL, params=params, headers=headers, content=pcm)
            response.raise_for_status()
            body = response.json()
        best = body["results"]["channels"][0]["alternatives"][0]
        yield Transcript(best["transcript"][:14] or "...", final=False)
        yield Transcript(best["transcript"], final=True, confidence=best.get("confidence", 0.0))


class WhisperSTT(STTProvider):
    """Local faster-whisper transcription (offline model, no network key)."""

    name = "whisper"

    async def transcribe(self, pcm: bytes, config: PipelineConfig) -> AsyncIterator[Transcript]:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pip install faster-whisper to use the whisper STT provider") from exc
        import io
        import wave

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(config.sample_rate)
            wav.writeframes(pcm)
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(buffer.getvalue(), language="en")
        text = " ".join(segment.text.strip() for segment in segments).strip()
        yield Transcript(text[:14], final=False)
        yield Transcript(text, final=True)


def make_stt(config: PipelineConfig, *, simulate: bool = True) -> STTProvider:
    if config.stt_provider == "deepgram":
        return DeepgramSTT()
    if config.stt_provider == "whisper":
        return WhisperSTT()
    return SimulatedSTT(sleep=_async_sleep if simulate else _no_sleep)


# ---------------------------------------------------------------------------- LLM


@dataclass
class LLMResult:
    text: str
    tool_calls: list[dict[str, Any]]


class LLMProvider:
    name = "base"

    async def generate(
        self, transcript: str, tools: ToolRegistry, config: PipelineConfig
    ) -> AsyncIterator[LLMResult | str]:
        if False:
            yield ""  # pragma: no cover - async generator marker
        raise NotImplementedError


class SimulatedLLM(LLMProvider):
    """Intent keyword matching with streaming tokens and real tool calls."""

    name = "simulated"

    SYSTEM = "You are a concise support agent. Resolve the request in one or two sentences."

    def __init__(self, sleep: _Sleep | None = None) -> None:
        self.sleep = sleep or _async_sleep

    async def generate(
        self, transcript: str, tools: ToolRegistry, config: PipelineConfig
    ) -> AsyncIterator[LLMResult | str]:
        import re

        text = transcript.lower()
        await self.sleep(config.profile.llm_ttft_ms)
        calls: list[dict[str, Any]] = []
        reply: str
        # Accept "order ORD-123", "order 123", "ORD-123", "order number 123" etc.
        match = re.search(r"(?:ord)?[-\s]*(\d{2,4})", text)
        if ("order" in text or "ord" in text) and match:
            order_id = f"ORD-{match.group(1).lstrip('0') or match.group(1)}"
            calls.append({"name": "lookup_order", "args": {"order_id": order_id}})
            result = tools.call(calls[0]["name"], calls[0]["args"])
            reply = f"Thanks for waiting. {result} Anything else I can help with?"
        elif "reschedule" in text or "delivery" in text:
            date_match = re.search(r"\b(\d{3})\b", text)
            order_id = f"ORD-{date_match.group(1)}" if date_match else "ORD-123"
            date = "friday" if "friday" in text else "monday"
            calls.append(
                {"name": "reschedule_delivery", "args": {"order_id": order_id, "new_date": date}}
            )
            result = tools.call(calls[0]["name"], calls[0]["args"])
            reply = f"Done. {result} Is there anything else?"
        elif "hour" in text or "open" in text:
            calls.append({"name": "business_hours", "args": {}})
            result = tools.call(calls[0]["name"], calls[0]["args"])
            reply = f"{result} Can I help with anything else?"
        else:
            reply = "I can help you track an order, change a delivery, or check our hours."

        yield LLMResult(reply, calls)
        for token in reply.split(" "):
            yield token + " "
            await self.sleep(config.profile.token_ms)


class OpenAILLM(LLMProvider):
    """OpenAI chat completions with streaming and function calling."""

    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    async def generate(
        self, transcript: str, tools: ToolRegistry, config: PipelineConfig
    ) -> AsyncIterator[LLMResult | str]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the openai LLM provider")
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "messages": [
                {"role": "system", "content": SimulatedLLM.SYSTEM},
                {"role": "user", "content": transcript},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": spec["name"],
                        "description": spec["description"],
                        "parameters": {
                            k: v
                            for k, v in spec.items()
                            if k not in ("name", "description", "title")
                        },
                    },
                }
                for spec in tools.describe()
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        arguments: dict[int, str] = {}
        names: dict[int, str] = {}
        text: list[str] = []
        async with httpx.AsyncClient(timeout=20.0) as client:
            async with client.stream("POST", OPENAI_URL, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"]
                    if delta.get("content"):
                        text.append(delta["content"])
                        yield delta["content"]
                    for call in delta.get("tool_calls", []):
                        index = call["index"]
                        if call.get("id"):
                            names[index] = call["function"]["name"]
                        arguments[index] = arguments.get(index, "") + call["function"].get(
                            "arguments", ""
                        )
        calls = [
            {"name": names[index], "args": json.loads(args)}
            for index, args in arguments.items()
            if index in names and args
        ]
        yield LLMResult("".join(text), calls)


def make_llm(config: PipelineConfig, *, simulate: bool = True) -> LLMProvider:
    return (
        OpenAILLM()
        if config.llm_provider == "openai"
        else SimulatedLLM(sleep=_async_sleep if simulate else _no_sleep)
    )


# ----------------------------------------------------------------------------- TTS


class TTSProvider:
    name = "base"

    async def synthesize(self, text: str, config: PipelineConfig) -> AsyncIterator[bytes]:
        if False:
            yield b""  # pragma: no cover - async generator marker
        raise NotImplementedError


class SimulatedTTS(TTSProvider):
    """Produces deterministic PCM 'audio' chunks with a realistic start delay."""

    name = "simulated"

    def __init__(self, sleep: _Sleep | None = None) -> None:
        self.sleep = sleep or _async_sleep

    @staticmethod
    def tone_chunk(seed: str, sample_rate: int) -> bytes:
        samples = min(1600, 200 + len(seed) * 16)
        return b"".join(
            struct.pack("<h", int(6000 * math.sin(2 * math.pi * 440 * i / sample_rate)))
            for i in range(samples)
        )

    async def synthesize(self, text: str, config: PipelineConfig) -> AsyncIterator[bytes]:
        await self.sleep(config.profile.tts_first_ms)
        for word in text.split():
            yield self.tone_chunk(word, config.sample_rate)
            await self.sleep(config.profile.token_ms)


class ElevenLabsTTS(TTSProvider):
    """Streaming ElevenLabs speech synthesis."""

    name = "elevenlabs"

    def __init__(self, api_key: str | None = None, voice_id: str | None = None) -> None:
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
        self.voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

    async def synthesize(self, text: str, config: PipelineConfig) -> AsyncIterator[bytes]:
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is required for the elevenlabs TTS provider")
        url = f"{ELEVENLABS_URL}/{self.voice_id}/stream"
        headers = {"xi-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {"text": text, "model_id": "eleven_turbo_v2_5", "output_format": "pcm_16000"}
        async with httpx.AsyncClient(timeout=20.0) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk


def make_tts(config: PipelineConfig, *, simulate: bool = True) -> TTSProvider:
    return (
        ElevenLabsTTS()
        if config.tts_provider == "elevenlabs"
        else SimulatedTTS(sleep=_async_sleep if simulate else _no_sleep)
    )
