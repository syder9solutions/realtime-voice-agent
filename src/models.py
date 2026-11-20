"""Shared event, configuration and tool-schema models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Reject unknown fields so bad input cannot smuggle extra arguments."""

    model_config = ConfigDict(extra="forbid")


class EventType(str, Enum):
    SESSION_START = "session_start"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    PARTIAL_TRANSCRIPT = "partial_transcript"
    FINAL_TRANSCRIPT = "final_transcript"
    LLM_TOKEN = "llm_token"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AUDIO = "audio"
    INTERRUPTION = "interruption"
    TURN_COMPLETE = "turn_complete"
    METRICS = "metrics"
    ERROR = "error"


class VoiceEvent(StrictModel):
    type: EventType
    session_id: str
    turn_id: int = 0
    ts_ms: float = 0.0
    payload: dict[str, Any] = Field(default_factory=dict)


class LatencyProfile(StrictModel):
    """Simulated per-stage delays (ms) used by offline providers.

    Defaults mirror the realistic budget in the project brief; evals override
    with deterministic values so measurements are reproducible.
    """

    stt_final_ms: float = 180.0
    llm_ttft_ms: float = 210.0
    tts_first_ms: float = 140.0
    token_ms: float = 8.0
    partial_stt_ms: float = 40.0


class PipelineConfig(StrictModel):
    sample_rate: int = 16_000
    frame_ms: int = 20
    vad_rms_threshold: float = 0.025
    vad_hangover_ms: int = 350
    max_turn_ms: int = 15_000
    max_transcript_chars: int = 500
    max_audio_bytes: int = 64_000
    stt_provider: Literal["simulated", "deepgram", "whisper"] = "simulated"
    llm_provider: Literal["simulated", "openai"] = "simulated"
    tts_provider: Literal["simulated", "elevenlabs"] = "simulated"
    profile: LatencyProfile = Field(default_factory=LatencyProfile)

    @field_validator("sample_rate")
    @classmethod
    def _valid_rate(cls, value: int) -> int:
        if value not in (8_000, 16_000, 24_000, 48_000):
            raise ValueError("sample_rate must be 8000/16000/24000/48000")
        return value

    @field_validator("frame_ms")
    @classmethod
    def _valid_frame(cls, value: int) -> int:
        if not 10 <= value <= 100:
            raise ValueError("frame_ms must be between 10 and 100")
        return value


class LookupOrderArgs(StrictModel):
    order_id: str = Field(min_length=2, max_length=32, pattern=r"^[A-Za-z0-9-]+$")


class RescheduleDeliveryArgs(StrictModel):
    order_id: str = Field(min_length=2, max_length=32, pattern=r"^[A-Za-z0-9-]+$")
    new_date: str = Field(min_length=6, max_length=24, pattern=r"^[A-Za-z0-9 ,/-]+$")


class BusinessHoursArgs(StrictModel):
    pass
