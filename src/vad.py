"""Sample-level voice activity detection with hangover, tuned for barge-in."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from src.models import PipelineConfig


class VadState(Enum):
    SILENCE = auto()
    SPEECH = auto()


@dataclass
class VadFrameResult:
    state: VadState
    speech_started: bool = False
    speech_ended: bool = False
    rms: float = 0.0


@dataclass
class VoiceActivityDetector:
    """Energy VAD over mono 16-bit PCM frames.

    A frame is voiced when RMS exceeds ``threshold``. ``hangover_ms`` keeps the
    speech state alive through short pauses so natural speech is not chopped.
    """

    config: PipelineConfig = field(default_factory=PipelineConfig)
    state: VadState = VadState.SILENCE
    _silent_ms: float = 0.0

    @staticmethod
    def rms(pcm: bytes) -> float:
        if len(pcm) < 2:
            return 0.0
        # Use array to avoid per-sample struct overhead on 20ms/320-sample frames.
        from array import array

        samples = array("h")
        samples.frombytes(pcm[: len(pcm) // 2 * 2])
        if not samples:
            return 0.0
        return (sum(sample * sample for sample in samples) / len(samples)) ** 0.5

    @property
    def frame_ms(self) -> int:
        return self.config.frame_ms

    def process(self, pcm: bytes) -> VadFrameResult:
        level = self.rms(pcm)
        voiced = level >= self.config.vad_rms_threshold
        started = False
        ended = False
        if self.state is VadState.SILENCE:
            if voiced:
                self.state = VadState.SPEECH
                self._silent_ms = 0.0
                started = True
        else:
            if voiced:
                self._silent_ms = 0.0
            else:
                self._silent_ms += self.frame_ms
                if self._silent_ms >= self.config.vad_hangover_ms:
                    self.state = VadState.SILENCE
                    self._silent_ms = 0.0
                    ended = True
        return VadFrameResult(self.state, started, ended, level)

    def reset(self) -> None:
        self.state = VadState.SILENCE
        self._silent_ms = 0.0
