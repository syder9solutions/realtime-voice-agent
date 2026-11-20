"""Stage timing instrumentation and percentile reporting."""

from __future__ import annotations

import math
import statistics
import time
from typing import Callable

from pydantic import BaseModel, Field


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 1]); 0.0 for an empty sample."""
    if not values:
        return 0.0
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


class StageStats(BaseModel):
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    mean_ms: float = 0.0
    count: int = 0


class LatencyReport(BaseModel):
    stages: dict[str, StageStats] = Field(default_factory=dict)
    completed_turns: int = 0

    def stage(self, name: str) -> StageStats:
        return self.stages.get(name, StageStats())


class LatencyTracker:
    """Records stage offsets relative to the start of each turn.

    Injected ``clock`` keeps tests deterministic. ``mark`` records an absolute
    offset from turn start; ``stage`` records an explicit duration.
    """

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.perf_counter
        self._turn_id: int | None = None
        self._started: float | None = None
        self._cancelled: bool = False
        self.samples: dict[str, list[float]] = {}
        self.turns: list[dict[str, float]] = []

    @property
    def active_turn(self) -> int | None:
        return self._turn_id

    def begin_turn(self, turn_id: int) -> None:
        if self._turn_id is not None and not self._cancelled:
            self.end_turn()
        self._turn_id = turn_id
        self._started = self._clock()
        self._cancelled = False

    def mark(self, stage: str) -> float:
        """Return ms since turn start and remember it as a stage sample."""
        if self._started is None:
            return 0.0
        elapsed_ms = (self._clock() - self._started) * 1000.0
        self.samples.setdefault(stage, []).append(elapsed_ms)
        return elapsed_ms

    def stage(self, name: str, duration_ms: float) -> None:
        self.samples.setdefault(name, []).append(max(0.0, float(duration_ms)))

    def cancel_turn(self) -> None:
        self._cancelled = True

    def end_turn(self) -> dict[str, float]:
        if self._started is None:
            return {}
        self.mark("total")
        turn = {name: values[-1] for name, values in self.samples.items()}
        self.turns.append({"turn_id": float(self._turn_id or 0), **turn})
        self._turn_id = None
        self._started = None
        return turn

    def report(self) -> LatencyReport:
        report = LatencyReport(completed_turns=len(self.turns))
        for name, values in self.samples.items():
            report.stages[name] = StageStats(
                p50_ms=round(percentile(values, 0.50), 3),
                p95_ms=round(percentile(values, 0.95), 3),
                mean_ms=round(statistics.fmean(values), 3),
                count=len(values),
            )
        return report
