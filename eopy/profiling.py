"""Lightweight profiling for propagation force models."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator


@dataclass(slots=True)
class EffectTiming:
    calls: int = 0
    seconds: float = 0.0

    @property
    def seconds_per_call(self) -> float:
        return self.seconds / self.calls if self.calls else 0.0


@dataclass(slots=True)
class PropagationProfile:
    """Accumulate wall-clock timings for propagation and individual effects."""

    timings: dict[str, EffectTiming] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            timing = self.timings.setdefault(name, EffectTiming())
            timing.calls += 1
            timing.seconds += perf_counter() - start

    def reset(self) -> None:
        self.timings.clear()

    def report(self) -> str:
        """Return a compact, slowest-first timing table.

        Categories may be nested (for example, atmosphere density is part of
        drag), so percentages are diagnostic and need not sum to 100 percent.
        """

        total = self.timings.get("propagation")
        total_seconds = total.seconds if total is not None else 0.0
        header = f"{'effect':<26} {'calls':>10} {'total [s]':>12} {'per call [us]':>15} {'propagation %':>14}"
        rows = [header, "-" * len(header)]
        for name, timing in sorted(
            self.timings.items(), key=lambda item: item[1].seconds, reverse=True
        ):
            percent = 100.0 * timing.seconds / total_seconds if total_seconds else 0.0
            rows.append(
                f"{name:<26} {timing.calls:>10d} {timing.seconds:>12.6f} "
                f"{timing.seconds_per_call * 1e6:>15.3f} {percent:>13.2f}%"
            )
        return "\n".join(rows)
