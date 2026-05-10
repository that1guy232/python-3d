"""Small frame profiler for finding real-time bottlenecks.

The logger is intentionally lightweight: it accumulates named timing sections
for a few seconds, then prints average/max costs sorted by avg time per frame.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Iterator


@dataclass
class _TimingStat:
    total_s: float = 0.0
    max_s: float = 0.0
    count: int = 0

    def add(self, elapsed_s: float) -> None:
        self.total_s += elapsed_s
        self.max_s = max(self.max_s, elapsed_s)
        self.count += 1


class PerformanceLogger:
    """Collect and periodically print frame-time section costs."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        report_interval: float = 3.0,
        top_n: int = 14,
        warmup_frames: int = 20,
    ) -> None:
        self.enabled = bool(enabled)
        self.report_interval = max(0.25, float(report_interval))
        self.top_n = max(1, int(top_n))
        self.warmup_frames = max(0, int(warmup_frames))
        self._warmup_remaining = self.warmup_frames
        self._frame_start_s: float | None = None
        self._interval_start_s = time.perf_counter()
        self._stats: dict[str, _TimingStat] = {}
        self._counters: dict[str, float] = {}
        self._frame_times_s: list[float] = []
        self._frame_count = 0

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self.enabled == enabled:
            return
        self.enabled = enabled
        self.reset()
        state = "enabled" if enabled else "disabled"
        print(f"[perf] Performance logging {state}. Press F3 to toggle.")

    def toggle(self) -> bool:
        self.set_enabled(not self.enabled)
        return self.enabled

    def reset(self) -> None:
        self._stats.clear()
        self._counters.clear()
        self._frame_times_s.clear()
        self._frame_count = 0
        self._frame_start_s = None
        self._interval_start_s = time.perf_counter()
        self._warmup_remaining = self.warmup_frames

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        start_s = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - start_s)

    def record(self, name: str, elapsed_s: float) -> None:
        if not self.enabled:
            return
        stat = self._stats.get(name)
        if stat is None:
            stat = _TimingStat()
            self._stats[name] = stat
        stat.add(max(0.0, float(elapsed_s)))

    def count(self, name: str, amount: float = 1.0) -> None:
        if not self.enabled:
            return
        self._counters[name] = self._counters.get(name, 0.0) + float(amount)

    def begin_frame(self) -> None:
        if not self.enabled:
            return
        self._frame_start_s = time.perf_counter()

    def end_frame(self) -> None:
        if not self.enabled or self._frame_start_s is None:
            return

        now_s = time.perf_counter()
        elapsed_s = now_s - self._frame_start_s
        self._frame_start_s = None
        self._frame_times_s.append(elapsed_s)
        self._frame_count += 1
        self.record("frame.total", elapsed_s)

        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            self._clear_interval(now_s)
            return

        if now_s - self._interval_start_s >= self.report_interval:
            self.report(now_s)
            self._clear_interval(now_s)

    def report(self, now_s: float | None = None) -> None:
        if not self.enabled or self._frame_count <= 0:
            return

        elapsed_s = (now_s if now_s is not None else time.perf_counter()) - self._interval_start_s
        total_frame_s = sum(self._frame_times_s)
        avg_frame_s = total_frame_s / self._frame_count if self._frame_count else 0.0
        max_frame_s = max(self._frame_times_s) if self._frame_times_s else 0.0
        p95_frame_s = self._percentile(self._frame_times_s, 95.0)
        fps = 1.0 / avg_frame_s if avg_frame_s > 0.0 else 0.0

        print(
            "[perf] "
            f"{self._frame_count} frames in {elapsed_s:.2f}s | "
            f"avg {avg_frame_s * 1000.0:.2f} ms ({fps:.1f} FPS) | "
            f"p95 {p95_frame_s * 1000.0:.2f} ms | "
            f"max {max_frame_s * 1000.0:.2f} ms"
        )
        print("[perf] section                         avg/frame  avg/hit      max  hits/frame")

        rows = []
        for name, stat in self._stats.items():
            if name == "frame.total":
                continue
            avg_per_frame_s = stat.total_s / self._frame_count
            avg_per_hit_s = stat.total_s / stat.count if stat.count else 0.0
            hits_per_frame = stat.count / self._frame_count
            rows.append((avg_per_frame_s, name, avg_per_hit_s, stat.max_s, hits_per_frame))

        rows.sort(reverse=True)
        for avg_frame_s, name, avg_hit_s, max_s, hits_per_frame in rows[: self.top_n]:
            print(
                "[perf] "
                f"{name[:30]:30} "
                f"{avg_frame_s * 1000.0:8.3f} "
                f"{avg_hit_s * 1000.0:8.3f} "
                f"{max_s * 1000.0:8.3f} "
                f"{hits_per_frame:10.2f}"
            )

        if self._counters:
            print("[perf] counters                        avg/frame")
            for name, value in sorted(
                self._counters.items(),
                key=lambda item: item[1],
                reverse=True,
            )[: self.top_n]:
                print("[perf] " f"{name[:30]:30} {value / self._frame_count:9.2f}")

    def _clear_interval(self, now_s: float) -> None:
        self._interval_start_s = now_s
        self._stats.clear()
        self._counters.clear()
        self._frame_times_s.clear()
        self._frame_count = 0

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = int(round((len(ordered) - 1) * max(0.0, min(100.0, percentile)) / 100.0))
        return ordered[index]
