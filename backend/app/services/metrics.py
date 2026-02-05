"""
Metrics utilities for BPMN detection pipeline.

Provides timing, memory tracking, and structured logging for performance analysis.
"""
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


@dataclass
class StageMetrics:
    """Metrics for a single processing stage."""
    name: str
    duration_ms: float = 0.0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    memory_delta_mb: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def log(self):
        """Log this stage's metrics."""
        mem_info = ""
        if self.memory_delta_mb != 0:
            sign = "+" if self.memory_delta_mb > 0 else ""
            mem_info = f" | mem: {self.memory_after_mb:.1f}MB ({sign}{self.memory_delta_mb:.1f}MB)"

        extra_info = ""
        if self.extra:
            extra_parts = [f"{k}={v}" for k, v in self.extra.items()]
            extra_info = f" | {', '.join(extra_parts)}"

        logger.info(f"[METRICS] {self.name}: {self.duration_ms:.1f}ms{mem_info}{extra_info}")


@dataclass
class PipelineMetrics:
    """Aggregated metrics for the entire processing pipeline."""
    stages: List[StageMetrics] = field(default_factory=list)
    total_duration_ms: float = 0.0
    initial_memory_mb: float = 0.0
    final_memory_mb: float = 0.0

    def add_stage(self, stage: StageMetrics):
        self.stages.append(stage)

    def finalize(self):
        """Calculate totals and log summary."""
        self.total_duration_ms = sum(s.duration_ms for s in self.stages)
        if self.stages:
            self.initial_memory_mb = self.stages[0].memory_before_mb
            self.final_memory_mb = self.stages[-1].memory_after_mb

        # Log summary
        logger.info("=" * 60)
        logger.info("[METRICS] Pipeline Summary:")
        for stage in self.stages:
            stage.log()
        logger.info("-" * 60)
        mem_delta = self.final_memory_mb - self.initial_memory_mb
        sign = "+" if mem_delta > 0 else ""
        logger.info(
            f"[METRICS] TOTAL: {self.total_duration_ms:.1f}ms | "
            f"mem: {self.initial_memory_mb:.1f}MB -> {self.final_memory_mb:.1f}MB ({sign}{mem_delta:.1f}MB)"
        )
        logger.info("=" * 60)

    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            "total_ms": round(self.total_duration_ms, 1),
            "initial_memory_mb": round(self.initial_memory_mb, 1),
            "final_memory_mb": round(self.final_memory_mb, 1),
            "stages": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 1),
                    "memory_mb": round(s.memory_after_mb, 1),
                    **s.extra
                }
                for s in self.stages
            ]
        }


@contextmanager
def measure_stage(name: str, track_memory: bool = True):
    """
    Context manager for measuring a processing stage.

    Usage:
        with measure_stage("detection") as metrics:
            # do work
            metrics.extra["objects_count"] = 42
    """
    metrics = StageMetrics(name=name)

    if track_memory:
        metrics.memory_before_mb = get_memory_mb()

    start = time.perf_counter()

    try:
        yield metrics
    finally:
        metrics.duration_ms = (time.perf_counter() - start) * 1000

        if track_memory:
            metrics.memory_after_mb = get_memory_mb()
            metrics.memory_delta_mb = metrics.memory_after_mb - metrics.memory_before_mb


class MetricsCollector:
    """
    Collects metrics across multiple stages of processing.

    Usage:
        collector = MetricsCollector()

        with collector.stage("stage1") as m:
            # do work
            m.extra["count"] = 10

        with collector.stage("stage2") as m:
            # do more work

        collector.finalize()  # logs summary
    """

    def __init__(self):
        self.pipeline = PipelineMetrics()
        self._current_stage: Optional[StageMetrics] = None

    @contextmanager
    def stage(self, name: str, track_memory: bool = True):
        """Start a new measured stage."""
        with measure_stage(name, track_memory) as metrics:
            self._current_stage = metrics
            yield metrics
            self.pipeline.add_stage(metrics)
        self._current_stage = None

    def finalize(self) -> PipelineMetrics:
        """Finalize and log all metrics."""
        self.pipeline.finalize()
        return self.pipeline
