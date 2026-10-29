"""Per-image, per-stage execution tracing with object-level lineage."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


def _cuda_reset_peak() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _cuda_peak_mb() -> float:
    try:
        import torch
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


@dataclass
class StageRecord:
    image_id: str
    stage_name: str
    parameters_used: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    vram_peak_mb: float = 0.0
    input_count: int = 0
    output_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def set_counts(self, input_count: int | None = None, output_count: int | None = None) -> None:
        if input_count is not None:
            self.input_count = int(input_count)
        if output_count is not None:
            self.output_count = int(output_count)

    def warn(self, message: str) -> None:
        self.warnings.append(str(message))


class ExecutionTracer:
    """Collects stage records and object lineage (detector_source, raw_score, calibrated_score, ...)."""

    def __init__(self) -> None:
        self._records: list[StageRecord] = []
        self._lineage: dict[tuple[str, str], dict[str, Any]] = {}

    @contextmanager
    def stage(self, image_id: str, stage_name: str,
              parameters: dict[str, Any] | None = None) -> Iterator[StageRecord]:
        record = StageRecord(str(image_id), stage_name, dict(parameters or {}))
        _cuda_reset_peak()
        start = time.perf_counter()
        try:
            yield record
        except Exception as exc:
            record.warn(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record.duration_ms = (time.perf_counter() - start) * 1000.0
            record.vram_peak_mb = _cuda_peak_mb()
            self._records.append(record)

    def trace_object(self, image_id: str, object_id: str, **lineage: Any) -> None:
        self._lineage.setdefault((str(image_id), str(object_id)), {}).update(lineage)

    def object_lineage(self, image_id: str, object_id: str) -> dict[str, Any]:
        return dict(self._lineage.get((str(image_id), str(object_id)), {}))

    def to_records(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self._records]

    def summary(self) -> dict[str, dict[str, dict[str, float]]]:
        """Per image -> per stage: calls, total duration, max VRAM and warning count."""
        out: dict[str, dict[str, dict[str, float]]] = {}
        for r in self._records:
            s = out.setdefault(r.image_id, {}).setdefault(
                r.stage_name, {"calls": 0, "duration_ms": 0.0, "vram_peak_mb": 0.0, "warnings": 0})
            s["calls"] += 1
            s["duration_ms"] += r.duration_ms
            s["vram_peak_mb"] = max(s["vram_peak_mb"], r.vram_peak_mb)
            s["warnings"] += len(r.warnings)
        return out

    def to_jsonl(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in self.to_records():
                f.write(json.dumps({"type": "stage", **rec}) + "\n")
            for (image_id, object_id), lineage in self._lineage.items():
                f.write(json.dumps({"type": "object", "image_id": image_id, "object_id": object_id,
                                    **lineage}) + "\n")
        return path
