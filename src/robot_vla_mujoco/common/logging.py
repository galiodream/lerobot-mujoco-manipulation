"""Local logging utilities (JSONL + CSV). Stub for Milestone 1."""

import json
import os
from pathlib import Path


class LocalLogger:
    """Minimal JSONL logger for rollout metrics."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self.output_dir / "metrics.jsonl"

    def log(self, record: dict) -> None:
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")
