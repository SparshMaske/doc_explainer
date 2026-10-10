"""Per-run artifact logger: every agent output is saved as JSON under data/runs/<run_id>/."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from stratai.core.config import get_settings


def new_run_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


class RunLogger:
    def __init__(self, run_id: str | None = None, data_dir: Path | None = None) -> None:
        self.run_id = run_id or new_run_id()
        base = data_dir if data_dir is not None else get_settings().DATA_DIR
        self.run_dir = Path(base) / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save(self, stage: str, name: str, payload: dict[str, Any]) -> Path:
        """Write `payload` to data/runs/<run_id>/<stage>/<name>.json and return the path."""
        stage_dir = self.run_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        path = stage_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path
