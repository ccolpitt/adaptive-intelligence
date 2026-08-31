"""Append-only JSONL persistence helpers, shared by the policy archive and
the task/experiment registries. One record per line, greppable, no updates
in place — the storage primitive behind every append-only invariant."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
