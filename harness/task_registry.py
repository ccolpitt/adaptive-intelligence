"""Task registry — append-only, versioned task definitions.

Eval results are meaningless unless pinned to a (policy, task@v, benchmark@v)
triple; this registry is what makes every archived score interpretable
forever (roadmap Phase 0). Append-only JSONL: "start dumb — greppable".
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from .jsonl_store import append_jsonl, now_iso, read_jsonl


class TaskRegistry:
    """Append-only, versioned task definitions. Registering an existing
    (task_id, version) with an identical definition is a no-op; with a
    DIFFERENT definition it is an error — changed task means new version."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)

    def register(self, definition: Dict) -> None:
        for key in ("task_id", "version", "benchmark_ref"):
            if key not in definition:
                raise ValueError(f"task definition missing required field: {key}")
        existing = self.get(definition["task_id"], definition["version"])
        if existing is not None:
            core_old = {k: v for k, v in existing.items() if k != "registered_at"}
            if core_old != definition:
                raise ValueError(
                    f"task {definition['task_id']}@v{definition['version']} already "
                    "registered with a different definition — bump the version"
                )
            return
        record = dict(definition)
        record["registered_at"] = now_iso()
        append_jsonl(self.path, record)

    def get(self, task_id: str, version: int) -> Optional[Dict]:
        for rec in read_jsonl(self.path):
            if rec.get("task_id") == task_id and rec.get("version") == version:
                return rec
        return None

    def all(self) -> List[Dict]:
        return read_jsonl(self.path)
