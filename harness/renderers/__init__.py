"""Versioned renderers, bound to a task.

A renderer turns a cycle record into charts. Renderers are versioned and
named inside every record they drew, so a cycle recorded a year ago is drawn
the way it was meant to be drawn even after the current renderer has changed.

When a hypothesis needs a measurement that nothing yet plots, add a new
renderer version BEFORE running the cycle, and point the cycle at it. Older
versions are never edited — that is what keeps old records readable.

Naming: ``<task>_v<version>``, e.g. ``connect4_v1``.
"""

from __future__ import annotations

from typing import Dict, List

from . import connect4_v1, connect4_v2

_RENDERERS: Dict[str, object] = {
    "connect4_v1": connect4_v1,
    "connect4_v2": connect4_v2,  # adds win rate against a random opponent
}

LATEST: Dict[str, str] = {
    "connect4": "connect4_v2",
}


def get_renderer(name: str):
    if name not in _RENDERERS:
        raise ValueError(
            f"unknown renderer {name!r}. Available: {sorted(_RENDERERS)}. "
            "A record naming a renderer that does not exist means the renderer "
            "was deleted or renamed — restore it rather than repointing the "
            "record, so the record keeps its original meaning."
        )
    return _RENDERERS[name]


def latest_for(task_id: str) -> str:
    if task_id not in LATEST:
        raise ValueError(f"no renderer registered for task {task_id!r}")
    return LATEST[task_id]


def available() -> List[str]:
    return sorted(_RENDERERS)
