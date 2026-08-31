"""Observation / action specs.

Every Task publishes these from day one (roadmap Phase 2, Stage A): per-task
input adapters and output heads are auto-built from the specs, so the trunk
never needs to know which task it is playing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ObsSpec:
    """Shape and dtype of a single observation."""

    shape: Tuple[int, ...]
    dtype: str = "float32"
    description: str = ""


@dataclass(frozen=True)
class ActionSpec:
    """Discrete action space of size ``n``. Legality is state-dependent and
    queried from the environment, not the spec."""

    n: int
    description: str = ""
