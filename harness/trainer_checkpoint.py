"""Keeping a learner's full state so a run can be stopped and continued.

A policy in the archive is its weights, and that is the right definition: a
policy is a thing that plays. But a *learner* is more than its weights. It
also carries the optimiser's per-weight averages, built up over thousands of
steps, and the replay buffer holding the games those steps learned from.

Restoring only the weights therefore does not continue a run, it restarts one
with a head start — and that is measurably worse than continuing. R-007
records what it cost: a lineage resumed in a new process fell from 0.622 to
0.390 on the benchmark and needed 27 cycles, 5,400 games and about four and a
half hours, to climb back to where it had already been. Nothing in the recipe
had changed. The only difference was a cold optimiser and an empty buffer.

So the learner's whole state is written once per cycle and read back on
resume. One file per lineage, overwritten each time: this is not history, it
is a place to stand. The history is the cycle records, which are kept
forever. The file is large (tens of megabytes, mostly replay buffer) and
derived, so it is not committed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import torch


class TrainerCheckpointStore:
    """One resumable checkpoint per lineage."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, lineage_id: str) -> Path:
        return self.root / lineage_id / "trainer.pt"

    def save(self, lineage_id: str, state: Dict, cycle: int) -> Path:
        """Write the checkpoint, via a temporary file so an interrupted write
        cannot leave an unreadable one behind."""
        path = self.path_for(lineage_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".pt.writing")
        torch.save({**state, "cycle": cycle}, temporary)
        temporary.replace(path)
        return path

    def load(self, lineage_id: str) -> Optional[Dict]:
        """The checkpoint, or None if there is not one. A corrupt or
        unreadable file returns None rather than raising: continuing from the
        last archived weights is worse than continuing properly, but far
        better than refusing to run at all."""
        path = self.path_for(lineage_id)
        if not path.exists():
            return None
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return None
