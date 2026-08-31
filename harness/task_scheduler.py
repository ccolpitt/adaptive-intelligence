"""Scheduler — decides which task gets the next training slot.

Phase 0: dumb on purpose (round-robin over registered tasks). Replaced by a
learning-progress scheduler in Phase 3a; the interface is the stable part.
"""

from __future__ import annotations

from typing import List, Sequence

from .interfaces import Task


class RoundRobinScheduler:
    def __init__(self, tasks: Sequence[Task]):
        if not tasks:
            raise ValueError("scheduler needs at least one task")
        self.tasks: List[Task] = list(tasks)
        self._i = 0

    def next_task(self) -> Task:
        task = self.tasks[self._i % len(self.tasks)]
        self._i += 1
        return task
