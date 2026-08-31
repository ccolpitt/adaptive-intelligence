"""Core Phase 0 contracts: Task, Benchmark, Opponent.

A *task* is a contract with four parts (roadmap Phase 0): an observation
space, an action space, an environment mapping (state, action) -> (next
observation, reward, done), and a success criterion measured by a fixed
external benchmark. The harness talks only to these shapes — it never knows
it is playing Connect 4.

A *benchmark* is a versioned, immutable measurement procedure
``score: Policy -> float`` (roadmap Phase 1). It is NOT a policy; its fixed
opposition may be an oracle, a frozen policy, a script, or a test suite.
Nothing inside it moves when the population moves.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

from .task_specs import ActionSpec, ObsSpec


class Environment(Protocol):
    """Minimal two-player, perspective-flipped environment protocol.

    ``reset`` returns the first observation (mover's perspective).
    ``step`` returns (next_obs from the NEXT mover's perspective,
    reward from the MOVING player's perspective, done, winner) where
    winner is +1 / -1 / 0 (draw) / None (ongoing).
    """

    def reset(self) -> np.ndarray: ...

    def step(self, action: int) -> "StepResult": ...

    def legal_actions(self) -> List[int]: ...


@dataclass(frozen=True)
class StepResult:
    next_obs: np.ndarray
    reward: float
    done: bool
    winner: Optional[int]  # +1, -1, 0 = draw, None = ongoing


class Opponent(abc.ABC):
    """A fixed, scripted or frozen player used by evaluators and benchmarks."""

    name: str = "opponent"

    @abc.abstractmethod
    def act(self, obs: np.ndarray, legal: Sequence[int]) -> int: ...


# A policy, at inference time, is just: observation -> action-values.
QFunction = Callable[[np.ndarray], np.ndarray]


class Benchmark(abc.ABC):
    """Versioned, immutable measurement procedure. score: policy -> [0, 1]."""

    benchmark_id: str
    version: int

    @property
    def ref(self) -> str:
        return f"{self.benchmark_id}@v{self.version}"

    @abc.abstractmethod
    def score(self, q_function: QFunction) -> "BenchmarkResult": ...


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark_ref: str
    score: float  # aggregate in [0, 1]
    detail: Dict[str, float] = field(default_factory=dict)


class Task(abc.ABC):
    """Versioned bundle: environment + reward semantics + fixed benchmark.

    Changing any part -> new task version. Same environment + different
    reward = different task.
    """

    task_id: str
    version: int

    @property
    def ref(self) -> str:
        return f"{self.task_id}@v{self.version}"

    @property
    @abc.abstractmethod
    def obs_spec(self) -> ObsSpec: ...

    @property
    @abc.abstractmethod
    def action_spec(self) -> ActionSpec: ...

    @abc.abstractmethod
    def make_env(self) -> Environment: ...

    @property
    @abc.abstractmethod
    def benchmark(self) -> Benchmark: ...

    @property
    def mastery_threshold(self) -> float:
        """Benchmark score at or above which this task counts as mastered
        (curriculum-exhaustion protocol, roadmap Phase 3)."""
        return 0.95

    def definition(self) -> Dict:
        """Serializable definition for the task registry."""
        return {
            "task_id": self.task_id,
            "version": self.version,
            "obs_spec": {"shape": list(self.obs_spec.shape), "dtype": self.obs_spec.dtype},
            "action_spec": {"n": self.action_spec.n},
            "benchmark_ref": self.benchmark.ref,
            "mastery_threshold": self.mastery_threshold,
        }
