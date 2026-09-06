"""The improvement loop: train -> evaluate -> select -> archive -> repeat.

One run of ``run_loop`` = one registered experiment. The hypothesis is
registered BEFORE training starts; every eval score is stamped with
(policy, task@v, benchmark@v); promotion is decided by the double gate; every
challenger is archived win or lose, with an epitaph. The loop never records
the experiment's verdict — verdicts are condition-scoped scientific judgments
recorded once, by a human (or an explicitly configured rule), via
``ExperimentRegistry.record_verdict``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .agents.dqn import DQNTrainer, TrainerConfig, seed_everything
from .policy_archive import Archive
from .policy_evaluator import Evaluator
from .interfaces import QFunction, Task
from .experiment_registry import ExperimentRegistry
from .stats import Z_95
from .task_registry import TaskRegistry
from .promotion_gate import statistical_double_gate


class Store:
    """Everything persistent lives under one root: archive + registries +
    telemetry."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.archive = Archive(self.root / "archive")
        self.tasks = TaskRegistry(self.root / "registry" / "tasks.jsonl")
        self.experiments = ExperimentRegistry(self.root / "registry" / "experiments.jsonl")
        self.telemetry_dir = self.root / "telemetry"
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class LoopConfig:
    experiment_id: str
    hypothesis: str
    changes: str
    prior_art: Union[List[str], str] = "none"
    tags: List[str] = field(default_factory=list)
    seed: int = 0
    iterations: int = 5
    episodes_per_iteration: int = 200
    gate_games: int = 200
    gate_temperature: float = 0.3
    # One-sided confidence controls: z=1.645 -> ~5% false-positive rate per
    # gate decision. False negatives shrink with gate_games (see stats.py).
    z_promote: float = Z_95
    z_regress: float = Z_95
    # Fraction of training episodes played against the solver at the
    # champion's current benchmark frontier depth (0.0 = pure self-play,
    # the baseline). A component change: flip it only inside a registered
    # experiment.
    frontier_opponent_fraction: float = 0.0
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


def _module_q_function(module: torch.jit.ScriptModule) -> QFunction:
    def q(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(obs).float().unsqueeze(0)
            return module(t).squeeze(0).numpy()

    return q


def run_loop(task: Task, store: Store, cfg: LoopConfig) -> Dict:
    seed_everything(cfg.seed)
    cfg.trainer.seed = cfg.seed

    # -- register: task first, then the experiment (hypothesis before run) --
    store.tasks.register(task.definition())

    incumbent_id = store.archive.current_champion(task.task_id)
    store.experiments.register(
        experiment_id=cfg.experiment_id,
        hypothesis=cfg.hypothesis,
        changes=cfg.changes,
        conditions={
            "task_ref": task.ref,
            "benchmark_ref": task.benchmark.ref,
            "config": asdict(cfg),
            "seed": cfg.seed,
        },
        prior_art=cfg.prior_art,
        input_policies=[incumbent_id] if incumbent_id else [],
        tags=cfg.tags,
    )

    evaluator = Evaluator(task, seed=cfg.seed)
    rows, cols = task.obs_spec.shape[1], task.obs_spec.shape[2]
    trainer = DQNTrainer(rows, cols, task.action_spec.n, cfg.trainer)

    # -- champion: resume from archive, or archive a fresh gen-0 ----------
    if incumbent_id is not None:
        champion_module, champion_meta = store.archive.load_policy(incumbent_id)
        champion_id = incumbent_id
        try:
            trainer.net.load_state_dict(champion_module.state_dict())
        except RuntimeError:
            trainer.net.load_state_dict(champion_module.state_dict(), strict=False)
        trainer.target.load_state_dict(trainer.net.state_dict())
        champion_q = _module_q_function(champion_module)
        # Scores are only comparable within one benchmark version. If the
        # incumbent was last scored under a different benchmark, re-measure
        # it on the CURRENT one rather than comparing apples to oranges.
        prior = next(
            (
                r
                for r in reversed(champion_meta["eval_records"])
                if r.get("benchmark_ref") == task.benchmark.ref
            ),
            None,
        )
        if prior is not None:
            champion_bench, champion_detail = prior["score"], prior.get("detail", {})
        else:
            rebench = task.benchmark.score(champion_q)
            champion_bench, champion_detail = rebench.score, rebench.detail
            print(
                f"[{cfg.experiment_id}] incumbent {champion_id} re-benchmarked on "
                f"{task.benchmark.ref}: {champion_bench:.3f}"
            )
    else:
        champion_id = store.archive.next_policy_id(task.task_id)
        bench = task.benchmark.score(trainer.q_function())
        scripted = trainer.scripted()
        store.archive.save_policy(
            champion_id,
            scripted,
            metadata={
                "task_ref": task.ref,
                "experiment_id": cfg.experiment_id,
                "parent_id": None,
                "generation": 0,
                "training_config": asdict(cfg.trainer),
                "eval_records": [
                    {
                        "task_ref": task.ref,
                        "benchmark_ref": bench.benchmark_ref,
                        "score": bench.score,
                        "detail": bench.detail,
                    }
                ],
            },
        )
        store.experiments.add_output_policy(cfg.experiment_id, champion_id)
        store.archive.record_promotion(task.task_id, champion_id, "initial champion (gen 0)")
        store.archive.record_epitaph(champion_id, "gen 0: untrained seed champion")
        champion_bench, champion_detail = bench.score, bench.detail
        champion_module = trainer.scripted()
        champion_q = _module_q_function(champion_module)

    generation = store.archive.get_entry(champion_id).get("generation", 0)
    bench_games = champion_detail.get("games_total", champion_detail.get("games", 100))

    # Optional frontier-solver training opponent (component change, gated by
    # config; see LoopConfig.frontier_opponent_fraction).
    def frontier_opponent_act():
        from .tasks.connect4_solver import SolverOpponent

        depth = int(champion_detail.get("frontier_depth", 1))
        return SolverOpponent(depth=depth), depth

    telemetry: Dict = {
        "experiment_id": cfg.experiment_id,
        "task_ref": task.ref,
        "iterations": [],
        "episode_loss_means": [],
    }
    promotions = 0

    for iteration in range(cfg.iterations):
        # -- train --------------------------------------------------------
        losses: List[float] = []
        solver_opp, frontier_depth = (
            frontier_opponent_act() if cfg.frontier_opponent_fraction > 0 else (None, None)
        )
        period = (
            max(1, round(1 / cfg.frontier_opponent_fraction))
            if cfg.frontier_opponent_fraction > 0
            else 0
        )
        for ep in range(cfg.episodes_per_iteration):
            env = task.make_env()
            if solver_opp is not None and period and ep % period == 0:
                stats = trainer.play_episode(env, opponent_q=None, opponent_act=solver_opp.act)
            else:
                stats = trainer.play_episode(env, opponent_q=champion_q)
            if stats.losses:
                losses.append(float(np.mean(stats.losses)))
        telemetry["episode_loss_means"].append(float(np.mean(losses)) if losses else None)

        # -- evaluate -------------------------------------------------------
        challenger_id = store.archive.next_policy_id(task.task_id)
        h2h = evaluator.head_to_head(
            trainer.q_function(),
            champion_q,
            games=cfg.gate_games,
            label_a=challenger_id,
            label_b=champion_id,
            temperature=cfg.gate_temperature,
        )
        bench = task.benchmark.score(trainer.q_function())

        # -- select ---------------------------------------------------------
        decision = statistical_double_gate(
            challenger_h2h_score=h2h.score,
            h2h_games=h2h.games,
            challenger_benchmark=bench.score,
            challenger_benchmark_games=bench.detail.get("games_total", bench_games),
            champion_benchmark=champion_bench,
            champion_benchmark_games=bench_games,
            z_promote=cfg.z_promote,
            z_regress=cfg.z_regress,
        )

        # -- archive (win or lose) -------------------------------------------
        store.archive.save_policy(
            challenger_id,
            trainer.scripted(),
            metadata={
                "task_ref": task.ref,
                "experiment_id": cfg.experiment_id,
                "parent_id": champion_id,
                "generation": generation + 1,
                "training_config": asdict(cfg.trainer),
                "eval_records": [
                    {
                        "task_ref": task.ref,
                        "benchmark_ref": "head-to-head",
                        "opponent": champion_id,
                        "score": h2h.score,
                        "detail": {"wins": h2h.wins, "losses": h2h.losses, "draws": h2h.draws,
                                   "games": h2h.games},
                    },
                    {
                        "task_ref": task.ref,
                        "benchmark_ref": bench.benchmark_ref,
                        "score": bench.score,
                        "detail": bench.detail,
                    },
                ],
            },
        )
        store.experiments.add_output_policy(cfg.experiment_id, challenger_id)
        store.archive.record_epitaph(challenger_id, decision.reason)

        if decision.promote:
            store.archive.record_promotion(task.task_id, challenger_id, decision.reason)
            champion_id = challenger_id
            champion_bench, champion_detail = bench.score, bench.detail
            champion_module = trainer.scripted()
            champion_q = _module_q_function(champion_module)
            generation += 1
            promotions += 1
            # Re-explore against the new champion (connect4-rl trick).
            trainer.eps = min(cfg.trainer.eps_start, trainer.eps + 0.1)

        mastered = champion_bench >= task.mastery_threshold
        frontier = bench.detail.get("frontier_depth")
        telemetry["iterations"].append(
            {
                "iteration": iteration,
                "challenger": challenger_id,
                "h2h_score_vs_champion": h2h.score,
                "h2h_games": h2h.games,
                "benchmark_score": bench.score,
                "benchmark_detail": bench.detail,
                "champion_benchmark": champion_bench,
                "promoted": decision.promote,
                "reason": decision.reason,
                "epsilon": trainer.eps,
                "frontier_training_depth": frontier_depth,
                "dormant_neurons": trainer.dormant_neuron_fraction(),
                "mastery": {task.ref: mastered},
            }
        )
        flag = "MASTERED" if mastered else "in progress"
        frontier_txt = f" frontier_depth={frontier}" if frontier is not None else ""
        print(
            f"[{cfg.experiment_id}] iter {iteration}: h2h={h2h.score:.2f}/{h2h.games} "
            f"bench={bench.score:.3f}{frontier_txt} promoted={decision.promote} | "
            f"{task.ref}: {flag}"
        )
        if mastered:
            print(
                f"[{cfg.experiment_id}] CURRICULUM SIGNAL: {task.ref} benchmark "
                f">= {task.mastery_threshold} — consider adding harder tasks (roadmap Phase 3)."
            )

    telemetry_path = store.telemetry_dir / f"{cfg.experiment_id}.json"
    telemetry_path.write_text(json.dumps(telemetry, indent=2))

    summary = {
        "experiment_id": cfg.experiment_id,
        "champion": champion_id,
        "champion_benchmark": champion_bench,
        "promotions": promotions,
        "iterations": cfg.iterations,
        "telemetry": str(telemetry_path),
    }
    print(
        f"[{cfg.experiment_id}] done: champion={champion_id} bench={champion_bench:.3f} "
        f"promotions={promotions}. Record the verdict with harness.verdict when judged."
    )
    return summary
