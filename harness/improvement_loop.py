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
from .promotion_gate import GATE_RULES
from .stats import plateaued


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
    # Diverse training opponents (exp-006): cycle champion, champion,
    # tactical fixture, random — instead of champion-only. Targets the
    # exp-002 finding (self-play data lacks competent threats) at a
    # difficulty the policy can actually learn from (exp-003's lesson).
    diverse_opponents: bool = False
    # Cold start (exp-012): do NOT warm-start the challenger from the
    # champion's weights (champion still serves as opponent, gate, and
    # benchmark incumbent). Needed when the reward scale changes: a
    # warm-started value function calibrated on the old scale fights the
    # new signal (exp-011's suspected failure mechanism).
    cold_start: bool = False
    # Promotion rule — see promotion_gate.GATE_RULES. "strict" (ADR-007,
    # 2026-10-03, human-directed): promote only if the challenger beats the
    # champion head-to-head AND improves on the absolute benchmark; parity
    # rejections are flagged for the hypothesis generator. "either" (ADR-005)
    # and "double" (original) are kept for reproducing earlier campaigns.
    gate_rule: str = "strict"
    # After the run, generate a performance card for the best challenger and
    # a rule-based diagnosis (ADR-007: the hypothesis generator reads the
    # card). Costs ~1-2 min of probing per run.
    auto_card: bool = True
    auto_card_games: int = 60
    # Plateau detection: flag (or stop) when the challenger benchmark's best
    # score has not improved by more than plateau_min_delta in the last
    # plateau_patience iterations. Answers "would more episodes help?" from
    # the curve itself. plateau_action: "flag" | "stop" | "off".
    # CALIBRATION (2026-10-03, harness defect fixed): cold-start runs sit flat
    # for ~8 iterations before breaking out (exp-012: flat through iter 7,
    # 0.116 at iter 11). The original patience=6 with no warm-up would have
    # killed that record run at iter 6 — and DID truncate exp-014/015/016/019
    # and exp-021-seed24. Defaults now: no plateau checks before
    # plateau_min_iterations, and patience 8.
    plateau_patience: int = 8
    plateau_min_delta: float = 0.02
    plateau_min_iterations: int = 12
    plateau_action: str = "flag"
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
        if not cfg.cold_start:
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
        "episode_grad_norm_means": [],
        # Per-episode series (flat across all iterations) — the fine-grained
        # curves the connect4-rl dashboard proved diagnostic: loss/episode,
        # grad-norm/episode, training result/episode (learner's +1/0/-1, a
        # free rolling win-rate signal), cumulative unique states, epsilon.
        "episodes": {
            "loss": [],
            "grad_norm": [],
            "q_mean_abs": [],
            "result": [],
            "unique_states": [],
            "epsilon": [],
        },
    }
    promotions = 0

    diverse = None
    if cfg.diverse_opponents:
        from .scripted_opponents import OneStepWinOpponent, RandomOpponent

        diverse = {
            "tactic": OneStepWinOpponent(seed=cfg.seed + 11),
            "random": RandomOpponent(seed=cfg.seed + 7),
        }

    for iteration in range(cfg.iterations):
        # -- train --------------------------------------------------------
        losses: List[float] = []
        grad_norms: List[float] = []
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
            elif diverse is not None and ep % 4 == 1:
                stats = trainer.play_episode(env, opponent_q=None, opponent_act=diverse["tactic"].act)
            elif diverse is not None and ep % 4 == 3:
                stats = trainer.play_episode(env, opponent_q=None, opponent_act=diverse["random"].act)
            else:
                stats = trainer.play_episode(env, opponent_q=champion_q)
            if stats.losses:
                losses.append(float(np.mean(stats.losses)))
            if stats.grad_norms:
                grad_norms.append(float(np.mean(stats.grad_norms)))
            eps_t = telemetry["episodes"]
            eps_t["loss"].append(float(np.mean(stats.losses)) if stats.losses else None)
            eps_t["grad_norm"].append(
                float(np.mean(stats.grad_norms)) if stats.grad_norms else None
            )
            eps_t["q_mean_abs"].append(
                float(np.mean(stats.q_abs_means)) if stats.q_abs_means else None
            )
            eps_t["result"].append(stats.learner_result)
            eps_t["unique_states"].append(len(trainer.unique_states_seen))
            eps_t["epsilon"].append(round(trainer.eps, 5))
        telemetry["episode_loss_means"].append(float(np.mean(losses)) if losses else None)
        telemetry["episode_grad_norm_means"].append(
            float(np.mean(grad_norms)) if grad_norms else None
        )

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
        gate_fn = GATE_RULES[cfg.gate_rule]
        decision = gate_fn(
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
                "parity": decision.parity,
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

        # -- plateau detection: is more training still buying anything? ------
        bench_history = [it["benchmark_score"] for it in telemetry["iterations"]]
        warmed_up = len(bench_history) >= cfg.plateau_min_iterations
        if cfg.plateau_action != "off" and warmed_up and plateaued(
            bench_history, cfg.plateau_patience, cfg.plateau_min_delta
        ):
            telemetry["iterations"][-1]["plateau"] = True
            note = (
                f"benchmark best has not improved by > {cfg.plateau_min_delta} in the "
                f"last {cfg.plateau_patience} iterations "
                f"(best {max(bench_history):.3f}); more episodes are not helping"
            )
            print(f"[{cfg.experiment_id}] PLATEAU: {note}")
            if cfg.plateau_action == "stop":
                telemetry["plateau_stop"] = {
                    "iteration": iteration,
                    "note": note,
                    "iterations_saved": cfg.iterations - iteration - 1,
                }
                print(
                    f"[{cfg.experiment_id}] stopping early at iter {iteration} "
                    f"(saving {cfg.iterations - iteration - 1} iterations of compute)"
                )
                break
        if mastered:
            print(
                f"[{cfg.experiment_id}] CURRICULUM SIGNAL: {task.ref} benchmark "
                f">= {task.mastery_threshold} — consider adding harder tasks (roadmap Phase 3)."
            )

    # -- diagnose (ADR-007): read the best challenger's card, emit hypotheses --
    if cfg.auto_card and telemetry["iterations"]:
        try:
            from .diagnosis import diagnose_run, render_diagnosis
            from .performance_card import build_card, save_card

            best = max(telemetry["iterations"], key=lambda it: it["benchmark_score"])
            card = build_card(store, best["challenger"], task=task, games=cfg.auto_card_games,
                              seed=cfg.seed)
            save_card(store, card)
            suggestions = diagnose_run(card, telemetry)
            telemetry["diagnosis"] = {
                "policy": best["challenger"],
                "card_worst_status": card.worst_status(),
                "suggestions": [s.to_dict() for s in suggestions],
            }
            print(f"[{cfg.experiment_id}] card for best challenger {best['challenger']} "
                  f"(worst status {card.worst_status()})")
            print(render_diagnosis(suggestions))
        except Exception as err:  # diagnostics must never kill a run
            telemetry["diagnosis"] = {"error": repr(err)}
            print(f"[{cfg.experiment_id}] diagnosis skipped: {err!r}")

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
