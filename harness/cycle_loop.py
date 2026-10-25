"""The improvement loop, recording cycles.

One **cycle**: train for a declared number of games, measure the result, and
decide whether it replaces the champion. Everything measured goes into a
cycle record (data, with the renderer that draws it named inside), and
everything that happened goes into the lineage's log.

This is the path for new lineages. The earlier loop (`improvement_loop.py`)
is kept so the 36 experiments recorded under it stay reproducible.

What is measured each cycle:
  - per training game: error, size of the weight correction, magnitude of the
    value estimates, the game's result, distinct positions seen so far, the
    chance of a random move, and how much lookahead was spent;
  - at the end: the score on the task's frozen evaluation, the win rate
    against a random opponent, the result against the champion, whether the
    policy values already-decided positions correctly, and whether any part
    of the network is dying or blowing up.

Training may use lookahead, as the rubric declares. Every measurement is
taken with the policy playing on its own, with no lookahead, because that is
what the task's evaluation definition requires.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch

from .agents.dqn import DQNTrainer, TrainerConfig, seed_everything
from .cycle_record import CycleRecord, CycleStore
from .interfaces import QFunction, Task
from .lineage import Lineage, LineageStore, RubricStore, TrainingRubric
from .policy_archive import Archive
from .policy_evaluator import Evaluator
from .promotion_gate import GATE_RULES, apply_health_gate
from .renderers import latest_for
from .scripted_opponents import RandomOpponent
from .stats import Z_95


@dataclass
class CycleLoopConfig:
    lineage_id: str
    rubric_id: str
    games_per_cycle: int = 400
    cycles: int = 10
    gate_games: int = 200
    gate_temperature: float = 0.3
    random_eval_games: int = 100
    gate_rule: str = "strict"
    z_promote: float = Z_95
    z_regress: float = Z_95
    probe_games: int = 30
    renderer: Optional[str] = None  # defaults to the task's latest
    experiment_id: Optional[str] = None
    seed: int = 0
    # Lookahead the policy is allowed WHEN BEING SCORED, as a number of states
    # per move. DeepMind's systems searched at play time as well as during
    # training, so a modest allowance here is the faithful comparison. Zero
    # keeps the original definition of the score. Both are measured every
    # cycle; eval_budget_decides selects which one the promotion gate uses.
    eval_budget: int = 0
    eval_budget_decides: bool = False
    # Lookahead in the challenger-versus-champion match. Both sides get the
    # SAME budget, which is what keeps the comparison about skill rather than
    # compute (learning rule 4). Zero means both play immediately. This is
    # sound for a two-player zero-sum game, where one policy's gain is the
    # other's loss and an equal allowance is therefore a fair contest.
    h2h_budget: int = 0
    # Resuming: a run that is interrupted can be continued, because cycle
    # records and the champion both live on disk. The weights of the policy
    # being trained are reloaded from the last challenger of the lineage.
    resume: bool = True
    # Data rules for the trainer that the rubric does not already fix.
    store_opponent_transitions: bool = True
    diverse_opponents: bool = True
    verbose: bool = True


def _q_of(module: torch.jit.ScriptModule) -> QFunction:
    """Move preference from a saved policy, whichever kind it is. A
    single-headed policy returns one array; a two-headed one returns move
    scores and a position value, and it is the move scores that choose."""
    def q(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = module(torch.from_numpy(obs).float().unsqueeze(0))
            if isinstance(out, tuple):
                out = out[0]
            return out.squeeze(0).numpy()
    return q


def _make_trainer(rubric: TrainingRubric, task: Task, seed: int, cfg):
    """Build the learner the rubric names."""
    rows, cols = task.obs_spec.shape[1], task.obs_spec.shape[2]
    if rubric.learner == "policy-value":
        from .agents.policy_value import PolicyValueConfig, PolicyValueTrainer

        h = rubric.hyperparameters
        pv = PolicyValueConfig(
            seed=seed,
            lr=h.get("lr", 1e-3),
            weight_decay=h.get("weight_decay", 1e-4),
            batch_size=h.get("batch_size", 128),
            buffer_capacity=h.get("buffer_capacity", 100_000),
            train_steps_per_game=h.get("train_steps_per_game", 4),
            channels=h.get("channels", 64),
            hidden=h.get("hidden", 128),
            conv_layers=h.get("conv_layers", 3),
            search_budget=rubric.search_budget,
            spend_rule=rubric.spend_rule,
            c_puct=h.get("c_puct", 1.4),
            gamma=h.get("gamma", 0.99),
            dirichlet_noise=h.get("dirichlet_noise", 0.25),
            sampling_moves=h.get("sampling_moves", 12),
            value_loss_weight=h.get("value_loss_weight", 1.0),
            mirror_augmentation=h.get("mirror_augmentation", True),
        )
        return PolicyValueTrainer(rows, cols, task.action_spec.n, pv), "policy-value"

    trainer_cfg = trainer_config_from_rubric(
        rubric, seed, store_opponent_transitions=cfg.store_opponent_transitions)
    return DQNTrainer(rows, cols, task.action_spec.n, trainer_cfg), "dqn"


def _play_one(trainer, kind: str, env, champion_q, opponent_act):
    """Play one training game and return per-game numbers in one shape,
    whichever learner produced them."""
    if kind == "policy-value":
        # A two-headed policy learns from its own search, so it plays itself
        # unless a fixed opponent is named.
        stats = trainer.play_game(env, opponent_act=opponent_act)
        loss = float(np.mean(stats.total_losses)) if stats.total_losses else None
        return {
            "loss": loss,
            "policy_loss": float(np.mean(stats.policy_losses)) if stats.policy_losses else None,
            "value_loss": float(np.mean(stats.value_losses)) if stats.value_losses else None,
            "grad_norm": float(np.mean(stats.grad_norms)) if stats.grad_norms else None,
            "q_mean_abs": float(np.mean(stats.value_abs_means)) if stats.value_abs_means else None,
            "result": stats.learner_result,
            "search_states": stats.search_states,
            "moves": stats.moves,
        }
    stats = trainer.play_episode(env, opponent_q=champion_q, opponent_act=opponent_act)
    return {
        "loss": float(np.mean(stats.losses)) if stats.losses else None,
        "policy_loss": None,
        "value_loss": None,
        "grad_norm": float(np.mean(stats.grad_norms)) if stats.grad_norms else None,
        "q_mean_abs": float(np.mean(stats.q_abs_means)) if stats.q_abs_means else None,
        "result": stats.learner_result,
        "search_states": None,
        "moves": stats.moves,
    }


def trainer_config_from_rubric(rubric: TrainingRubric, seed: int,
                               **overrides) -> TrainerConfig:
    """Turn a published recipe into the trainer's settings, so a cycle cannot
    silently train under something other than its declared rubric."""
    h = rubric.hyperparameters
    cfg = TrainerConfig(
        seed=seed,
        lr=h.get("lr", 1e-3),
        weight_decay=h.get("weight_decay", 1e-4),
        gamma=h.get("gamma", 0.99),
        batch_size=h.get("batch_size", 64),
        buffer_capacity=rubric.data_rules.get("buffer_capacity", 20000)
        if isinstance(rubric.data_rules.get("buffer_capacity"), int) else 50000,
        train_steps_per_episode=h.get("train_steps_per_episode", 2),
        target_update_every=h.get("target_update_every", 50),
        eps_start=h.get("eps_start", 0.5),
        eps_end=h.get("eps_end", 0.1),
        eps_decay=h.get("eps_decay", 0.999),
        channels=h.get("channels", 32),
        hidden=h.get("hidden", 128),
        conv_layers=h.get("conv_layers", 2),
        search_budget=rubric.search_budget,
        spend_rule=rubric.spend_rule if rubric.search_budget else "puct",
        c_puct=h.get("c_puct", 1.4),
        prior_temperature=h.get("prior_temperature", 0.5),
        search_targets=rubric.target_kind == "search-value",
        bounded_targets=rubric.bounded_targets,
        bounded_output=bool(h.get("bounded_output", False)),
        tactical_shaping=bool(rubric.data_rules.get("tactical_shaping") is True),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)

    # A rubric is a promise about how a policy was trained. Anything it
    # specifies that does not reach the trainer makes that promise false, and
    # a silent mismatch is worse than a crash: exp-025 asked for bounded value
    # estimates, did not get them, and the run looked plausible anyway.
    unapplied = [
        key for key, value in h.items()
        if hasattr(cfg, key) and getattr(cfg, key) != value
        and key not in overrides
    ]
    if unapplied:
        raise ValueError(
            f"rubric {rubric.rubric_id} specifies {unapplied} but the trainer was "
            f"configured differently — the rubric would not describe what actually "
            f"trained the policy. Map these settings in "
            f"trainer_config_from_rubric before running."
        )
    return cfg


def win_rate_vs_random(task: Task, q_function: QFunction, games: int = 100,
                       seed: int = 7) -> float:
    """The floor every competent policy should clear. Played with the policy
    alone, no lookahead, alternating who moves first."""
    opponent = RandomOpponent(seed=seed)
    score = 0.0
    for g in range(games):
        env = task.make_env()
        obs = env.reset()
        policy_first = g % 2 == 0
        ply = 0
        while True:
            legal = env.legal_actions()
            policy_to_move = (env.current_player == 1) == policy_first
            if policy_to_move:
                q = np.asarray(q_function(obs), dtype=np.float64)
                masked = np.full(len(q), -np.inf)
                masked[legal] = np.nan_to_num(q[legal], nan=-np.inf)
                action = int(np.argmax(masked))
            else:
                action = opponent.act(obs, legal)
            result = env.step(action)
            ply += 1
            if result.done:
                if result.winner == 0:
                    score += 0.5
                elif (result.winner == 1) == policy_first:
                    score += 1.0
                break
            obs = result.next_obs
    return score / max(games, 1)


def probe_policy(task: Task, q_function: QFunction, module: torch.nn.Module,
                 games: int = 30, seed: int = 0,
                 value_function: Optional[Callable] = None) -> Dict:
    """Does the policy value positions correctly, and is the network healthy?
    Reuses the established probes and flattens them into plain numbers for
    the cycle record."""
    from .performance_card import collect_rollouts, solver_calibration_readings

    probes: Dict = {}
    harvest = collect_rollouts(task, q_function, games=games, seed=seed)
    if harvest.win_move_qs:
        probes["q_of_game_winning_moves"] = float(np.mean(harvest.win_move_qs))
    if harvest.defense_nonblock_q:
        probes["q_of_not_blocking"] = float(np.mean(harvest.defense_nonblock_q))
    if harvest.offense_n:
        probes["take_the_win_rate"] = harvest.offense_taken / harvest.offense_n
    if harvest.defense_n:
        probes["block_the_loss_rate"] = harvest.defense_blocked / harvest.defense_n
    probes["unique_states_in_probe"] = harvest.unique_states

    if value_function is None:
        # One set of outputs doing both jobs: its magnitudes ARE value
        # estimates, so they belong in the reward range.
        probes["q_mean_abs"] = float(harvest.q_mean_abs)
        probes["q_max_abs"] = float(harvest.q_max_abs)
    else:
        # A two-headed policy. The move head's numbers are scores that become
        # probabilities, so they are not values and are not confined to the
        # reward range — a large one simply means a strong preference.
        # Checking them against ±1 would be a category error, so the value
        # range is checked on the head that actually estimates value.
        probes["move_score_mean_abs"] = float(harvest.q_mean_abs)
        probes["move_score_max_abs"] = float(harvest.q_max_abs)
        values = []
        for g in range(max(4, games // 4)):
            env = task.make_env()
            obs = env.reset()
            rng = np.random.default_rng(seed + 500 + g)
            while True:
                values.append(abs(float(value_function(obs))))
                legal = env.legal_actions()
                step = env.step(int(rng.choice(legal)))
                if step.done:
                    break
                obs = step.next_obs
        if values:
            probes["q_mean_abs"] = float(np.mean(values))
            probes["q_max_abs"] = float(np.max(values))

    for reading in solver_calibration_readings(task, q_function,
                                               games=max(4, games // 4), seed=seed):
        if reading.value is not None:
            probes[reading.name] = float(reading.value)

    # Per-layer weight health, which is what the renderer draws.
    distribution: Dict[str, Dict[str, float]] = {}
    for name, param in module.named_parameters():
        if param.dim() <= 1:
            continue
        w = param.detach().float().reshape(-1)
        finite = w[torch.isfinite(w)]
        if finite.numel() == 0:
            continue
        distribution[name] = {
            "dead_fraction": float((finite.abs() < 0.01).float().mean().item()),
            "exploding_fraction": float((finite.abs() > 10.0).float().mean().item()),
            "std": float(finite.std().item()) if finite.numel() > 1 else 0.0,
            "max_abs": float(finite.abs().max().item()),
            "nonfinite": int((~torch.isfinite(w)).sum().item()),
        }
    probes["weight_distribution"] = distribution
    return probes


def run_cycles(task: Task, store_root: Union[str, Path],
               cfg: CycleLoopConfig) -> List[CycleRecord]:
    root = Path(store_root)
    archive = Archive(root / "archive")
    cycles_store = CycleStore(root / "cycles")
    lineages = LineageStore(root / "lineages")
    rubrics = RubricStore(root / "rubrics")

    rubric = rubrics.get(cfg.rubric_id)
    if rubric is None:
        raise ValueError(
            f"rubric {cfg.rubric_id!r} is not published. Publish the recipe "
            "before training under it, so the policy can point at what made it."
        )
    lineage = lineages.get(cfg.lineage_id)
    if lineage is None:
        raise ValueError(
            f"lineage {cfg.lineage_id!r} does not exist — create it first, "
            "declaring the task and the rubric it trains under."
        )
    if lineage.rubric_id != cfg.rubric_id:
        lineages.record(cfg.lineage_id, "rubric_changed", rubric_id=cfg.rubric_id,
                        reason="run_cycles called with a different rubric")

    renderer = cfg.renderer or latest_for(lineage.task_id)
    seed_everything(cfg.seed)
    rows, cols = task.obs_spec.shape[1], task.obs_spec.shape[2]
    trainer, learner_kind = _make_trainer(rubric, task, cfg.seed, cfg)
    trainer_cfg = trainer.cfg
    parameter_count = sum(p.numel() for p in trainer.net.parameters())
    evaluator = Evaluator(task, seed=cfg.seed)
    gate = GATE_RULES[cfg.gate_rule]

    diverse = None
    if cfg.diverse_opponents:
        from .scripted_opponents import OneStepWinOpponent

        diverse = {"tactic": OneStepWinOpponent(seed=cfg.seed + 11),
                   "random": RandomOpponent(seed=cfg.seed + 7)}

    # The champion. A lineage that starts from scratch archives its untrained
    # network as the first champion, so there is always something to compare
    # against and the starting point is on the record.
    champion_id = archive.current_champion(lineage.lineage_id)
    if champion_id is None:
        champion_id = archive.next_policy_id(lineage.lineage_id)
        bench0 = task.benchmark.score(trainer.q_function())
        archive.save_policy(
            champion_id, trainer.scripted(),
            metadata={"task_ref": task.ref, "lineage_id": lineage.lineage_id,
                      "rubric_id": cfg.rubric_id, "experiment_id": cfg.experiment_id,
                      "parent_id": None, "generation": 0,
                      "training_config": asdict(trainer_cfg),
                      "eval_records": [{"task_ref": task.ref,
                                        "benchmark_ref": bench0.benchmark_ref,
                                        "score": bench0.score,
                                        "detail": bench0.detail}]},
        )
        archive.record_promotion(lineage.lineage_id, champion_id,
                                 "starting point: untrained network")
        archive.record_epitaph(champion_id, "generation 0, untrained")
        lineages.record(lineage.lineage_id, "policy_added", policy_id=champion_id,
                        cycle=-1, generation=0)
        lineages.record(lineage.lineage_id, "promoted", policy_id=champion_id,
                        cycle=-1, score=bench0.score, reason="starting point")
        champion_bench, champion_detail = bench0.score, bench0.detail
        if cfg.verbose:
            print(f"[{lineage.lineage_id}] starting point {champion_id}: "
                  f"score {bench0.score:.3f}")
    else:
        _, champion_meta = archive.load_policy(champion_id)
        record = [r for r in champion_meta.get("eval_records", [])
                  if "benchmark" in str(r.get("benchmark_ref"))]
        champion_bench = record[-1]["score"] if record else 0.0
        champion_detail = record[-1].get("detail", {}) if record else {}

    champion_module, _ = archive.load_policy(champion_id)
    champion_q = _q_of(champion_module)
    generation = archive.get_entry(champion_id).get("generation", 0)

    start_cycle = cycles_store.next_cycle_index(lineage.lineage_id)
    cumulative = cycles_store.cumulative_games(lineage.lineage_id)

    # Resuming an interrupted run. Cycle records and the champion are already
    # on disk; what is missing is the network being trained, which is not the
    # champion unless the last cycle promoted. Reload the most recent
    # challenger so training continues rather than silently restarting.
    if cfg.resume and start_cycle > 0:
        previous = cycles_store.load(lineage.lineage_id, start_cycle - 1)
        if previous is not None and previous.challenger_id:
            try:
                last_module, _ = archive.load_policy(previous.challenger_id)
                trainer.load_weights_from(last_module)
                if cfg.verbose:
                    print(f"[{lineage.lineage_id}] resuming at cycle {start_cycle} "
                          f"from {previous.challenger_id} "
                          f"({cumulative:,} games already trained)")
            except Exception as err:
                print(f"[{lineage.lineage_id}] could not reload "
                      f"{previous.challenger_id} ({err!r}); continuing from the "
                      f"current weights")
        # The replay buffer and exploration rate are not persisted, so a
        # resumed run restarts both. Recorded here because it makes a resumed
        # cycle slightly different from an uninterrupted one.
        if hasattr(trainer.cfg, "eps_start"):
            trainer.eps = max(trainer.cfg.eps_end, trainer.cfg.eps_start * 0.5)
        lineages.record(lineage.lineage_id, "resumed", cycle=start_cycle,
                        from_policy=previous.challenger_id if previous else None,
                        note="replay buffer and exploration rate restart on resume")
    produced: List[CycleRecord] = []

    for offset in range(cfg.cycles):
        cycle_index = start_cycle + offset
        t_start = time.time()
        series: Dict[str, List] = {k: [] for k in
                                   ("loss", "grad_norm", "q_mean_abs", "result",
                                    "unique_states", "epsilon", "search_states",
                                    # moves per game: the denominator that turns
                                    # positions-per-game into positions-per-move,
                                    # which is the unit the budget is stated in
                                    "moves")}
        states_before = trainer.search_states_examined

        for game in range(cfg.games_per_cycle):
            env = task.make_env()
            opponent_act = None
            if diverse is not None and game % 4 == 1:
                opponent_act = diverse["tactic"].act
            elif diverse is not None and game % 4 == 3:
                opponent_act = diverse["random"].act
            measured = _play_one(trainer, learner_kind, env, champion_q, opponent_act)
            for key in ("loss", "grad_norm", "q_mean_abs", "result"):
                series[key].append(measured[key])
            for key in ("policy_loss", "value_loss"):
                if measured[key] is not None:
                    series.setdefault(key, []).append(measured[key])
            series["unique_states"].append(len(trainer.unique_states_seen))
            series["epsilon"].append(round(getattr(trainer, "eps", 0.0), 5))
            spent = trainer.search_states_examined - states_before
            states_before = trainer.search_states_examined
            series["search_states"].append(
                spent if trainer.cfg.search_budget else None)
            series["moves"].append(measured.get("moves"))

        # Measurement. The policy alone first, which is the definition every
        # score in the project's history was produced under.
        bare_q = trainer.q_function()
        bench = task.benchmark.score(bare_q)
        wr_random = win_rate_vs_random(task, bare_q, cfg.random_eval_games,
                                       seed=cfg.seed + 101)

        # Then, if allowed, the same policy with a modest amount of thinking.
        bench_with_lookahead = None
        if cfg.eval_budget > 0:
            from .budgeted_search import make_search

            thinking_q = make_search(bare_q, budget=cfg.eval_budget,
                                     spend_rule="puct", n_actions=task.action_spec.n,
                                     rows=rows, cols=cols, dirichlet_noise=0.0,
                                     seed=cfg.seed)
            bench_with_lookahead = task.benchmark.score(thinking_q)

        challenger_id = archive.next_policy_id(lineage.lineage_id)

        # The match against the champion. With a budget, BOTH sides get the
        # same allowance, so what is being compared stays skill and not
        # compute.
        if cfg.h2h_budget > 0:
            from .budgeted_search import make_search

            challenger_player = make_search(
                bare_q, budget=cfg.h2h_budget, spend_rule="puct",
                n_actions=task.action_spec.n, rows=rows, cols=cols,
                dirichlet_noise=0.0, seed=cfg.seed, select_by="visits")
            champion_player = make_search(
                champion_q, budget=cfg.h2h_budget, spend_rule="puct",
                n_actions=task.action_spec.n, rows=rows, cols=cols,
                dirichlet_noise=0.0, seed=cfg.seed + 1, select_by="visits")
        else:
            challenger_player, champion_player = bare_q, champion_q

        h2h = evaluator.head_to_head(challenger_player, champion_player,
                                     games=cfg.gate_games,
                                     label_a=challenger_id, label_b=champion_id,
                                     temperature=cfg.gate_temperature)

        # Which score the gate uses. Both are recorded either way, so the
        # choice never destroys a comparison; it only decides what counts.
        deciding = (bench_with_lookahead
                    if (cfg.eval_budget_decides and bench_with_lookahead is not None)
                    else bench)
        decision = gate(
            challenger_h2h_score=h2h.score, h2h_games=h2h.games,
            challenger_benchmark=deciding.score,
            challenger_benchmark_games=deciding.detail.get("games_total", 168),
            champion_benchmark=champion_bench, champion_benchmark_games=168,
            z_promote=cfg.z_promote, z_regress=cfg.z_regress,
        )

        archive.save_policy(
            challenger_id, trainer.scripted(),
            metadata={"task_ref": task.ref, "lineage_id": lineage.lineage_id,
                      "rubric_id": cfg.rubric_id, "experiment_id": cfg.experiment_id,
                      "parent_id": champion_id, "generation": generation + 1,
                      "cycle": cycle_index,
                      "training_config": asdict(trainer_cfg),
                      "eval_records": [
                          {"task_ref": task.ref, "benchmark_ref": "head-to-head",
                           "opponent": champion_id, "score": h2h.score,
                           "detail": {"wins": h2h.wins, "losses": h2h.losses,
                                      "draws": h2h.draws, "games": h2h.games}},
                          {"task_ref": task.ref, "benchmark_ref": bench.benchmark_ref,
                           "score": bench.score, "detail": bench.detail},
                          {"task_ref": task.ref, "benchmark_ref": "win-rate-vs-random",
                           "score": wr_random,
                           "detail": {"games": cfg.random_eval_games}},
                      ] + ([{"task_ref": task.ref,
                             "benchmark_ref": f"{bench_with_lookahead.benchmark_ref}"
                                              f"+lookahead{cfg.eval_budget}",
                             "score": bench_with_lookahead.score,
                             "detail": bench_with_lookahead.detail}]
                           if bench_with_lookahead is not None else [])},
        )
        archive.record_epitaph(challenger_id, decision.reason)
        lineages.record(lineage.lineage_id, "policy_added", policy_id=challenger_id,
                        cycle=cycle_index, generation=generation + 1,
                        score=bench.score)

        probes = probe_policy(
            task, bare_q, trainer.net, games=cfg.probe_games, seed=cfg.seed,
            value_function=(trainer.value_function()
                            if learner_kind == "policy-value" else None),
        )
        # A policy with a broken apparatus must not become the thing every
        # later challenger is measured against, however it scored.
        decision = apply_health_gate(decision, probes)

        if decision.promote:
            archive.record_promotion(lineage.lineage_id, challenger_id, decision.reason)
            lineages.record(lineage.lineage_id, "promoted", policy_id=challenger_id,
                            cycle=cycle_index, score=deciding.score,
                            reason=decision.reason[:200])
            champion_id, champion_bench, champion_detail = (
                challenger_id, deciding.score, deciding.detail)
            champion_module = trainer.scripted()
            champion_q = _q_of(champion_module)
            generation += 1
            # Re-explore against the new champion. Only the single-output
            # learner has a random-move rate to raise; a two-headed policy
            # explores through noise in its search and by sampling its move
            # from the search's attention, so there is nothing to bump.
            if hasattr(trainer.cfg, "eps_start"):
                trainer.eps = min(trainer.cfg.eps_start, trainer.eps + 0.1)
        else:
            lineages.record(lineage.lineage_id, "rejected", policy_id=challenger_id,
                            cycle=cycle_index, score=bench.score,
                            parity=decision.parity, reason=decision.reason[:200])

        cumulative += cfg.games_per_cycle
        record = CycleRecord(
            cycle=cycle_index, lineage_id=lineage.lineage_id, task_id=lineage.task_id,
            task_evaluation_version=lineage.task_evaluation_version,
            rubric_id=cfg.rubric_id, renderer=renderer,
            experiment_id=cfg.experiment_id,
            games_trained=cfg.games_per_cycle, cumulative_games=cumulative,
            parameter_count=parameter_count,
            wall_seconds=round(time.time() - t_start, 1),
            search_budget=trainer.cfg.search_budget,
            search_states_examined=trainer.search_states_examined,
            search_deepest_ply=trainer.search_deepest,
            series=series, challenger_id=challenger_id, champion_id=champion_id,
            evaluation={"score": bench.score,
                        "benchmark_ref": bench.benchmark_ref,
                        "frontier_depth": bench.detail.get("frontier_depth"),
                        "per_rung": bench.detail.get("per_rung"),
                        "win_rate_vs_random": wr_random,
                        "eval_budget": cfg.eval_budget,
                        "score_with_lookahead":
                            bench_with_lookahead.score if bench_with_lookahead else None,
                        "frontier_depth_with_lookahead":
                            bench_with_lookahead.detail.get("frontier_depth")
                            if bench_with_lookahead else None,
                        "deciding_score": deciding.score},
            head_to_head={"score": h2h.score, "games": h2h.games,
                          "wins": h2h.wins, "losses": h2h.losses,
                          "draws": h2h.draws, "budget_each": cfg.h2h_budget},
            promoted=decision.promote, parity=decision.parity,
            gate_reason=decision.reason, probes=probes,
        )

        # Read the instruments and say what to try next.
        try:
            from .diagnosis import diagnose_card
            from .performance_card import build_card

            card = build_card_for_cycle(archive, task, challenger_id, probes)
            record.diagnosis = [s.to_dict() for s in diagnose_card(card)] if card else []
        except Exception:
            record.diagnosis = []

        # Ask whether the numbers make sense before accepting them.
        try:
            from .sanity import check_cycle, render as render_sanity, worst_severity

            findings = check_cycle(record, produced or cycles_store.load_all(
                lineage.lineage_id))
            record.sanity = [f.to_dict() for f in findings]
            if findings and cfg.verbose:
                print(render_sanity(findings))
            if worst_severity(findings) == "contradiction":
                lineages.record(lineage.lineage_id, "contradiction_found",
                                cycle=cycle_index,
                                checks=[f.check for f in findings
                                        if f.severity == "contradiction"])
        except Exception as err:
            record.sanity = [{"severity": "unexpected", "check": "sanity checking ran",
                              "expected": "the checks run", "observed": repr(err)}]

        cycles_store.save(record)
        lineages.record(lineage.lineage_id, "cycle_completed", cycle=cycle_index,
                        games=cfg.games_per_cycle, score=bench.score,
                        promoted=decision.promote)
        produced.append(record)
        if cfg.verbose:
            print(f"[{lineage.lineage_id}] {record.summary_line()}  "
                  f"vs random {wr_random:.0%}  ({record.wall_seconds / 60:.1f} min)")

    return produced


def build_card_for_cycle(archive, task, policy_id, probes):
    """A minimal card over already-measured probe numbers, so the diagnosis
    rules can run without re-probing the policy."""
    from .performance_card import (
        Bands, FAIL, INFO, OK, PerformanceCard, Reading, WARN, _band_high, _band_low,
    )

    readings: List[Reading] = []

    def add(name, value, status, expectation):
        readings.append(Reading(name=name, value=value, status=status,
                                expectation=expectation))

    if "q_max_abs" in probes:
        add("q_max_abs", probes["q_max_abs"],
            _band_low(probes["q_max_abs"], Bands.Q_MAX_ABS_OK, Bands.Q_MAX_ABS_WARN),
            "within the reward scale")
    if "q_mean_abs" in probes:
        add("q_mean_abs", probes["q_mean_abs"],
            _band_low(probes["q_mean_abs"], Bands.Q_MEAN_ABS_OK, Bands.Q_MEAN_ABS_WARN),
            "within [0, 1]")
    if "q_of_game_winning_moves" in probes:
        add("q_of_game_winning_moves", probes["q_of_game_winning_moves"],
            _band_high(probes["q_of_game_winning_moves"], Bands.WIN_MOVE_Q_OK,
                       Bands.WIN_MOVE_Q_WARN), "-> +1")
    if "take_the_win_rate" in probes:
        add("take_the_win_rate", probes["take_the_win_rate"],
            _band_high(probes["take_the_win_rate"], Bands.TAKE_WIN_OK,
                       Bands.TAKE_WIN_WARN), "-> 1.0")
    if "block_the_loss_rate" in probes:
        add("block_the_loss_rate", probes["block_the_loss_rate"],
            _band_high(probes["block_the_loss_rate"], Bands.BLOCK_OK,
                       Bands.BLOCK_WARN), "-> 1.0")
    if "q_of_not_blocking" in probes:
        add("q_of_not_blocking", probes["q_of_not_blocking"],
            _band_low(probes["q_of_not_blocking"], Bands.NON_BLOCK_Q_OK,
                      Bands.NON_BLOCK_Q_WARN), "-> -1")
    for key in ("q_at_solver_proven_wins", "q_at_solver_proven_losses"):
        if key in probes:
            value = probes[key]
            status = (_band_high(value, Bands.WIN_MOVE_Q_OK, Bands.WIN_MOVE_Q_WARN)
                      if "wins" in key
                      else (OK if value <= -Bands.WIN_MOVE_Q_OK
                            else (WARN if value <= 0 else FAIL)))
            add(key, value, status, "-> +1" if "wins" in key else "-> -1")
    for layer, stats in (probes.get("weight_distribution") or {}).items():
        add(f"dead_weight_fraction::{layer}", stats.get("dead_fraction"),
            _band_low(stats.get("dead_fraction", 0.0), Bands.DEAD_FRAC_OK,
                      Bands.DEAD_FRAC_WARN), "roughly constant over training")

    return PerformanceCard(policy_id=policy_id, task_ref=task.ref,
                           generated_at="", probe_config={},
                           sections={"cycle": readings})
