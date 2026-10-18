"""ADR-005 gate v2 (either-path promotion), plateau detection, and the
tactical replay-bias lever."""

import json

import pytest

from harness.agents.dqn import DQNTrainer, ReplayBuffer, TrainerConfig
from harness.improvement_loop import LoopConfig, Store, run_loop
from harness.promotion_gate import statistical_either_gate
from harness.scripted_opponents import OneStepWinOpponent
from harness.stats import plateaued
from harness.tasks.connect4 import Connect4Env, Connect4Task


# ---------------------------------------------------------------------------
# gate v2
# ---------------------------------------------------------------------------


def gate(h2h, bench, champ_bench, h2h_n=200, bench_n=168):
    return statistical_either_gate(
        challenger_h2h_score=h2h, h2h_games=h2h_n,
        challenger_benchmark=bench, challenger_benchmark_games=bench_n,
        champion_benchmark=champ_bench, champion_benchmark_games=bench_n,
    )


def test_relative_path_still_promotes():
    d = gate(h2h=0.60, bench=0.08, champ_bench=0.08)
    assert d.promote and "relative path" in d.reason


def test_absolute_path_promotes_at_h2h_parity():
    """The new case the human directed: benchmark significantly better,
    head-to-head at parity — promote."""
    d = gate(h2h=0.50, bench=0.30, champ_bench=0.08)
    assert d.promote and "absolute path" in d.reason


def test_absolute_superiority_blocked_by_proven_h2h_regression():
    """exp-012 iter-11 shape: bench better but h2h PROVEN below parity
    (0.41 over 200 games, upper bound < 0.5) — still rejected."""
    d = gate(h2h=0.41, bench=0.30, champ_bench=0.08)
    assert not d.promote and "h2h regression" in d.reason


def test_h2h_win_blocked_by_benchmark_regression_v32_trap():
    d = gate(h2h=0.65, bench=0.01, champ_bench=0.30)
    assert not d.promote and "v32" in d.reason


def test_neither_significant_rejects_as_not_proven():
    d = gate(h2h=0.52, bench=0.116, champ_bench=0.083)  # exp-012's actual margins
    assert not d.promote and "not proven" in d.reason


# ---------------------------------------------------------------------------
# plateau detection
# ---------------------------------------------------------------------------


def test_plateaued_helper():
    rising = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert not plateaued(rising, patience=3, min_delta=0.02)
    flat = [0.1, 0.30, 0.31, 0.30, 0.29, 0.31]
    assert plateaued(flat, patience=3, min_delta=0.02)
    # needs more history than patience
    assert not plateaued([0.1, 0.1], patience=3, min_delta=0.02)
    # a late breakout resets the verdict
    breakout = [0.1, 0.3, 0.3, 0.3, 0.45]
    assert not plateaued(breakout, patience=3, min_delta=0.02)


def test_loop_stops_early_on_plateau(tmp_path):
    store = Store(tmp_path / "store")
    cfg = LoopConfig(
        experiment_id="exp-plateau-stop",
        hypothesis="if plateaued, the loop stops early and records why",
        changes="plateau_action=stop with an unreachable min_delta",
        prior_art="none",
        seed=0,
        iterations=6,
        episodes_per_iteration=4,
        gate_games=4,
        plateau_patience=2,
        plateau_min_delta=1.0,  # nothing can improve by >1.0 — plateaus by iter 2
        plateau_min_iterations=0,  # no warm-up in this unit test
        plateau_action="stop",
        trainer=TrainerConfig(seed=0, batch_size=8),
    )
    run_loop(Connect4Task(benchmark_games=4), store, cfg)
    telemetry = json.loads((store.telemetry_dir / "exp-plateau-stop.json").read_text())
    assert "plateau_stop" in telemetry
    assert telemetry["plateau_stop"]["iterations_saved"] > 0
    assert len(telemetry["iterations"]) < 6
    assert telemetry["iterations"][-1].get("plateau") is True


def test_loop_flag_mode_does_not_stop(tmp_path):
    store = Store(tmp_path / "store")
    cfg = LoopConfig(
        experiment_id="exp-plateau-flag",
        hypothesis="if plateaued with action=flag, the loop finishes anyway",
        changes="plateau_action=flag",
        prior_art="none",
        seed=0,
        iterations=4,
        episodes_per_iteration=4,
        gate_games=4,
        plateau_patience=2,
        plateau_min_delta=1.0,
        plateau_min_iterations=0,
        plateau_action="flag",
        trainer=TrainerConfig(seed=0, batch_size=8),
    )
    run_loop(Connect4Task(benchmark_games=4), store, cfg)
    telemetry = json.loads((store.telemetry_dir / "exp-plateau-flag.json").read_text())
    assert "plateau_stop" not in telemetry
    assert len(telemetry["iterations"]) == 4
    assert any(it.get("plateau") for it in telemetry["iterations"])


# ---------------------------------------------------------------------------
# tactical replay bias
# ---------------------------------------------------------------------------


def _fake_transition(done=False):
    import numpy as np

    obs = np.zeros((2, 6, 7), dtype=np.float32)
    return (obs, 0, 0.0, obs, np.ones(7, dtype=np.float32), done)


def test_tactical_quota_in_sampled_batches():
    buf = ReplayBuffer(capacity=1000, seed=0, tactical_fraction=0.5)
    tactical = _fake_transition()
    plain = _fake_transition()
    for _ in range(20):
        buf.add(tactical, tactical=True)
    for _ in range(200):
        buf.add(plain, tactical=False)
    batch = buf.sample(20)
    assert len(batch) == 20
    assert len(buf.tacticals) == 20  # tracked separately
    # quota: 50% of the batch must come from the tactical deque
    assert sum(1 for t in batch if t is tactical) >= 10


def test_trainer_marks_tactical_transitions():
    """Playing vs a one-step-win opponent generates must-block positions;
    with tactical_fraction on, they land in the tactical deque even though
    reward shaping is OFF (flags and shaping are independent levers)."""
    trainer = DQNTrainer(6, 7, 7, TrainerConfig(seed=0, batch_size=8, tactical_fraction=0.3))
    opp = OneStepWinOpponent(seed=2)
    for _ in range(12):
        trainer.play_episode(Connect4Env(), opponent_q=None, opponent_act=opp.act)
    assert len(trainer.buffer.tacticals) > 0
    # shaping off => rewards stay pure win/lose (no -0.5s injected)
    rewards = {t[2] for t in trainer.buffer.buffer}
    assert rewards <= {-1.0, 0.0, 1.0}


def test_plateau_warmup_protects_slow_starters(tmp_path):
    """Regression for the 2026-10-03 defect: a cold-start curve that is flat
    early must NOT be stopped before plateau_min_iterations."""
    store = Store(tmp_path / "store")
    cfg = LoopConfig(
        experiment_id="exp-plateau-warmup",
        hypothesis="if the run is flat early, warm-up prevents a premature stop",
        changes="plateau_min_iterations > iterations",
        prior_art="none",
        seed=0,
        iterations=4,
        episodes_per_iteration=4,
        gate_games=4,
        plateau_patience=1,
        plateau_min_delta=1.0,   # would plateau immediately without warm-up
        plateau_min_iterations=10,
        plateau_action="stop",
        trainer=TrainerConfig(seed=0, batch_size=8),
    )
    run_loop(Connect4Task(benchmark_games=4), store, cfg)
    telemetry = json.loads((store.telemetry_dir / "exp-plateau-warmup.json").read_text())
    assert "plateau_stop" not in telemetry
    assert len(telemetry["iterations"]) == 4
