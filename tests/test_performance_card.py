"""Performance-card tests: probes must flag known-broken policies, pass
known-good fixtures, and the assembled card must be serializable and
generatable straight from the archive."""

import json

import numpy as np
import pytest

from harness.agents.dqn import DQNTrainer, TrainerConfig
from harness.performance_card import (
    FAIL,
    OK,
    WARN,
    Bands,
    build_card,
    collect_rollouts,
    network_health_readings,
    save_card,
    tactical_readings,
    telemetry_readings,
    value_calibration_readings,
)
from harness.tasks.connect4 import Connect4Env, Connect4Task


def _reading(readings, name):
    match = [r for r in readings if r.name == name]
    assert match, f"no reading named {name} in {[r.name for r in readings]}"
    return match[0]


# ---------------------------------------------------------------------------
# rollout probes
# ---------------------------------------------------------------------------


def test_exploding_q_values_flagged_fail():
    """A Q-function far outside the ±1 reward scale must FAIL calibration."""
    task = Connect4Task(benchmark_games=4)

    def q(obs):
        return np.full(7, 50.0)

    h = collect_rollouts(task, q, games=6, seed=0)
    readings = value_calibration_readings(h)
    assert _reading(readings, "q_max_abs").status == FAIL
    assert _reading(readings, "q_mean_abs").status == FAIL


def test_nan_q_values_flagged_fail():
    task = Connect4Task(benchmark_games=4)

    def q(obs):
        out = np.zeros(7)
        out[3] = np.nan
        return out

    h = collect_rollouts(task, q, games=4, seed=0)
    readings = value_calibration_readings(h)
    assert _reading(readings, "q_nonfinite_count").status == FAIL


def test_tactical_oracle_scores_ok():
    """A Q-function that puts +1 on winning/blocking columns and small noise
    elsewhere should pass every tactical probe."""
    task = Connect4Task(benchmark_games=4)
    probe_env = Connect4Env()
    rng = np.random.default_rng(1)

    def q(obs):
        probe_env.board = (obs[0] - obs[1]).astype(np.int8)
        probe_env.current_player = 1
        out = rng.uniform(-0.05, 0.05, size=7)
        for c in probe_env.winning_moves(-1):
            out[c] = 0.9
        for c in probe_env.winning_moves(1):
            out[c] = 1.0
        return out

    h = collect_rollouts(task, q, games=40, seed=0)
    assert h.tactical_available
    assert h.offense_n > 0, "40 games vs random must surface winnable positions"
    readings = tactical_readings(h)
    assert _reading(readings, "take_the_win_rate").status == OK
    assert _reading(readings, "q_of_winning_move_available").status == OK
    win_q = _reading(value_calibration_readings(h), "q_of_game_winning_moves")
    assert win_q.status == OK, "oracle's winning moves carry Q≈1"


def test_tactically_blind_policy_flagged():
    """A constant Q-function never prefers the winning column — the tactical
    probes are what catch this even when win/loss vs weak opponents looks ok."""
    task = Connect4Task(benchmark_games=4)

    def q(obs):
        return np.linspace(0.1, 0.0, 7)  # always prefers column 0

    h = collect_rollouts(task, q, games=40, seed=0)
    if h.offense_n:  # column-0 wins can coincidentally be "taken"
        r = _reading(tactical_readings(h), "take_the_win_rate")
        assert r.status in (WARN, FAIL)


def test_unique_states_and_score_reported():
    task = Connect4Task(benchmark_games=4)

    def q(obs):
        return np.zeros(7)

    h = collect_rollouts(task, q, games=10, seed=0)
    assert h.unique_states > 10
    assert 0.0 <= h.policy_score <= 1.0


# ---------------------------------------------------------------------------
# network health
# ---------------------------------------------------------------------------


def test_network_health_on_fresh_net_ok():
    trainer = DQNTrainer(6, 7, 7, TrainerConfig(seed=0))
    readings = network_health_readings(trainer.net)
    assert _reading(readings, "nonfinite_weights").status == OK
    assert _reading(readings, "exploding_weight_fraction").status == OK


def test_network_health_flags_nan_and_explosion():
    trainer = DQNTrainer(6, 7, 7, TrainerConfig(seed=0))
    with __import__("torch").no_grad():
        trainer.net.fc1.weight[0, 0] = float("nan")
        trainer.net.fc1.weight[1, :] = 100.0
    readings = network_health_readings(trainer.net)
    assert _reading(readings, "nonfinite_weights").status == FAIL
    assert _reading(readings, "exploding_weight_fraction").status in (WARN, FAIL)


# ---------------------------------------------------------------------------
# telemetry reads
# ---------------------------------------------------------------------------


def test_telemetry_trends():
    telemetry = {
        "episode_loss_means": [1.0, 0.8, 0.5, 0.3, 0.2],
        "iterations": [
            {"benchmark_score": 0.5, "dormant_neurons": {"fc1.weight": 0.10}},
            {"benchmark_score": 0.7, "dormant_neurons": {"fc1.weight": 0.30}},
        ],
    }
    readings = telemetry_readings(telemetry)
    assert _reading(readings, "training_loss_trend").status == OK
    assert _reading(readings, "benchmark_trajectory").status == OK
    assert _reading(readings, "dormant_neuron_trend").status == FAIL  # 0.10 -> 0.30 rising


def test_telemetry_missing_is_info_not_crash():
    readings = telemetry_readings({})
    assert readings[0].status == "info"


# ---------------------------------------------------------------------------
# end-to-end: card from the archive
# ---------------------------------------------------------------------------


def test_card_builds_from_archive_and_serializes(tmp_path):
    from harness.improvement_loop import Store

    store = Store(tmp_path / "store")
    task = Connect4Task(benchmark_games=10)
    trainer = DQNTrainer(6, 7, 7, TrainerConfig(seed=0))
    policy_id = store.archive.next_policy_id("connect4")
    store.archive.save_policy(
        policy_id,
        trainer.scripted(),
        metadata={
            "task_ref": task.ref,
            "experiment_id": "exp-card-test",
            "parent_id": None,
            "generation": 0,
            "training_config": {},
            "eval_records": [
                {"task_ref": task.ref, "benchmark_ref": task.benchmark.ref, "score": 0.5}
            ],
        },
    )

    card = build_card(store, policy_id, task=task, games=8, seed=0)
    assert card.policy_id == policy_id
    for section in ("identity", "benchmark", "value_calibration", "tactics",
                    "value_propagation", "exploration", "network_health",
                    "training_telemetry"):
        assert section in card.sections, f"missing section {section}"

    # JSON round-trip and text render both work
    blob = json.dumps(card.to_dict())
    assert card.worst_status() in ("ok", "warn", "fail", "info")
    assert policy_id in card.render()

    paths = save_card(store, card)
    saved = json.loads((tmp_path / "store" / "cards" / f"{policy_id}.json").read_text())
    assert saved["policy_id"] == policy_id
    assert "value_calibration" in saved["sections"]
    assert paths["txt"].endswith(".txt")


def test_card_unknown_policy_raises(tmp_path):
    from harness.improvement_loop import Store

    store = Store(tmp_path / "store")
    with pytest.raises(ValueError, match="unknown policy"):
        build_card(store, "connect4-pol-99999")


# ---------------------------------------------------------------------------
# solver-based Q calibration (ground truth)
# ---------------------------------------------------------------------------


def test_solver_calibration_rewards_calibrated_oracle():
    """A Q-function that is +1-ish whenever a forced win exists and -1-ish in
    lost positions should pass; probe must find provable positions."""
    from harness.performance_card import solver_calibration_readings
    from harness.tasks.connect4_solver import Connect4Solver, HEURISTIC_CAP

    task = Connect4Task(benchmark_games=4)
    probe_env = Connect4Env()
    solver = Connect4Solver(max_depth=4)

    def q(obs):
        probe_env.board = (obs[0] - obs[1]).astype(np.int8)
        probe_env.current_player = 1
        evals = solver.evaluate_moves(probe_env.board, 1)
        out = np.zeros(7)
        for e in evals:
            if e.exact and e.score > HEURISTIC_CAP:
                out[e.col] = 0.95
            elif e.exact and e.score < -HEURISTIC_CAP:
                out[e.col] = -0.95
        return out

    readings = solver_calibration_readings(task, q, games=8, seed=0, solver_depth=4,
                                           max_positions=40)
    names = {r.name for r in readings}
    assert "q_at_solver_proven_wins" in names, f"probe found no provable wins: {names}"
    win_reading = [r for r in readings if r.name == "q_at_solver_proven_wins"][0]
    assert win_reading.status == OK


def test_solver_calibration_flags_flat_q():
    """An untrained (all-zeros) Q-function cannot know it is winning or beaten
    — proven-win positions must not read OK."""
    from harness.performance_card import solver_calibration_readings

    task = Connect4Task(benchmark_games=4)
    readings = solver_calibration_readings(
        task, lambda obs: np.zeros(7), games=8, seed=0, solver_depth=4, max_positions=40
    )
    win = [r for r in readings if r.name == "q_at_solver_proven_wins"]
    if win:  # provable positions found
        assert win[0].status in (WARN, FAIL)


def test_trainer_reports_q_abs_means():
    from harness.scripted_opponents import RandomOpponent

    trainer = DQNTrainer(6, 7, 7, TrainerConfig(seed=0, batch_size=8))
    opp = RandomOpponent(seed=1)
    vals = []
    for _ in range(6):
        stats = trainer.play_episode(Connect4Env(), opponent_q=None, opponent_act=opp.act)
        vals.extend(stats.q_abs_means)
    assert vals and all(np.isfinite(v) and v >= 0 for v in vals)
