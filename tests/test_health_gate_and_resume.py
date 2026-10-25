"""Health conditions on promotion, the corrected solver-calibration probe,
and resuming an interrupted run."""

import numpy as np
import pytest

from harness.promotion_gate import (
    GateDecision,
    apply_health_gate,
    check_health,
    statistical_strict_gate,
)


def healthy_probes(**overrides):
    probes = {
        "q_max_abs": 1.2,
        "take_the_win_rate": 0.95,
        "block_the_loss_rate": 0.92,
        "weight_distribution": {
            "conv.0.weight": {"dead_fraction": 0.05, "exploding_fraction": 0.0,
                              "nonfinite": 0},
            "fc1.weight": {"dead_fraction": 0.22, "exploding_fraction": 0.0,
                           "nonfinite": 0},
        },
    }
    probes.update(overrides)
    return probes


def promoting_decision():
    return statistical_strict_gate(0.60, 200, 0.12, 168, 0.08, 168)


# ---------------------------------------------------------------------------
# what disqualifies a policy
# ---------------------------------------------------------------------------


def test_a_sound_policy_passes():
    verdict = check_health(healthy_probes())
    assert verdict.healthy and not verdict.blocking


def test_non_finite_weights_disqualify():
    probes = healthy_probes()
    probes["weight_distribution"]["fc1.weight"]["nonfinite"] = 3
    verdict = check_health(probes)
    assert not verdict.healthy and "non-finite" in verdict.reason


def test_diverging_value_estimates_disqualify():
    verdict = check_health(healthy_probes(q_max_abs=4.5))
    assert not verdict.healthy and "diverging" in verdict.reason


def test_a_collapsed_layer_disqualifies():
    probes = healthy_probes()
    probes["weight_distribution"]["fc1.weight"]["dead_fraction"] = 0.95
    verdict = check_health(probes)
    assert not verdict.healthy and "silent" in verdict.reason


def test_mild_drift_and_weak_tactics_only_warn():
    """Improving from terrible to poor is still improving, so weak tactics
    must never block a promotion — otherwise a lineage can never start."""
    verdict = check_health(healthy_probes(q_max_abs=1.9, take_the_win_rate=0.40,
                                          block_the_loss_rate=0.30))
    assert verdict.healthy
    assert any("takes an available win" in w for w in verdict.warnings)
    assert any("above the reward range" in w for w in verdict.warnings)


# ---------------------------------------------------------------------------
# how it combines with the score gates
# ---------------------------------------------------------------------------


def test_a_fault_withdraws_a_qualified_promotion():
    decision = promoting_decision()
    assert decision.promote
    gated = apply_health_gate(decision, healthy_probes(q_max_abs=9.0))
    assert not gated.promote
    assert "withheld" in gated.reason and "Scores qualified" in gated.reason


def test_warnings_are_recorded_without_blocking():
    gated = apply_health_gate(promoting_decision(),
                              healthy_probes(take_the_win_rate=0.5))
    assert gated.promote and "health warnings" in gated.reason


def test_health_never_overturns_a_rejection():
    rejection = statistical_strict_gate(0.40, 200, 0.02, 168, 0.08, 168)
    assert not rejection.promote
    gated = apply_health_gate(rejection, healthy_probes())
    assert not gated.promote
    assert gated.reason == rejection.reason


def test_missing_probes_do_not_block():
    """A cycle that could not take its measurements must not be punished."""
    gated = apply_health_gate(promoting_decision(), {})
    assert gated.promote


# ---------------------------------------------------------------------------
# the corrected probe: a maximum over noisy estimates reads high
# ---------------------------------------------------------------------------


def test_best_of_noise_is_positive_by_construction():
    """The reason cycle 0 appeared to value a lost position positively."""
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 0.3, (50_000, 7))
    assert abs(samples.mean()) < 0.01, "each estimate is centred on zero"
    assert samples.max(axis=1).mean() > 0.3, "yet the best of seven is clearly positive"


def test_probe_reports_mean_and_flags_the_artefact():
    from harness.performance_card import INFO, solver_calibration_readings
    from harness.tasks.connect4 import Connect4Task

    rng = np.random.default_rng(1)
    noisy = lambda obs: rng.normal(0, 0.3, 7)  # knows nothing, centred on zero
    readings = {r.name: r for r in solver_calibration_readings(
        Connect4Task(benchmark_games=4), noisy, games=10, seed=0,
        solver_depth=4, max_positions=60)}

    if "q_at_solver_proven_losses" in readings:
        best = readings["q_at_solver_proven_losses"]
        # An untrained network must not be reported as a sign error.
        if best.value is not None and best.value > 0:
            assert best.status == INFO
            assert "noisy estimates" in best.note
            assert best.detail["max_of_noise_artefact"] > 0
        # The mean-based reading is the honest measure and is also present.
        assert "q_at_solver_proven_losses_mean" in readings


# ---------------------------------------------------------------------------
# resuming
# ---------------------------------------------------------------------------


def test_an_interrupted_run_continues_where_it_stopped(tmp_path):
    from harness.cycle_loop import CycleLoopConfig, run_cycles
    from harness.cycle_record import CycleStore
    from harness.lineage import (
        Lineage, LineageStore, RubricStore, rubric_r002_budgeted_lookahead,
    )
    from harness.tasks.connect4 import Connect4Task

    RubricStore(tmp_path / "rubrics").publish(rubric_r002_budgeted_lookahead(4))
    LineageStore(tmp_path / "lineages").create(Lineage(
        lineage_id="resume-L", task_id="connect4",
        rubric_id="r002-lookahead-4-puct", description="resume test"))
    task = Connect4Task(benchmark_games=4)
    base = dict(lineage_id="resume-L", rubric_id="r002-lookahead-4-puct",
                games_per_cycle=6, gate_games=4, random_eval_games=4,
                probe_games=2, seed=3, renderer="connect4_v2", verbose=False)

    first = run_cycles(task, tmp_path, CycleLoopConfig(cycles=2, **base))
    assert [r.cycle for r in first] == [0, 1]

    # Interrupted here. A second call must continue, not start over.
    second = run_cycles(task, tmp_path, CycleLoopConfig(cycles=2, **base))
    assert [r.cycle for r in second] == [2, 3]
    assert second[0].cumulative_games == 18  # 6 games x 3 cycles
    assert [r.cycle for r in CycleStore(tmp_path / "cycles").load_all("resume-L")] == \
        [0, 1, 2, 3]
    assert LineageStore(tmp_path / "lineages").events("resume-L", "resumed")


def test_evaluation_lookahead_is_recorded_alongside_the_plain_score(tmp_path):
    from harness.cycle_loop import CycleLoopConfig, run_cycles
    from harness.lineage import (
        Lineage, LineageStore, RubricStore, rubric_r002_budgeted_lookahead,
    )
    from harness.tasks.connect4 import Connect4Task

    RubricStore(tmp_path / "rubrics").publish(rubric_r002_budgeted_lookahead(4))
    LineageStore(tmp_path / "lineages").create(Lineage(
        lineage_id="evalbudget-L", task_id="connect4",
        rubric_id="r002-lookahead-4-puct", description="eval budget test"))

    records = run_cycles(Connect4Task(benchmark_games=4), tmp_path, CycleLoopConfig(
        lineage_id="evalbudget-L", rubric_id="r002-lookahead-4-puct",
        games_per_cycle=6, cycles=1, gate_games=4, random_eval_games=4,
        probe_games=2, seed=5, renderer="connect4_v2", verbose=False,
        eval_budget=16, eval_budget_decides=False))

    evaluation = records[0].evaluation
    # Both are measured, so choosing which one counts never loses a comparison.
    assert evaluation["score"] is not None
    assert evaluation["score_with_lookahead"] is not None
    assert evaluation["eval_budget"] == 16
    assert evaluation["deciding_score"] == evaluation["score"]
