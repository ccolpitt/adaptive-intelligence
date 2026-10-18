"""ADR-007: strict promotion gate (win BOTH axes), parity signal, and the
card-reading diagnosis that turns instrument failures into hypotheses."""

import json

from harness.diagnosis import diagnose_card, diagnose_run, render_diagnosis
from harness.performance_card import FAIL, OK, WARN, PerformanceCard, Reading
from harness.promotion_gate import GATE_RULES, statistical_strict_gate


def gate(h2h, bench, champ, n_h2h=200, n_bench=168):
    return statistical_strict_gate(h2h, n_h2h, bench, n_bench, champ, n_bench)


def test_strict_promotes_only_when_both_axes_win():
    assert gate(0.60, 0.090, 0.083).promote
    d = gate(0.60, 0.083, 0.083)  # beats champion, no bench improvement
    assert not d.promote and "does NOT improve" in d.reason
    d = gate(0.50, 0.200, 0.083)  # bench better, h2h parity
    assert not d.promote


def test_strict_flags_parity_for_hypothesis_generator():
    d = gate(0.50, 0.090, 0.083)
    assert d.parity and not d.promote and "PARITY" in d.reason
    d = gate(0.30, 0.090, 0.083)  # proven loss is not parity
    assert not d.parity
    d = gate(0.50, 0.040, 0.083)  # parity h2h but bench below champion
    assert not d.parity


def test_gate_registry_has_all_rules():
    assert set(GATE_RULES) == {"double", "either", "strict"}


def _card(readings):
    return PerformanceCard(policy_id="p", task_ref="connect4@v2", generated_at="t",
                           probe_config={}, sections={"x": readings})


def test_diagnosis_maps_failures_to_levers():
    card = _card([
        Reading(name="q_max_abs", status=FAIL, expectation="", value=3.2),
        Reading(name="take_the_win_rate", status=FAIL, expectation="", value=0.44),
        Reading(name="block_the_loss_rate", status=WARN, expectation="", value=0.70),
        Reading(name="value_horizon_plies", status="info", expectation="", value=0.0),
    ])
    suggestions = diagnose_card(card)
    text = render_diagnosis(suggestions)
    assert "q_max_abs" in text and "target_update_every" in text
    assert "take_the_win" in text and "tactical_fraction" in text
    assert "value_horizon" in text and "n-step" in text


def test_clean_card_yields_no_suggestions():
    card = _card([Reading(name="q_max_abs", status=OK, expectation="", value=0.9),
                  Reading(name="take_the_win_rate", status=OK, expectation="", value=0.99)])
    assert diagnose_card(card) == []
    assert "clean" in render_diagnosis([])


def test_parity_run_signal_comes_first():
    card = _card([])
    telemetry = {"iterations": [
        {"iteration": 0, "promoted": False, "parity": False, "benchmark_score": 0.02},
        {"iteration": 1, "promoted": False, "parity": True, "benchmark_score": 0.08},
    ]}
    s = diagnose_run(card, telemetry)
    assert s and s[0].severity == "signal" and "PARITY" in s[0].finding
    assert any("2x the training budget" in h for h in s[0].hypotheses)


def test_loop_writes_diagnosis_into_telemetry(tmp_path):
    from harness.agents.dqn import TrainerConfig
    from harness.improvement_loop import LoopConfig, Store, run_loop
    from harness.tasks.connect4 import Connect4Task

    store = Store(tmp_path / "store")
    cfg = LoopConfig(
        experiment_id="exp-diag", hypothesis="if run, a diagnosis is written",
        changes="none", prior_art="none", seed=0, iterations=2,
        episodes_per_iteration=4, gate_games=4, plateau_action="off",
        auto_card=True, auto_card_games=10,
        trainer=TrainerConfig(seed=0, batch_size=8),
    )
    run_loop(Connect4Task(benchmark_games=4), store, cfg)
    t = json.loads((store.telemetry_dir / "exp-diag.json").read_text())
    assert "diagnosis" in t and "suggestions" in t["diagnosis"]
    assert all("parity" in it for it in t["iterations"])
