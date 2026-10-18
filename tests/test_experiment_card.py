"""Gradient-norm telemetry and experiment-card rendering."""

import json

import numpy as np
import pytest

from harness.agents.dqn import DQNTrainer, TrainerConfig
from harness.experiment_card import gather, render_html, render_text, save_all, save_experiment_card, svg_chart
from harness.improvement_loop import Store
from harness.scripted_opponents import RandomOpponent
from harness.tasks.connect4 import Connect4Env


# ---------------------------------------------------------------------------
# gradient norms
# ---------------------------------------------------------------------------


def test_trainer_records_gradient_norms():
    trainer = DQNTrainer(6, 7, 7, TrainerConfig(seed=0, batch_size=8))
    opp = RandomOpponent(seed=1)
    norms = []
    for _ in range(6):
        stats = trainer.play_episode(Connect4Env(), opponent_q=None, opponent_act=opp.act)
        norms.extend(stats.grad_norms)
        assert len(stats.grad_norms) == len(stats.losses)
    assert norms, "gradient norms recorded once the buffer is warm"
    assert all(np.isfinite(n) and n >= 0 for n in norms)


# ---------------------------------------------------------------------------
# experiment card
# ---------------------------------------------------------------------------


@pytest.fixture
def store_with_experiment(tmp_path):
    store = Store(tmp_path / "store")
    store.experiments.register(
        experiment_id="exp-test-card",
        hypothesis="if we add X, benchmark rises above 0.2",
        changes="X enabled",
        conditions={"task_ref": "connect4@v2", "benchmark_ref": "connect4-benchmark@v1",
                    "config": {"lr": 0.001}, "seed": 3},
        prior_art="none",
        input_policies=["connect4-pol-00001"],
    )
    store.experiments.add_output_policy("exp-test-card", "connect4-pol-00002")
    telemetry = {
        "experiment_id": "exp-test-card",
        "task_ref": "connect4@v2",
        "episode_loss_means": [0.5, 0.3, 0.2],
        "episode_grad_norm_means": [1.2, 0.9, 0.8],
        "iterations": [
            {"iteration": 0, "challenger": "connect4-pol-00002", "h2h_score_vs_champion": 0.62,
             "h2h_games": 100, "benchmark_score": 0.25, "champion_benchmark": 0.10,
             "promoted": True, "reason": "promoted: h2h LB > 0.5", "epsilon": 0.4,
             "dormant_neurons": {"fc1.weight": 0.1, "out.weight": 0.02}},
            {"iteration": 1, "challenger": "connect4-pol-00003", "h2h_score_vs_champion": 0.45,
             "h2h_games": 100, "benchmark_score": 0.22, "champion_benchmark": 0.25,
             "promoted": False, "reason": "rejected", "epsilon": 0.3,
             "dormant_neurons": {"fc1.weight": 0.12, "out.weight": 0.02}},
        ],
    }
    (store.telemetry_dir / "exp-test-card.json").write_text(json.dumps(telemetry))
    return store


def test_gather_joins_registry_and_telemetry(store_with_experiment):
    data = gather(store_with_experiment, "exp-test-card")
    assert "if we add X" in data["hypothesis"]
    assert data["inputs"] == ["connect4-pol-00001"]
    assert data["promoted"] == ["connect4-pol-00002"]
    assert data["verdict"] is None  # pending
    assert len(data["telemetry"]["iterations"]) == 2


def test_html_card_contains_hypothesis_setup_and_charts(store_with_experiment):
    data = gather(store_with_experiment, "exp-test-card")
    page = render_html(data)
    assert "if we add X, benchmark rises above 0.2" in page
    assert "connect4-benchmark@v1" in page       # auditable conditions
    assert "connect4-pol-00001" in page          # input policy
    assert page.count("<svg") >= 4               # bench, h2h, loss, grad-norm, dormant
    assert "gradient norm" in page
    assert "pending" in page                      # verdict badge


def test_verdict_shows_on_card(store_with_experiment):
    store_with_experiment.experiments.record_verdict(
        "exp-test-card", "supported", "X worked", {"final_bench": 0.25}
    )
    data = gather(store_with_experiment, "exp-test-card")
    page = render_html(data)
    assert "supported" in page and "X worked" in page
    txt = render_text(data)
    assert "SUPPORTED" in txt and "★" in txt


def test_save_and_index(store_with_experiment, tmp_path):
    out = save_experiment_card(store_with_experiment, "exp-test-card")
    assert (tmp_path / "store" / "cards" / "experiments" / "exp-test-card.html").exists()
    assert "EXPERIMENT CARD" in out["text"]
    index = save_all(store_with_experiment)
    assert "index.html" in index
    assert "exp-test-card" in (tmp_path / "store" / "cards" / "experiments" / "index.html").read_text()


def test_gather_unknown_experiment_raises(tmp_path):
    store = Store(tmp_path / "store")
    with pytest.raises(ValueError, match="unknown experiment"):
        gather(store, "exp-nope")


def test_svg_chart_handles_empty_and_none():
    assert "no data" in svg_chart({"a": []}, "empty")
    page = svg_chart({"a": [1.0, None, 2.0]}, "gaps", hline=1.5)
    assert "<polyline" in page


# ---------------------------------------------------------------------------
# per-episode telemetry + matplotlib dashboard
# ---------------------------------------------------------------------------


def test_loop_records_per_episode_series(tmp_path):
    """A real tiny run must produce flat per-episode curves: loss, grad norm,
    result, cumulative unique states, epsilon — same length, one per episode."""
    from harness.improvement_loop import LoopConfig, run_loop
    from harness.tasks.connect4 import Connect4Task

    store = Store(tmp_path / "store")
    cfg = LoopConfig(
        experiment_id="exp-episodes",
        hypothesis="if we run, per-episode telemetry appears",
        changes="none",
        prior_art="none",
        seed=0,
        iterations=2,
        episodes_per_iteration=8,
        gate_games=4,
        trainer=TrainerConfig(seed=0, batch_size=8),
    )
    run_loop(Connect4Task(benchmark_games=4), store, cfg)

    telemetry = json.loads((store.telemetry_dir / "exp-episodes.json").read_text())
    ep = telemetry["episodes"]
    n = 2 * 8
    for key in ("loss", "grad_norm", "q_mean_abs", "result", "unique_states", "epsilon"):
        assert len(ep[key]) == n, f"{key}: expected {n} entries, got {len(ep[key])}"
    assert all(r in (-1, 0, 1) for r in ep["result"])
    # cumulative unique states never decreases and actually grows
    assert ep["unique_states"] == sorted(ep["unique_states"])
    assert ep["unique_states"][-1] > 0
    # once the buffer is warm, losses and grad norms are real numbers
    warm = [x for x in ep["loss"] if x is not None]
    assert warm and all(np.isfinite(x) for x in warm)


def test_matplotlib_dashboard_renders(store_with_experiment, tmp_path):
    pytest.importorskip("matplotlib")
    from harness.experiment_card import save_matplotlib_dashboard

    # augment fixture telemetry with per-episode series
    tpath = store_with_experiment.telemetry_dir / "exp-test-card.json"
    telemetry = json.loads(tpath.read_text())
    telemetry["episodes"] = {
        "loss": [None, 0.5, 0.4, 0.3],
        "grad_norm": [None, 1.0, 0.9, 0.8],
        "result": [-1, 1, 0, 1],
        "unique_states": [5, 12, 18, 22],
        "epsilon": [0.5, 0.45, 0.4, 0.35],
    }
    tpath.write_text(json.dumps(telemetry))

    png = save_matplotlib_dashboard(store_with_experiment, "exp-test-card")
    out = tmp_path / "store" / "cards" / "experiments" / "exp-test-card.png"
    assert out.exists() and out.stat().st_size > 10_000
    assert png.endswith(".png")


def test_matplotlib_dashboard_tolerates_missing_episode_data(store_with_experiment):
    """Old telemetry (no 'episodes' block) must still render a dashboard."""
    pytest.importorskip("matplotlib")
    from harness.experiment_card import save_matplotlib_dashboard

    png = save_matplotlib_dashboard(store_with_experiment, "exp-test-card")
    assert png.endswith("exp-test-card.png")


# ---------------------------------------------------------------------------
# campaign story
# ---------------------------------------------------------------------------


def test_story_ledger_renders_levers_expected_observed(store_with_experiment):
    from harness.experiment_card import render_story, save_story

    store_with_experiment.experiments.record_verdict(
        "exp-test-card", "supported", "X worked", {"final_bench": 0.25}
    )
    story = render_story(store_with_experiment)
    assert "exp-test-card — SUPPORTED" in story
    assert "**Lever(s) pulled:** X enabled" in story
    assert "**Expected:**" in story and "if we add X" in story
    assert "**Observed:** X worked" in story
    path = save_story(store_with_experiment)
    assert path.endswith("story.md")


def test_story_prediction_excerpt_prefers_prediction_section():
    from harness.experiment_card import _prediction_excerpt

    hyp = "CONTEXT: blah.\n\nPREDICTION: bench >= 0.10 by iter 15.\n\nNOTES: etc."
    assert _prediction_excerpt(hyp) == "bench >= 0.10 by iter 15."
    assert _prediction_excerpt("if X then Y") == "if X then Y"
