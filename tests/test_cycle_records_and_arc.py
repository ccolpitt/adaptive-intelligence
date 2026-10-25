"""Cycle records (data, not pictures), versioned renderer dispatch, and the
learning arc that tracks strength across cycles."""

import json

import pytest

from harness.cycle_record import CycleRecord, CycleStore
from harness.learning_arc import render_plot, render_text, save_summary, summarize
from harness.lineage import Lineage, LineageStore, RubricStore, rubric_r002_budgeted_lookahead
from harness.renderers import available, get_renderer, latest_for


def make_record(cycle=0, lineage="connect4-L2", score=0.05, promoted=False,
                games=500, with_series=True, renderer="connect4_v1"):
    series = {}
    if with_series:
        n = 40
        series = {
            "loss": [1.0 / (i + 1) for i in range(n)],
            "grad_norm": [0.5] * n,
            "q_mean_abs": [0.4 + 0.001 * i for i in range(n)],
            "result": [(-1) ** i for i in range(n)],
            "unique_states": list(range(10, 10 + n)),
            "epsilon": [0.5 - 0.01 * i for i in range(n)],
            "search_states": [30] * n,
        }
    return CycleRecord(
        cycle=cycle, lineage_id=lineage, task_id="connect4",
        task_evaluation_version=1, rubric_id="r002-lookahead-32-puct",
        renderer=renderer, games_trained=games,
        cumulative_games=games * (cycle + 1), parameter_count=183_000,
        wall_seconds=600.0, search_budget=32, search_states_examined=16_000,
        search_deepest_ply=9, series=series,
        challenger_id=f"p{cycle}", champion_id="p0",
        evaluation={"score": score, "frontier_depth": 1,
                    "per_rung": {"1": 0.3, "2": 0.0}},
        head_to_head={"score": 0.52, "games": 200},
        promoted=promoted, parity=not promoted,
        gate_reason="rejected at parity: the change looks sound but under-dosed",
        probes={"q_of_game_winning_moves": 0.8, "q_at_solver_proven_wins": 0.7,
                "q_of_not_blocking": -0.6, "q_at_solver_proven_losses": -0.5,
                "weight_distribution": {
                    "conv.0.weight": {"dead_fraction": 0.08, "exploding_fraction": 0.0},
                    "fc1.weight": {"dead_fraction": 0.15, "exploding_fraction": 0.001}}},
        diagnosis=[{"severity": "warn", "finding": "value estimates drifting",
                    "meaning": "m", "hypotheses": ["lower the learning rate"]}],
    )


# ---------------------------------------------------------------------------
# records are data
# ---------------------------------------------------------------------------


def test_record_round_trips_through_json(tmp_path):
    store = CycleStore(tmp_path / "cycles")
    path = store.save(make_record(cycle=3))
    assert path.suffix == ".json", "a cycle record is data, never a picture"
    loaded = store.load("connect4-L2", 3)
    assert loaded.cycle == 3
    assert loaded.rubric_id == "r002-lookahead-32-puct"
    assert loaded.renderer == "connect4_v1"
    assert loaded.series["loss"][0] == 1.0
    assert loaded.probes["q_of_game_winning_moves"] == 0.8


def test_record_tolerates_unknown_future_fields(tmp_path):
    """A record written by a later version must still load."""
    store = CycleStore(tmp_path / "cycles")
    store.save(make_record(cycle=0))
    path = store.path_for("connect4-L2", 0)
    blob = json.loads(path.read_text())
    blob["some_future_measurement"] = {"a": 1}
    path.write_text(json.dumps(blob))
    assert store.load("connect4-L2", 0).cycle == 0


def test_store_tracks_sequence_and_cost(tmp_path):
    store = CycleStore(tmp_path / "cycles")
    assert store.next_cycle_index("connect4-L2") == 0
    for c in range(3):
        store.save(make_record(cycle=c, games=400))
    assert store.next_cycle_index("connect4-L2") == 3
    assert store.cumulative_games("connect4-L2") == 1200
    assert [r.cycle for r in store.load_all("connect4-L2")] == [0, 1, 2]


def test_summary_line_is_readable():
    line = make_record(cycle=7, score=0.123, promoted=True).summary_line()
    assert "cycle   7" in line and "0.123" in line and "PROMOTED" in line


# ---------------------------------------------------------------------------
# versioned renderers
# ---------------------------------------------------------------------------


def test_renderer_registry():
    # Older versions stay available forever, because records name them.
    assert "connect4_v1" in available()
    # The latest moves on as new measurements are added.
    assert latest_for("connect4") in available()
    assert latest_for("connect4") >= "connect4_v1"
    with pytest.raises(ValueError, match="no renderer registered"):
        latest_for("chess")
    with pytest.raises(ValueError, match="unknown renderer"):
        get_renderer("connect4_v99")


def test_newer_renderer_adds_measurements_without_changing_the_old_one():
    """The versioning contract: v2 declares what it added, and v1 is
    unchanged so cycles recorded under it still draw as intended."""
    v1 = get_renderer("connect4_v1").describe()
    v2 = get_renderer("connect4_v2").describe()
    assert v1["panels"][4] == "win rate in training games"
    assert "random" in v2["panels"][4]
    assert v2["new_since_v1"]
    assert "win_rate_vs_random" in v2["expected_evaluation"]


def test_a_v1_record_still_renders_after_v2_exists(tmp_path):
    pytest.importorskip("matplotlib")
    out = make_record(cycle=2, renderer="connect4_v1").render(out_dir=tmp_path)
    assert out.endswith("cycle-0002.png")


def test_renderer_declares_what_it_measures():
    described = get_renderer("connect4_v1").describe()
    assert described["version"] == "connect4_v1"
    assert "loss" in described["expected_series"]
    assert len(described["panels"]) == 9


def test_record_renders_itself_through_its_own_renderer(tmp_path):
    pytest.importorskip("matplotlib")
    out = make_record(cycle=5).render(out_dir=tmp_path / "renders")
    assert out.endswith("cycle-0005.png")
    assert (tmp_path / "renders" / "cycle-0005.png").stat().st_size > 10_000


def test_rendering_survives_missing_measurements(tmp_path):
    """A cycle recorded before a measurement existed must still draw."""
    pytest.importorskip("matplotlib")
    record = make_record(cycle=1, with_series=False)
    record.probes = {}
    record.search_budget = 0
    assert record.render(out_dir=tmp_path / "renders").endswith("cycle-0001.png")


def test_a_record_naming_a_missing_renderer_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="unknown renderer"):
        make_record(renderer="connect4_v7").render(out_dir=tmp_path)


# ---------------------------------------------------------------------------
# the learning arc
# ---------------------------------------------------------------------------


@pytest.fixture
def arc_store(tmp_path):
    RubricStore(tmp_path / "rubrics").publish(rubric_r002_budgeted_lookahead(32))
    LineageStore(tmp_path / "lineages").create(Lineage(
        lineage_id="connect4-L2", task_id="connect4",
        rubric_id="r002-lookahead-32-puct",
        description="Second line: lookahead during training.",
        parent_lineage="connect4-L1"))
    cycles = CycleStore(tmp_path / "cycles")
    # A climbing arc with promotions at cycles 0, 3 and 7.
    scores = [0.04, 0.03, 0.05, 0.09, 0.07, 0.08, 0.10, 0.16, 0.15]
    for c, s in enumerate(scores):
        cycles.save(make_record(cycle=c, score=s, promoted=c in (0, 3, 7), games=500))
    return tmp_path


def test_arc_tracks_strength_and_cost(arc_store):
    summary = summarize(arc_store, "connect4-L2")
    assert summary["cycles"] == 9
    assert summary["total_games"] == 4500
    assert summary["promotions"] == 3
    assert summary["best_challenger_score"] == 0.16
    assert summary["current_champion_score"] == 0.16  # last promotion
    assert summary["parent_lineage"] == "connect4-L1"


def test_arc_carries_the_rubric_that_trained_it(arc_store):
    summary = summarize(arc_store, "connect4-L2")
    assert summary["rubrics_used"] == ["r002-lookahead-32-puct"]
    assert "32 states/move by puct" in summary["rubric_summaries"]["r002-lookahead-32-puct"]


def test_arc_detects_improvement_and_flatness(arc_store, tmp_path):
    assert summarize(arc_store, "connect4-L2")["trend"] == "improving"

    flat = tmp_path / "flat"
    LineageStore(flat / "lineages").create(Lineage(
        lineage_id="flat-L", task_id="connect4", rubric_id="r", description="d"))
    cycles = CycleStore(flat / "cycles")
    for c in range(9):
        cycles.save(make_record(cycle=c, lineage="flat-L", score=0.05,
                                promoted=(c == 0)))
    assert summarize(flat, "flat-L")["trend"] == "flat"


def test_champion_score_only_moves_on_promotion(arc_store):
    points = summarize(arc_store, "connect4-L2")["points"]
    # cycles 1 and 2 are rejections between promotions at 0 and 3
    assert points[1]["champion_score"] == points[0]["champion_score"] == 0.04
    assert points[3]["champion_score"] == 0.09


def test_arc_renders_as_text_json_and_plot(arc_store):
    pytest.importorskip("matplotlib")
    summary = summarize(arc_store, "connect4-L2")
    text = render_text(summary)
    assert "LEARNING ARC" in text and "PROMOTED" in text and "4,500" in text

    path = save_summary(arc_store, "connect4-L2")
    assert path.endswith("connect4-L2.json")
    assert json.loads(open(path).read())["cycles"] == 9

    png = render_plot(summary, arc_store / "arc.png")
    assert (arc_store / "arc.png").stat().st_size > 10_000


def test_arc_of_an_empty_lineage_is_harmless(tmp_path):
    LineageStore(tmp_path / "lineages").create(Lineage(
        lineage_id="empty-L", task_id="connect4", rubric_id="r", description="d"))
    summary = summarize(tmp_path, "empty-L")
    assert summary["cycles"] == 0 and summary["trend"] == "insufficient data"
    with pytest.raises(ValueError, match="no cycles recorded"):
        render_plot(summary)
