"""The expectation checks that ask "does this result make sense?", move
selection by visit count, and equal-budget head-to-head matches.

Each sanity check here is written against a defect this project actually
had, so the tests are also a record of what went wrong.
"""

import numpy as np
import pytest

from harness.budgeted_search import SearchConfig, budgeted_q, make_search
from harness.sanity import (
    CONTRADICTION,
    INCONSISTENT,
    UNEXPECTED,
    check_cycle,
    render,
    worst_severity,
)
from harness.tasks.connect4 import Connect4Env

from tests.test_cycle_records_and_arc import make_record


def findings_for(**changes):
    record = make_record(**{k: v for k, v in changes.items()
                            if k in ("cycle", "promoted", "games")})
    for key, value in changes.items():
        if key not in ("cycle", "promoted", "games"):
            setattr(record, key, value)
    return check_cycle(record, changes.get("history", []))


def names(findings):
    return {f.check for f in findings}


# ---------------------------------------------------------------------------
# things that cannot be true
# ---------------------------------------------------------------------------


def test_a_clean_cycle_raises_nothing():
    record = make_record()
    record.probes = dict(record.probes)
    record.probes.pop("weight_distribution", None)
    record.search_deepest_ply = 9
    record.evaluation = {**record.evaluation, "win_rate_vs_random": 0.95,
                         "score_with_lookahead": 0.3}
    assert check_cycle(record, []) == []
    assert "every expectation held" in render([])


def test_spending_more_than_the_budget_is_a_contradiction():
    """The defect the human asked to have fixed: a budget of 32 that examined
    400 positions. Both series are per GAME, so the budget comparison must
    divide by that game's move count."""
    record = make_record()
    record.search_budget = 32
    record.series = {**record.series,
                     "search_states": [400 * 20] * 40,  # 400 per move
                     "moves": [20] * 40}
    findings = check_cycle(record, [])
    assert any(f.severity == CONTRADICTION for f in findings)
    assert "lookahead stayed inside its budget" in names(findings)


def test_an_on_budget_cycle_is_not_flagged():
    """Regression for R-006. A per-game total of 8,400 positions against a
    per-move budget of 200 is exactly on budget across a 42-move game; the
    check compared the two without dividing and raised a false
    contradiction, sending the reader after a defect that did not exist."""
    record = make_record()
    record.search_budget = 200
    record.series = {**record.series,
                     "search_states": [200 * 42] * 40,
                     "moves": [42] * 40}
    assert not any(f.severity == CONTRADICTION for f in check_cycle(record, []))


def test_a_budget_that_cannot_be_verified_says_so():
    """Records written before moves-per-game was captured must report that
    the check could not run, rather than passing silently or failing."""
    record = make_record()
    record.search_budget = 200
    record.series = {**record.series, "search_states": [8400] * 40}
    record.series.pop("moves", None)
    findings = check_cycle(record, [])
    assert not any(f.severity == CONTRADICTION for f in findings)
    assert "the budget can be checked" in names(findings)


def test_a_bounded_value_head_exceeding_one_is_a_contradiction():
    """exp-025: the rubric asked for bounded values and the setting never
    reached the trainer."""
    record = make_record()
    record.probes = {**record.probes, "q_max_abs": 2.43,
                     "move_score_max_abs": 5.0}  # marks a two-headed policy
    findings = check_cycle(record, [])
    assert any(f.severity == CONTRADICTION and "reward range" in f.check
               for f in findings)


def test_unbounded_single_head_values_are_not_flagged_as_contradictions():
    """A single-output learner is not squashed, so a large value is a
    finding for the diagnosis rules, not an impossibility."""
    record = make_record()
    record.probes = {k: v for k, v in record.probes.items()
                     if k != "move_score_max_abs"}
    record.probes["q_max_abs"] = 2.43
    assert not any(f.severity == CONTRADICTION for f in check_cycle(record, []))


def test_non_finite_weights_are_a_contradiction():
    record = make_record()
    record.probes = {**record.probes, "weight_distribution": {
        "fc1.weight": {"dead_fraction": 0.1, "exploding_fraction": 0.0,
                       "nonfinite": 7}}}
    findings = check_cycle(record, [])
    assert any(f.severity == CONTRADICTION and "finite" in f.check for f in findings)


# ---------------------------------------------------------------------------
# things that disagree with each other
# ---------------------------------------------------------------------------


def test_thinking_making_the_policy_worse_is_inconsistent():
    record = make_record(score=0.40)
    record.evaluation = {**record.evaluation, "score": 0.40,
                         "score_with_lookahead": 0.12, "eval_budget": 128}
    findings = check_cycle(record, [])
    assert "thinking does not make the policy worse" in names(findings)
    finding = next(f for f in findings if "thinking" in f.check)
    assert finding.severity == INCONSISTENT
    assert finding.look_at, "a finding must say where to look"


def test_missing_easy_wins_despite_a_large_budget_is_inconsistent():
    """The live question in the current lineages: a search of this size always
    sees a win one move away."""
    record = make_record()
    record.search_budget = 200
    record.probes = {**record.probes, "take_the_win_rate": 0.25}
    findings = check_cycle(record, [])
    assert "a policy trained with lookahead takes an immediate win" in names(findings)


def test_lost_positions_rated_above_won_positions_is_inconsistent():
    record = make_record()
    record.probes = {**record.probes, "q_at_solver_proven_wins": -0.3,
                     "q_at_solver_proven_losses_mean": 0.4}
    findings = check_cycle(record, [])
    assert "won positions are rated above lost positions" in names(findings)


def test_a_promotion_that_lowers_the_champion_is_inconsistent():
    """The ratchet: two promotions each individually forgivable took the
    champion from 0.083 to 0.042."""
    history = [make_record(cycle=0, score=0.30, promoted=True)]
    record = make_record(cycle=1, score=0.10, promoted=True)
    findings = check_cycle(record, history)
    assert "a promotion does not lower the champion's score" in names(findings)


def test_the_network_changing_shape_mid_lineage_is_inconsistent():
    history = [make_record(cycle=0)]
    record = make_record(cycle=1)
    record.parameter_count = 999_999
    findings = check_cycle(record, history)
    assert "the network does not change shape mid-lineage" in names(findings)


def test_exploration_stalling_is_inconsistent():
    record = make_record()
    record.series = {**record.series, "unique_states": [100] * 40}
    findings = check_cycle(record, [])
    assert "new positions keep being seen" in names(findings)


# ---------------------------------------------------------------------------
# allowed but unexpected
# ---------------------------------------------------------------------------


def test_losing_to_random_after_many_games_is_unexpected():
    record = make_record()
    record.cumulative_games = 8000
    record.evaluation = {**record.evaluation, "win_rate_vs_random": 0.30}
    findings = check_cycle(record, [])
    assert "a trained policy beats a random player" in names(findings)
    assert all(f.severity != CONTRADICTION for f in findings)


def test_the_floor_falling_back_is_unexpected():
    history = [make_record(cycle=i) for i in range(3)]
    for h in history:
        h.evaluation = {**h.evaluation, "win_rate_vs_random": 0.98}
    record = make_record(cycle=3)
    record.evaluation = {**record.evaluation, "win_rate_vs_random": 0.60}
    assert "the floor does not fall back" in names(check_cycle(record, history))


def test_a_large_budget_reaching_only_two_moves_is_unexpected():
    record = make_record()
    record.search_budget = 200
    record.search_deepest_ply = 2
    assert "a large budget looks further than a small one" in names(
        check_cycle(record, []))


def test_severity_ordering_and_headline():
    record = make_record()
    record.search_budget = 32
    record.series = {**record.series, "search_states": [400 * 20] * 40,
                     "moves": [20] * 40}
    findings = check_cycle(record, [])
    assert findings[0].severity == CONTRADICTION, "worst first"
    assert worst_severity(findings) == CONTRADICTION
    assert "CONTRADICTION WAS FOUND" in render(findings)


# ---------------------------------------------------------------------------
# move selection by visit count (as AlphaZero does)
# ---------------------------------------------------------------------------


def test_visit_selection_picks_the_most_attended_move():
    env = Connect4Env()
    env.reset()
    env.step(3)
    result = budgeted_q(lambda obs: np.zeros(7), node_budget=200).run(env.obs())
    assert result.best_action_by_visits == int(np.argmax(result.visit_counts))


def test_visit_selection_still_takes_a_proven_win():
    env = Connect4Env()
    env.reset()
    for m in (0, 6, 1, 6, 2, 6):
        env.step(m)
    search = make_search(lambda obs: np.zeros(7), budget=120, select_by="visits")
    assert search.run(env.obs()).best_action_by_visits == 3
    # Called as a policy, the scores are visit counts, so the largest is the
    # most-attended move and illegal columns stay excluded.
    scores = search(env.obs())
    assert int(np.argmax(scores)) == 3


def test_visit_selection_excludes_illegal_moves():
    env = Connect4Env()
    env.reset()
    for _ in range(6):
        env.step(0)
    scores = make_search(lambda obs: np.zeros(7), budget=60,
                         select_by="visits")(env.obs())
    assert int(np.argmax(scores)) != 0


def test_value_selection_remains_available_for_recorded_experiments():
    env = Connect4Env()
    env.reset()
    env.step(3)
    assert SearchConfig().select_by == "value", (
        "the default must not change, or already-recorded experiments would "
        "not be reproducible")


# ---------------------------------------------------------------------------
# equal-budget head-to-head
# ---------------------------------------------------------------------------


def test_head_to_head_gives_both_sides_the_same_budget(tmp_path):
    from harness.cycle_loop import CycleLoopConfig, run_cycles
    from harness.lineage import (
        Lineage, LineageStore, RubricStore, rubric_r002_budgeted_lookahead,
    )
    from harness.tasks.connect4 import Connect4Task

    RubricStore(tmp_path / "rubrics").publish(rubric_r002_budgeted_lookahead(4))
    LineageStore(tmp_path / "lineages").create(Lineage(
        lineage_id="h2h-L", task_id="connect4",
        rubric_id="r002-lookahead-4-puct", description="equal budget test"))

    records = run_cycles(Connect4Task(benchmark_games=4), tmp_path, CycleLoopConfig(
        lineage_id="h2h-L", rubric_id="r002-lookahead-4-puct",
        games_per_cycle=4, cycles=1, gate_games=4, random_eval_games=4,
        probe_games=2, seed=11, renderer="connect4_v2", verbose=False,
        h2h_budget=16))

    h2h = records[0].head_to_head
    assert h2h["budget_each"] == 16, "the budget each side received is recorded"
    assert h2h["games"] == 4
    assert 0.0 <= h2h["score"] <= 1.0


# ---------------------------------------------------------------------------
# has the network stopped moving?
# ---------------------------------------------------------------------------


def test_a_network_that_does_not_move_at_all_is_a_contradiction():
    """Training applied an update, so the weights must have changed. Zero
    movement means no learning happened, and more games on the same material
    cannot change that."""
    record = make_record()
    record.series = {**record.series, "weight_delta": [0.0] * 40}
    findings = check_cycle(record, [])
    assert "the network is still changing" in names(findings)
    assert any(f.severity == CONTRADICTION and "still changing" in f.check
               for f in findings)


def test_a_network_that_has_nearly_stopped_moving_is_unexpected():
    """Shrinking updates are normal as a policy converges, but below roughly
    1e-6 the network is absorbing nothing new, which is the signal to change
    the training material rather than add more of it."""
    record = make_record()
    record.series = {**record.series, "weight_delta": [5e-7] * 40}
    findings = check_cycle(record, [])
    assert "the network is still learning usefully" in names(findings)
    assert not any(f.severity == CONTRADICTION for f in findings)


def test_healthy_weight_movement_is_not_flagged():
    record = make_record()
    record.series = {**record.series, "weight_delta": [1e-4] * 40}
    findings = check_cycle(record, [])
    assert not any("network is still" in f.check for f in findings)


def test_a_cycle_without_the_measurement_is_not_flagged():
    """Records written before the metric existed must not raise a finding
    about a number they never captured."""
    record = make_record()
    record.series = {**record.series, "weight_delta": [None] * 40}
    findings = check_cycle(record, [])
    assert not any("network is still" in f.check for f in findings)
