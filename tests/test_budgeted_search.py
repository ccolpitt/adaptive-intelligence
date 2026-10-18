"""Budget-limited value-guided search: correctness of the sign conventions,
honest budget accounting, value-guided pruning, and the central claim — at an
equal number of states explored, a state budget reaches deeper than a depth
budget."""

import numpy as np
import pytest

from harness.budgeted_search import (
    ILLEGAL,
    BudgetedSearch,
    PerfectConnect4Model,
    QNetEvaluator,
    SearchConfig,
    budgeted_q,
)
from harness.tasks.connect4 import Connect4Env

BLIND = lambda obs: np.zeros(7)  # a network that knows nothing


def board_after(moves):
    env = Connect4Env()
    env.reset()
    for m in moves:
        env.step(m)
    return env


def win_in_one():
    """X has three on the bottom row; column 3 wins immediately for X."""
    env = board_after([0, 6, 1, 6, 2, 6])
    assert env.current_player == 1 and 3 in env.winning_moves(1)
    return env


def must_block():
    """O has three on the bottom row; X must play column 3 or lose."""
    env = board_after([6, 0, 6, 1, 5, 2])
    assert env.current_player == 1 and 3 in env.winning_moves(-1)
    return env


# ---------------------------------------------------------------------------
# correctness: it must see what the rules make certain
# ---------------------------------------------------------------------------


def test_finds_immediate_win_with_a_blind_network():
    env = win_in_one()
    search = budgeted_q(BLIND, node_budget=40)
    result = search.run(env.obs())
    assert result.best_action == 3
    assert np.isclose(result.action_values[3], 1.0), "a proven win is worth exactly +1"


def test_finds_forced_block_with_a_blind_network():
    """Requires two plies: play elsewhere, opponent wins. The search must
    discover that by itself, with no help from the network."""
    env = must_block()
    result = budgeted_q(BLIND, node_budget=400).run(env.obs())
    assert result.best_action == 3
    # Every alternative is discovered to be losing and ranks clearly below
    # the block. The magnitudes are diluted rather than exactly -1 because a
    # rejected move is only sampled a few times — spending the budget on the
    # move it believes in is the point of a best-first search, so the ORDERING
    # is the claim, not the precision of a line it declined to explore.
    for c in (0, 1, 2, 4, 5, 6):
        assert result.action_values[c] < -0.2, (
            f"column {c} loses; got {result.action_values[c]:.3f}"
        )
        assert result.action_values[c] < result.action_values[3] - 0.3


def test_illegal_actions_are_masked():
    env = Connect4Env()
    env.reset()
    for _ in range(6):  # fill column 0 with alternating pieces
        env.step(0)
    assert not env.done and 0 not in env.legal_actions()
    result = budgeted_q(BLIND, node_budget=60).run(env.obs())
    assert result.action_values[0] == ILLEGAL
    assert result.best_action != 0


def test_terminal_sign_convention_draw_is_zero():
    """A move that fills the board for a draw is worth exactly 0. Uses a 2x2
    board where connecting four is impossible, so filling it must be a draw —
    the sign convention is tested without a fragile 42-move construction."""
    env = Connect4Env(rows=2, cols=2, connect_n=4)
    env.reset()
    for m in (0, 0, 1):  # one empty cell left; playing it ends the game drawn
        env.step(m)
    assert not env.done and env.legal_actions() == [1]

    search = BudgetedSearch(
        QNetEvaluator(lambda obs: np.zeros(2), 2),
        PerfectConnect4Model(rows=2, cols=2),
        SearchConfig(node_budget=10),
    )
    result = search.run(env.obs())
    assert np.isclose(result.action_values[1], 0.0, atol=1e-6)
    assert result.action_values[0] == ILLEGAL


# ---------------------------------------------------------------------------
# budget accounting and the depth claim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("budget", [10, 50, 200, 600])
def test_budget_is_respected_exactly(budget):
    env = board_after([3, 3, 4])
    result = budgeted_q(BLIND, node_budget=budget).run(env.obs())
    assert result.nodes_evaluated <= budget, "the budget is a hard ceiling"
    assert result.nodes_evaluated >= min(budget, 2), "the budget should be used"


def test_state_budget_reaches_deeper_than_an_equal_depth_budget():
    """The reason for preferring a state budget. A uniform depth-3 search of a
    7-wide tree examines 1+7+49+343 = 400 states to see exactly 3 plies.
    Given the same 400 states on a position with real tactical structure,
    best-first search follows the line that matters and sees further."""
    env = must_block()
    uniform_depth_3_cost = 1 + 7 + 49 + 343
    result = budgeted_q(BLIND, node_budget=uniform_depth_3_cost).run(env.obs())
    assert result.nodes_evaluated == uniform_depth_3_cost
    assert result.max_depth_reached > 3, (
        f"expected deeper than the depth-3 equivalent, reached "
        f"{result.max_depth_reached} at {result.nodes_evaluated} states"
    )
    # ...and it got there by concentrating, not by spreading evenly.
    share = result.visit_counts.max() / result.visit_counts.sum()
    assert share > 0.5, f"expected a concentrated search, best move got {share:.2f}"


def test_uninformative_position_spreads_the_budget():
    """The honest converse: with a blind network on a quiet position there is
    nothing to prune toward, and best-first correctly degenerates to breadth.
    Pruning is only as good as the evaluator guiding it."""
    result = budgeted_q(BLIND, node_budget=400).run(board_after([3, 3, 4]).obs())
    spread = result.visit_counts[result.visit_counts > 0]
    assert spread.max() - spread.min() <= 2, "no signal, so no concentration"


def test_more_budget_never_hurts_a_proven_win():
    env = win_in_one()
    for budget in (20, 100, 500):
        result = budgeted_q(BLIND, node_budget=budget).run(env.obs())
        assert result.best_action == 3, f"lost the win at budget {budget}"


# ---------------------------------------------------------------------------
# value-guided pruning: the network decides where the budget goes
# ---------------------------------------------------------------------------


def test_priors_concentrate_the_budget():
    """A network that likes column 0 should make the search spend most of its
    visits there — that is pruning by value, not enumeration."""
    env = board_after([3, 3])

    def opinionated(obs):
        q = np.full(7, -0.9)
        q[0] = 0.9
        return q

    result = BudgetedSearch(
        QNetEvaluator(opinionated, 7, prior_temperature=0.25),
        PerfectConnect4Model(),
        SearchConfig(node_budget=300, c_puct=1.0),
    ).run(env.obs())
    share = result.visit_counts[0] / max(result.visit_counts.sum(), 1)
    assert share > 0.4, f"expected the budget to concentrate, column 0 got {share:.2f}"


def test_search_overrides_a_confidently_wrong_network():
    """The network insists on a losing move; the rules say block. Search wins
    the argument — this is why search amplifies a mediocre evaluator."""
    env = must_block()

    def wrong(obs):
        q = np.full(7, -0.5)
        q[6] = 1.0  # loves the move that loses
        return q

    greedy_choice = int(np.argmax(wrong(env.obs())))
    assert greedy_choice == 6, "fixture: the bare network picks the losing move"
    result = budgeted_q(wrong, node_budget=600).run(env.obs())
    assert result.best_action == 3


# ---------------------------------------------------------------------------
# outputs used downstream
# ---------------------------------------------------------------------------


def test_visit_policy_is_a_distribution_and_is_a_training_target():
    env = board_after([3, 3, 4])
    result = budgeted_q(BLIND, node_budget=200).run(env.obs())
    policy = result.visit_policy()
    assert np.isclose(policy.sum(), 1.0)
    assert np.all(policy >= 0)
    assert np.isclose(result.visit_policy(temperature=0).sum(), 1.0)
    assert policy[result.visit_counts.argmax()] > 0


def test_is_a_drop_in_q_function():
    env = board_after([3])
    search = budgeted_q(BLIND, node_budget=50)
    values = search(env.obs())  # called like any QFunction
    assert values.shape == (7,)
    assert search.total_moves == 1 and search.total_nodes_evaluated > 0


def test_dirichlet_noise_varies_training_games_but_not_evaluation():
    env = board_after([3, 3])
    noisy = [BudgetedSearch(
        QNetEvaluator(BLIND, 7), PerfectConnect4Model(),
        SearchConfig(node_budget=120, dirichlet_noise=0.5, seed=s),
    ).run(env.obs()).visit_counts for s in (1, 2, 3)]
    assert any(not np.array_equal(noisy[0], v) for v in noisy[1:]), "noise should vary play"

    clean = [budgeted_q(BLIND, node_budget=120, seed=s).run(env.obs()).visit_counts
             for s in (1, 2, 3)]
    assert all(np.array_equal(clean[0], v) for v in clean[1:]), "evaluation must be deterministic"


# ---------------------------------------------------------------------------
# the learned-model path (simulation-to-reality gap)
# ---------------------------------------------------------------------------


def test_low_model_confidence_shrinks_deep_values():
    """A learned model that is only partly trusted must not let a deep,
    speculative line dominate. Confidence decay does that automatically."""
    env = must_block()
    trusted = budgeted_q(BLIND, node_budget=400).run(env.obs())
    untrusted = budgeted_q(BLIND, node_budget=400, model_confidence_decay=0.5).run(env.obs())
    # Both should still block (one ply of real consequence), but the magnitude
    # of the deep verdict must be smaller when the model is doubted.
    assert untrusted.best_action == 3
    assert abs(untrusted.action_values[6]) < abs(trusted.action_values[6])


def test_model_interface_reports_confidence():
    model = PerfectConnect4Model()
    state = model.root_state(board_after([3]).obs())
    transition = model.apply(state, 3)
    assert transition.confidence == 1.0, "exact rules are fully trusted"
    assert not transition.done
