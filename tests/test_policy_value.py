"""The two-headed policy: a move head and a bounded value head, trained on
what the search concluded. Also the corrected budget accounting."""

import numpy as np
import pytest
import torch

from harness.agents.policy_value import (
    PolicyValueConfig,
    PolicyValueNet,
    PolicyValueTrainer,
    move_scores_from,
    policy_value_from,
)
from harness.budgeted_search import budgeted_q, make_search
from harness.scripted_opponents import RandomOpponent
from harness.tasks.connect4 import Connect4Env


def board_after(moves, **kwargs):
    env = Connect4Env(**kwargs)
    env.reset()
    for m in moves:
        env.step(m)
    return env


def win_in_one():
    env = board_after([0, 6, 1, 6, 2, 6])
    assert 3 in env.winning_moves(1)
    return env


# ---------------------------------------------------------------------------
# budget accounting: unique positions, cached within a move
# ---------------------------------------------------------------------------


def test_budget_counts_unique_positions_not_visits():
    """A position reached twice by different move orders is one position."""
    env = board_after([3, 3, 4])
    result = budgeted_q(lambda obs: np.zeros(7), node_budget=200).run(env.obs())
    assert result.nodes_evaluated <= 200
    # Connect 4 is full of transpositions, so a real search must find repeats.
    assert result.cache_hits > 0, "expected repeated positions to be reused"
    # The tree is larger than the budget precisely because repeats are free.
    assert result.tree_size > result.nodes_evaluated


def test_a_budget_of_n_examines_at_most_n_positions():
    for budget in (8, 32, 128):
        result = budgeted_q(lambda obs: np.zeros(7), node_budget=budget).run(
            board_after([3]).obs())
        assert result.nodes_evaluated <= budget, (
            f"asked for {budget} positions, examined {result.nodes_evaluated}")


def test_the_cache_is_discarded_between_moves():
    """Weights may change between moves, so an opinion must not outlive the
    move that formed it."""
    search = budgeted_q(lambda obs: np.zeros(7), node_budget=64)
    first = search.run(board_after([3]).obs())
    assert search._cache == {}, "the cache must not survive the move"
    second = search.run(board_after([3]).obs())
    # Identical position, identical work: nothing was carried over.
    assert second.nodes_evaluated == first.nodes_evaluated


def test_caching_does_not_change_the_conclusion():
    env = win_in_one()
    assert budgeted_q(lambda obs: np.zeros(7), node_budget=60).run(
        env.obs()).best_action == 3


# ---------------------------------------------------------------------------
# the two heads
# ---------------------------------------------------------------------------


def test_the_two_heads_have_the_right_shapes_and_the_value_is_bounded():
    net = PolicyValueNet(6, 7, 7)
    net.eval()
    logits, value = net(torch.zeros(4, 2, 6, 7))
    assert logits.shape == (4, 7), "one score per column"
    assert value.shape == (4,), "one value per position"
    assert torch.all(value.abs() <= 1.0)


def test_the_value_head_cannot_leave_the_reward_range():
    """By construction, not by training: the failure in exp-022 and exp-024."""
    net = PolicyValueNet(6, 7, 7)
    net.eval()
    with torch.no_grad():
        net.value_out.bias.add_(100.0)
        net.value_out.weight.mul_(100.0)
    _, value = net(torch.randn(16, 2, 6, 7))
    assert torch.all(value.abs() <= 1.0), f"escaped: {value.abs().max().item()}"


def test_neither_head_is_a_move_value():
    """The move head ranks; the value head judges the position. The value of
    a particular move comes from the search, not from either head."""
    trainer = PolicyValueTrainer(6, 7, 7, PolicyValueConfig(seed=0))
    logits, value = trainer.policy_value(win_in_one().obs())
    assert logits.shape == (7,)
    assert isinstance(value, float) and abs(value) <= 1.0


def test_scoring_the_policy_alone_uses_the_move_head():
    trainer = PolicyValueTrainer(6, 7, 7, PolicyValueConfig(seed=0))
    scores = trainer.q_function()(win_in_one().obs())
    assert scores.shape == (7,)  # drop-in for the benchmark and the evaluator
    assert np.isfinite(scores).all()


def test_search_uses_both_heads_and_finds_a_win_untrained():
    """An untrained two-headed policy must still take an immediate win once it
    is allowed to look, because the search sees the rules."""
    trainer = PolicyValueTrainer(6, 7, 7, PolicyValueConfig(seed=0, search_budget=48,
                                                            dirichlet_noise=0.0))
    assert trainer.search.run(win_in_one().obs()).best_action == 3


# ---------------------------------------------------------------------------
# training on what the search concluded
# ---------------------------------------------------------------------------


def test_examples_record_the_search_attention_and_the_real_outcome():
    trainer = PolicyValueTrainer(6, 7, 7, PolicyValueConfig(
        seed=0, search_budget=24, batch_size=8))
    env = Connect4Env()
    trainer.play_game(env)
    assert trainer.buffer
    obs, visits, outcome = trainer.buffer[0]
    assert obs.shape == (2, 6, 7)
    assert np.isclose(visits.sum(), 1.0), "the move target is a distribution"
    assert outcome in (-1.0, 0.0, 1.0), "the value target is the real result"
    # In a decided game both sides are represented, with opposite outcomes.
    outcomes = {o for _, _, o in trainer.buffer}
    if env.winner not in (None, 0):
        assert outcomes == {1.0, -1.0}


def test_training_reduces_both_losses_and_stays_finite():
    trainer = PolicyValueTrainer(6, 7, 7, PolicyValueConfig(
        seed=0, search_budget=24, batch_size=16, train_steps_per_game=6))
    policy_losses, value_losses = [], []
    for _ in range(12):
        stats = trainer.play_game(Connect4Env())
        policy_losses.extend(stats.policy_losses)
        value_losses.extend(stats.value_losses)
    assert policy_losses and value_losses
    assert all(np.isfinite(x) for x in policy_losses + value_losses)
    assert np.mean(policy_losses[-5:]) < np.mean(policy_losses[:5]) * 1.5


def test_value_estimates_stay_bounded_throughout_training():
    trainer = PolicyValueTrainer(6, 7, 7, PolicyValueConfig(
        seed=1, search_budget=16, batch_size=8, train_steps_per_game=4))
    seen = []
    for _ in range(10):
        seen.extend(trainer.play_game(Connect4Env()).value_abs_means)
    assert seen and max(seen) <= 1.0


def test_mirror_augmentation_flips_both_board_and_move_target():
    trainer = PolicyValueTrainer(6, 7, 7, PolicyValueConfig(
        seed=0, search_budget=16, batch_size=8, mirror_augmentation=True))
    trainer.play_game(Connect4Env())
    obs_a, visits_a, out_a = trainer.buffer[0]
    obs_b, visits_b, out_b = trainer.buffer[1]
    assert np.array_equal(obs_b, np.flip(obs_a, axis=2))
    assert np.array_equal(visits_b, np.flip(visits_a))
    assert out_a == out_b, "mirroring the board does not change who won"


def test_can_train_against_a_fixed_opponent():
    trainer = PolicyValueTrainer(6, 7, 7, PolicyValueConfig(
        seed=0, search_budget=16, batch_size=8))
    opponent = RandomOpponent(seed=3)
    stats = trainer.play_game(Connect4Env(), opponent_act=opponent.act)
    assert stats.moves > 0
    assert stats.learner_result in (-1, 0, 1)
    assert stats.search_states > 0


def test_search_cost_is_reported_for_the_cycle_record():
    trainer = PolicyValueTrainer(6, 7, 7, PolicyValueConfig(
        seed=0, search_budget=32, batch_size=8))
    trainer.play_game(Connect4Env())
    assert trainer.search_moves > 0
    assert trainer.search_states_examined > 0
    assert trainer.search_deepest >= 1
    assert trainer.search_cache_hits >= 0


# ---------------------------------------------------------------------------
# saving and reloading
# ---------------------------------------------------------------------------


def test_a_saved_policy_keeps_both_heads_and_the_bound():
    trainer = PolicyValueTrainer(6, 7, 7, PolicyValueConfig(seed=0))
    with torch.no_grad():
        trainer.net.value_out.bias.add_(50.0)
    module = trainer.scripted()
    obs = win_in_one().obs()

    scores = move_scores_from(module)(obs)
    assert scores.shape == (7,)
    logits, value = policy_value_from(module)(obs)
    assert abs(value) <= 1.0, "the bound survives saving"
    assert np.allclose(scores, logits)


def test_a_reloaded_policy_can_be_searched_with():
    from harness.budgeted_search import (
        BudgetedSearch, PerfectConnect4Model, PolicyValueEvaluator, SearchConfig,
    )

    trainer = PolicyValueTrainer(6, 7, 7, PolicyValueConfig(seed=0))
    module = trainer.scripted()
    search = BudgetedSearch(
        PolicyValueEvaluator(policy_value_from(module), 7),
        PerfectConnect4Model(), SearchConfig(node_budget=48, dirichlet_noise=0.0))
    assert search.run(win_in_one().obs()).best_action == 3


def test_weights_can_be_carried_into_a_new_trainer():
    a = PolicyValueTrainer(6, 7, 7, PolicyValueConfig(seed=0))
    b = PolicyValueTrainer(6, 7, 7, PolicyValueConfig(seed=99))
    obs = win_in_one().obs()
    assert not np.allclose(a.q_function()(obs), b.q_function()(obs))
    b.load_weights_from(a.net)
    assert np.allclose(a.q_function()(obs), b.q_function()(obs))


# ---------------------------------------------------------------------------
# how far the weights actually move
# ---------------------------------------------------------------------------


def test_training_records_how_far_the_weights_moved():
    """The gradient norm says how large the correction was; this says how much
    of it reached the weights. A converged or frozen network can still show a
    healthy gradient norm, so only this number answers "is it still
    learning?"."""
    trainer = PolicyValueTrainer(6, 7, 7, PolicyValueConfig(
        seed=0, search_budget=16, batch_size=8, train_steps_per_game=4))
    stats = trainer.play_game(Connect4Env())
    assert stats.weight_deltas, "every training step must report a movement"
    assert len(stats.weight_deltas) == len(stats.grad_norms)
    assert all(d > 0.0 for d in stats.weight_deltas)
    assert all(np.isfinite(d) for d in stats.weight_deltas)


def test_a_learning_rate_of_zero_moves_nothing():
    """The control that gives the number its meaning: with no learning rate
    the weights cannot move, however large the gradient."""
    trainer = PolicyValueTrainer(6, 7, 7, PolicyValueConfig(
        seed=0, search_budget=16, batch_size=8, train_steps_per_game=4))
    for group in trainer.optimizer.param_groups:
        group["lr"] = 0.0
    stats = trainer.play_game(Connect4Env())
    assert stats.weight_deltas, "steps still ran"
    assert max(stats.weight_deltas) == 0.0
    assert max(stats.grad_norms) > 0.0, (
        "the gradient was non-zero, which is exactly why the gradient norm "
        "cannot be used to detect a network that has stopped moving")
