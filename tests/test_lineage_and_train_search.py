"""Training rubrics, lineages, and budgeted lookahead inside training."""

import numpy as np
import pytest

from harness.agents.dqn import DQNTrainer, TrainerConfig
from harness.budgeted_search import SPEND_RULES, make_search
from harness.lineage import (
    Lineage,
    LineageStore,
    RubricStore,
    TrainingRubric,
    rubric_r001_dqn_no_lookahead,
    rubric_r002_budgeted_lookahead,
)
from harness.scripted_opponents import RandomOpponent
from harness.tasks.connect4 import Connect4Env


# ---------------------------------------------------------------------------
# rubrics: the recipe that made a policy, frozen
# ---------------------------------------------------------------------------


def test_publish_and_read_a_rubric(tmp_path):
    store = RubricStore(tmp_path / "rubrics")
    store.publish(rubric_r002_budgeted_lookahead(32))
    r = store.get("r002-lookahead-32-puct")
    assert r.search_budget == 32 and r.spend_rule == "puct"
    assert r.bounded_targets is True
    assert "32 states/move by puct" in r.summary()


def test_changing_a_published_rubric_is_refused(tmp_path):
    store = RubricStore(tmp_path / "rubrics")
    store.publish(rubric_r002_budgeted_lookahead(32))
    store.publish(rubric_r002_budgeted_lookahead(32))  # identical: fine
    altered = rubric_r002_budgeted_lookahead(32)
    altered.search_budget = 400  # same id, different recipe
    with pytest.raises(ValueError, match="already published"):
        store.publish(altered)


def test_rubric_validation_catches_incoherent_recipes(tmp_path):
    store = RubricStore(tmp_path / "rubrics")
    bad = rubric_r002_budgeted_lookahead(32)
    bad.spend_rule = "vibes"
    with pytest.raises(ValueError, match="spend_rule"):
        store.publish(bad)
    contradictory = rubric_r001_dqn_no_lookahead()
    contradictory.search_budget = 32  # a budget with no rule for spending it
    with pytest.raises(ValueError, match="needs a rule"):
        store.publish(contradictory)
    unused = rubric_r002_budgeted_lookahead(32)
    unused.search_budget = 0  # a spending rule with nothing to spend
    with pytest.raises(ValueError, match="zero budget"):
        store.publish(unused)


# ---------------------------------------------------------------------------
# lineages: how a line of policies evolved
# ---------------------------------------------------------------------------


@pytest.fixture
def lineage_store(tmp_path):
    store = LineageStore(tmp_path / "lineages")
    store.create(Lineage(
        lineage_id="connect4-L2", task_id="connect4",
        rubric_id="r002-lookahead-32-puct",
        description="Second line: lookahead during training.",
        parent_lineage="connect4-L1",
    ))
    return store


def test_lineage_records_its_rubric_and_parent(lineage_store):
    lineage = lineage_store.get("connect4-L2")
    assert lineage.rubric_id == "r002-lookahead-32-puct"
    assert lineage.parent_lineage == "connect4-L1"
    assert lineage.restarted_from_scratch is True
    assert lineage_store.events("connect4-L2", "created")


def test_evolution_log_and_strength_trajectory(lineage_store):
    for cycle, (pid, score, promoted) in enumerate([
        ("p1", 0.05, True), ("p2", 0.04, False), ("p3", 0.12, True),
    ]):
        lineage_store.record("connect4-L2", "policy_added", policy_id=pid, cycle=cycle)
        lineage_store.record("connect4-L2", "cycle_completed", cycle=cycle, games=500)
        if promoted:
            lineage_store.record("connect4-L2", "promoted", policy_id=pid,
                                 cycle=cycle, score=score)

    trajectory = lineage_store.strength_trajectory("connect4-L2")
    assert [t["policy_id"] for t in trajectory] == ["p1", "p3"]
    assert [t["score"] for t in trajectory] == [0.05, 0.12]

    summary = lineage_store.summary("connect4-L2")
    assert summary["cycles_completed"] == 3
    assert summary["policies_added"] == 3
    assert summary["promotions"] == 2
    assert summary["best_promoted_score"] == 0.12
    assert summary["games_trained"] == 1500


def test_rubric_changes_are_visible(lineage_store):
    """Changing the recipe mid-line is allowed but must be recorded, since
    scores either side of it were produced differently."""
    lineage_store.record("connect4-L2", "rubric_changed",
                         rubric_id="r002-lookahead-400-puct",
                         reason="budget raised after exp-024")
    history = lineage_store.rubric_history("connect4-L2")
    assert [h["rubric_id"] for h in history] == [
        "r002-lookahead-32-puct", "r002-lookahead-400-puct"]
    assert "exp-024" in history[1]["reason"]


# ---------------------------------------------------------------------------
# spending rules: equal budget, different ways to use it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", SPEND_RULES)
def test_every_spend_rule_runs_and_respects_the_budget(rule):
    env = Connect4Env()
    env.reset()
    env.step(3)
    search = make_search(lambda obs: np.zeros(7), budget=120, spend_rule=rule, seed=0)
    result = search.run(env.obs())
    assert result.nodes_evaluated <= 120
    assert result.best_action in env.legal_actions()


def test_greedy_goes_deeper_than_uniform_at_the_same_budget():
    """The clearest demonstration that the spending rule matters: identical
    budget, very different shape of search."""
    env = Connect4Env()
    env.reset()
    for m in (3, 3, 4):
        env.step(m)

    def opinionated(obs):
        q = np.full(7, -0.5)
        q[2] = 0.8
        return q

    greedy = make_search(opinionated, budget=300, spend_rule="greedy").run(env.obs())
    uniform = make_search(opinionated, budget=300, spend_rule="uniform").run(env.obs())
    assert greedy.max_depth_reached > uniform.max_depth_reached


def test_unknown_spend_rule_is_rejected():
    with pytest.raises(ValueError, match="spend_rule"):
        make_search(lambda obs: np.zeros(7), budget=10, spend_rule="telepathy")


# ---------------------------------------------------------------------------
# lookahead inside training
# ---------------------------------------------------------------------------


def win_in_one_env():
    env = Connect4Env()
    env.reset()
    for m in (0, 6, 1, 6, 2, 6):
        env.step(m)
    assert 3 in env.winning_moves(1)
    return env


def test_a_training_budget_makes_an_untrained_learner_play_the_win():
    """With no training at all, 32 states of lookahead is enough to stop a
    fresh network from missing an immediate win."""
    trainer = DQNTrainer(6, 7, 7, TrainerConfig(seed=0, search_budget=32,
                                                spend_rule="puct",
                                                search_dirichlet=0.0))
    trainer.eps = 0.0
    assert trainer.select_action(win_in_one_env().obs(), [0, 1, 2, 3, 4, 5]) == 3


def test_no_budget_means_no_lookahead():
    trainer = DQNTrainer(6, 7, 7, TrainerConfig(seed=0))
    assert trainer.search is None
    values = trainer.decision_values(np.zeros((2, 6, 7), dtype=np.float32))
    assert values.shape == (7,)


def test_search_sees_updated_weights_not_a_stale_copy():
    """The search must evaluate with the network as it is now, or training
    would improve a network the search never consults."""
    trainer = DQNTrainer(6, 7, 7, TrainerConfig(seed=0, search_budget=16))
    before = trainer.decision_values(win_in_one_env().obs()).copy()
    with __import__("torch").no_grad():
        for p in trainer.net.parameters():
            p.mul_(0.0)
    after = trainer.decision_values(win_in_one_env().obs())
    assert not np.allclose(before, after)


def test_bounded_targets_stay_inside_the_reward_range():
    """The exp-022 fix. An inflated network produces inflated search values;
    clamping keeps the training target inside the range rewards span."""
    cfg = TrainerConfig(seed=0, batch_size=8, search_budget=16,
                        search_targets=True, bounded_targets=True,
                        store_opponent_transitions=True)
    trainer = DQNTrainer(6, 7, 7, cfg)
    with __import__("torch").no_grad():  # force wildly inflated ratings
        trainer.net.out.bias.add_(7.0)
    opp = RandomOpponent(seed=1)
    for _ in range(3):
        trainer.play_episode(Connect4Env(), opponent_q=None, opponent_act=opp.act)
    targets = np.array([t[2] for t in trainer.buffer.buffer])
    assert targets.size and np.all(np.abs(targets) <= 1.0 + 1e-9), (
        f"targets escaped the reward range: max |target| = {np.abs(targets).max():.3f}"
    )


def test_unbounded_targets_can_escape_the_range():
    """The converse, so the test above is known to be testing something."""
    cfg = TrainerConfig(seed=0, batch_size=8, search_budget=16,
                        search_targets=True, bounded_targets=False,
                        store_opponent_transitions=True)
    trainer = DQNTrainer(6, 7, 7, cfg)
    with __import__("torch").no_grad():
        trainer.net.out.bias.add_(7.0)
    opp = RandomOpponent(seed=1)
    for _ in range(3):
        trainer.play_episode(Connect4Env(), opponent_q=None, opponent_act=opp.act)
    targets = np.array([t[2] for t in trainer.buffer.buffer])
    assert np.abs(targets).max() > 1.0


def test_training_with_lookahead_runs_and_reports_its_cost():
    cfg = TrainerConfig(seed=0, batch_size=8, search_budget=24,
                        search_targets=True, bounded_targets=True)
    trainer = DQNTrainer(6, 7, 7, cfg)
    opp = RandomOpponent(seed=2)
    losses = []
    for _ in range(4):
        stats = trainer.play_episode(Connect4Env(), opponent_q=None, opponent_act=opp.act)
        losses.extend(stats.losses)
    assert losses and all(np.isfinite(x) for x in losses)
    assert trainer.search_moves > 0
    assert trainer.search_states_examined > 0
    assert trainer.search_deepest >= 1
