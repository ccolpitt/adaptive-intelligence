"""Search during training only (exp-021): lookahead picks self-play moves
(train_search_depth) and/or supplies distillation targets (search_targets).
Evaluation stays bare — nothing here touches the benchmark."""

import numpy as np

from harness.agents.dqn import DQNTrainer, TrainerConfig
from harness.scripted_opponents import RandomOpponent
from harness.tasks.connect4 import Connect4Env


def _win_in_one_env():
    env = Connect4Env(); env.reset()
    for m in (0, 6, 1, 6, 2, 6):  # X has 0,1,2 on the bottom row; X to move
        env.step(m)
    assert 3 in env.winning_moves(1)
    return env


def test_search_depth_makes_blind_learner_take_immediate_win():
    trainer = DQNTrainer(6, 7, 7, TrainerConfig(seed=0, train_search_depth=1))
    trainer.eps = 0.0
    env = _win_in_one_env()
    assert trainer.select_action(env.obs(), env.legal_actions()) == 3


def test_default_config_does_not_search():
    trainer = DQNTrainer(6, 7, 7, TrainerConfig(seed=0))
    vals = trainer.decision_values(np.zeros((2, 6, 7), dtype=np.float32))
    assert vals.shape == (7,)  # raw Q, no lookahead


def test_search_targets_are_supervised_and_bounded():
    cfg = TrainerConfig(seed=0, batch_size=8, train_search_depth=2, search_targets=True,
                        store_opponent_transitions=True)
    trainer = DQNTrainer(6, 7, 7, cfg)
    opp = RandomOpponent(seed=1)
    for _ in range(4):
        trainer.play_episode(Connect4Env(), opponent_q=None, opponent_act=opp.act)
    transitions = list(trainer.buffer.buffer)
    assert transitions
    assert all(t[5] for t in transitions), "search targets are stored as terminal labels"
    rewards = np.array([t[2] for t in transitions])
    assert np.all(np.isfinite(rewards))
    # Real-rule outcomes within the search horizon are exact: the game-winning
    # moves carry exactly +1 and the losing final moves exactly -gamma.
    assert np.any(np.isclose(rewards, 1.0))
    assert np.any(np.isclose(rewards, -0.99))
    # (Leaf values inherit the untrained net's raw Q scale early on — that is
    # expected and is what training shrinks.)


def test_search_target_of_winning_move_is_exactly_one():
    cfg = TrainerConfig(seed=0, train_search_depth=1, search_targets=True)
    trainer = DQNTrainer(6, 7, 7, cfg)
    env = _win_in_one_env()
    vals = trainer.decision_values(env.obs())
    assert np.isclose(vals[3], 1.0)


def test_training_runs_end_to_end_with_search():
    cfg = TrainerConfig(seed=0, batch_size=8, train_search_depth=2, search_targets=True)
    trainer = DQNTrainer(6, 7, 7, cfg)
    opp = RandomOpponent(seed=2)
    losses = []
    for _ in range(6):
        stats = trainer.play_episode(Connect4Env(), opponent_q=None, opponent_act=opp.act)
        losses.extend(stats.losses)
    assert losses and all(np.isfinite(x) for x in losses)
