"""A two-headed policy: one head proposes moves, the other judges positions.

The learner so far had a single set of outputs doing two jobs at once —
ranking the seven columns and, by the size of those numbers, also judging how
good the position was. Measurements across three lineages said that does not
work: value judgement improved steadily (positions a perfect player proves
are lost were correctly rated near -0.9) while move preference stayed poor
(about half of all forced blocks missed). Asking one number to be both the
ranking and the magnitude means an error in the magnitude corrupts the
ranking.

This splits them, as AlphaZero does (Silver et al., 2017):

  **The move head** outputs a probability for each column: "which moves are
  worth considering here". It is not a value. Its job during search is to say
  where to spend the budget, and its job at scoring time is to name a move.

  **The value head** outputs one number for the position, from the point of
  view of whoever is about to move: +1 means that player is winning, -1 that
  they are losing. It replaces playing the game out to the end in order to
  find out. It is squashed into [-1, 1] by construction, so it cannot drift
  outside the range the rewards actually span — the failure recorded in
  exp-022 and exp-024.

Neither head produces the value of a particular move. That number — how good
is it to play column 3 here — comes from the search, which tries the move and
backs up what it finds. The network supplies judgement and suggestion; the
search supplies consequence.

How they are trained (expert iteration): the search, guided by both heads, is
stronger than either head alone. So its conclusions become the targets. The
move head is trained toward where the search actually spent its attention;
the value head is trained toward how the game actually ended. The next
generation's instant answer is therefore as good as this generation's
considered one, and searching on top of that is better again. That loop is the
thing the single-headed learner never had.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import Environment


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PolicyValueNet(nn.Module):
    """Shared trunk, then a move head and a value head."""

    def __init__(self, rows: int, cols: int, n_actions: int, channels: int = 64,
                 hidden: int = 128, conv_layers: int = 3):
        super().__init__()
        layers: list = []
        in_ch = 2
        for _ in range(conv_layers):
            layers += [nn.Conv2d(in_ch, channels, 3, padding=1),
                       nn.BatchNorm2d(channels), nn.ReLU()]
            in_ch = channels
        self.trunk = nn.Sequential(*layers)
        flat = channels * rows * cols

        # Which moves are worth considering.
        self.policy_fc = nn.Linear(flat, hidden)
        self.policy_out = nn.Linear(hidden, n_actions)
        # How good this position is for whoever moves next.
        self.value_fc = nn.Linear(flat, hidden)
        self.value_out = nn.Linear(hidden, 1)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x).flatten(1)
        logits = self.policy_out(F.relu(self.policy_fc(h)))
        # Squashed: no position can be worth more than winning.
        value = torch.tanh(self.value_out(F.relu(self.value_fc(h)))).squeeze(-1)
        return logits, value


@dataclass
class PolicyValueConfig:
    seed: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    buffer_capacity: int = 100_000
    train_steps_per_game: int = 4
    channels: int = 64
    hidden: int = 128
    conv_layers: int = 3
    # How many positions the search may examine per move while training.
    # The move head is trained toward where the search spent its attention,
    # so this also sets how finely that target can be expressed: with seven
    # columns, a budget of 32 gives about four visits each and a very coarse
    # target, which is why AlphaZero used hundreds.
    search_budget: int = 200
    spend_rule: str = "puct"
    c_puct: float = 1.4
    gamma: float = 0.99
    # Exploration while generating training games: randomise the search's
    # starting opinion, and sample the move rather than always taking the
    # best, for the opening moves of each game.
    dirichlet_noise: float = 0.25
    dirichlet_alpha: float = 1.0
    sampling_moves: int = 12  # moves sampled from the visit counts, then greedy
    value_loss_weight: float = 1.0
    # Connect 4 is left-right symmetric, so every position teaches its mirror.
    mirror_augmentation: bool = True


# (observation, visit distribution over moves, final outcome for the mover)
Example = Tuple[np.ndarray, np.ndarray, float]


@dataclass
class GameStats:
    moves: int
    winner: Optional[int]
    learner_result: int = 0
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    total_losses: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    value_abs_means: List[float] = field(default_factory=list)
    # Mean absolute change in every weight, per training step. This is how
    # much the network actually MOVED, which is not the same as the gradient
    # norm: a large gradient that the optimiser scales down, or that is
    # cancelled by momentum, moves the weights very little. If this falls to
    # zero the network has stopped learning and further training on the same
    # material cannot help.
    weight_deltas: List[float] = field(default_factory=list)
    search_states: int = 0
    search_cache_hits: int = 0
    deepest: int = 0


class PolicyValueTrainer:
    """Self-play with search, training both heads on what the search found."""

    def __init__(self, rows: int, cols: int, n_actions: int,
                 config: PolicyValueConfig):
        self.cfg = config
        self.rows, self.cols, self.n_actions = rows, cols, n_actions
        seed_everything(config.seed)
        self.net = PolicyValueNet(rows, cols, n_actions, config.channels,
                                  config.hidden, config.conv_layers)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr,
                                          weight_decay=config.weight_decay)
        self.buffer: Deque[Example] = deque(maxlen=config.buffer_capacity)
        self.rng = np.random.default_rng(config.seed)
        self.games_done = 0
        self.unique_states_seen: set = set()
        self.search_states_examined = 0
        self.search_cache_hits = 0
        self.search_moves = 0
        self.search_deepest = 0
        self._search = None

    # -- inference ------------------------------------------------------

    def policy_value(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """One look: which moves are worth considering, and how good is this."""
        self.net.eval()
        with torch.no_grad():
            t = torch.from_numpy(obs).float().unsqueeze(0)
            logits, value = self.net(t)
            return logits.squeeze(0).numpy(), float(value.item())

    def q_function(self):
        """Move preference as an array, so the benchmark, the evaluator and
        the cards can use this policy exactly like the previous ones. These
        are the move head's scores, NOT values: they rank moves and nothing
        more. Scoring a policy means taking the best of them, which is the
        policy acting on its own with no search."""
        def q(obs: np.ndarray) -> np.ndarray:
            logits, _ = self.policy_value(obs)
            return logits
        return q

    def value_function(self):
        def v(obs: np.ndarray) -> float:
            return self.policy_value(obs)[1]
        return v

    @property
    def search(self):
        """The search the policy thinks with while training, guided by both
        heads: the move head says where to look, the value head says what was
        found."""
        if self._search is None:
            from ..budgeted_search import (
                BudgetedSearch, PerfectConnect4Model, PolicyValueEvaluator, SearchConfig,
            )

            self._search = BudgetedSearch(
                PolicyValueEvaluator(self.policy_value, self.n_actions),
                PerfectConnect4Model(self.rows, self.cols),
                SearchConfig(node_budget=self.cfg.search_budget,
                             c_puct=self.cfg.c_puct, gamma=self.cfg.gamma,
                             dirichlet_noise=self.cfg.dirichlet_noise,
                             dirichlet_alpha=self.cfg.dirichlet_alpha,
                             seed=self.cfg.seed,
                             select_by="visits"),
            )
        return self._search

    def load_weights_from(self, module: torch.nn.Module) -> None:
        self.net.load_state_dict(module.state_dict())

    # -- stopping and continuing ----------------------------------------
    #
    # Weights alone are not enough to continue a run. The optimiser carries
    # per-weight averages built up over thousands of steps, and the replay
    # buffer carries the games those steps learned from. Restoring only the
    # weights restarts both, which measurably undoes progress: R-007 records
    # a resumed lineage falling from 0.622 to 0.390 and needing 27 cycles —
    # 5,400 games — to climb back. So a checkpoint is everything the learner
    # needs to carry on as though it had never stopped.

    def checkpoint(self) -> Dict:
        """Everything needed to continue training exactly where this left
        off, as plain data."""
        return {
            "format": 1,
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            # Observations are stacked once rather than kept as a list of
            # arrays, which makes the file a few large tensors instead of
            # hundreds of thousands of small ones.
            "buffer": {
                "obs": np.stack([e[0] for e in self.buffer]) if self.buffer else None,
                "visits": np.stack([e[1] for e in self.buffer]) if self.buffer else None,
                "outcomes": np.array([e[2] for e in self.buffer], dtype=np.float32)
                if self.buffer else None,
            },
            "counters": {
                "games_done": self.games_done,
                "search_states_examined": self.search_states_examined,
                "search_cache_hits": self.search_cache_hits,
                "search_moves": self.search_moves,
                "search_deepest": self.search_deepest,
                "unique_states_seen": list(self.unique_states_seen),
            },
            # Both generators that drive self-play, so a continued run does
            # not replay the same games it already played.
            "rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "shape": [self.rows, self.cols, self.n_actions,
                      self.cfg.channels, self.cfg.hidden, self.cfg.conv_layers],
        }

    def load_checkpoint(self, state: Dict) -> None:
        """Continue from a checkpoint. Raises if it describes a different
        network, because silently training a differently-shaped policy would
        make the lineage incomparable with itself."""
        shape = [self.rows, self.cols, self.n_actions,
                 self.cfg.channels, self.cfg.hidden, self.cfg.conv_layers]
        if list(state.get("shape", shape)) != shape:
            raise ValueError(
                f"checkpoint is for a network of shape {state['shape']}, but "
                f"this trainer builds {shape}; a lineage cannot change shape "
                f"mid-run")
        self.net.load_state_dict(state["net"])
        self.optimizer.load_state_dict(state["optimizer"])
        buffer = state.get("buffer") or {}
        self.buffer.clear()
        if buffer.get("obs") is not None:
            for obs, visits, outcome in zip(buffer["obs"], buffer["visits"],
                                            buffer["outcomes"]):
                self.buffer.append((np.asarray(obs), np.asarray(visits),
                                    float(outcome)))
        counters = state.get("counters") or {}
        self.games_done = counters.get("games_done", 0)
        self.search_states_examined = counters.get("search_states_examined", 0)
        self.search_cache_hits = counters.get("search_cache_hits", 0)
        self.search_moves = counters.get("search_moves", 0)
        self.search_deepest = counters.get("search_deepest", 0)
        self.unique_states_seen = set(counters.get("unique_states_seen", []))
        if state.get("rng") is not None:
            self.rng.bit_generator.state = state["rng"]
        if state.get("torch_rng") is not None:
            torch.set_rng_state(state["torch_rng"].to(torch.uint8)
                                if hasattr(state["torch_rng"], "to")
                                else state["torch_rng"])
        # The search holds a reference to the live network through
        # self.policy_value, so it needs no reloading — but it caches nothing
        # across moves, and its own generator is seeded from the config.

    # -- self-play ------------------------------------------------------

    def play_game(self, env: Environment,
                  opponent_act: Optional[Callable] = None) -> GameStats:
        """One training game. The learner thinks with the search before each
        move; what the search concluded becomes the training target.

        With ``opponent_act`` the learner plays a fixed opponent and only its
        own positions are recorded. Otherwise it plays itself, and every
        position is recorded — in self-play both sides are the same policy, so
        both sides' searches are equally informative.
        """
        obs = env.reset()
        learner_is_p1 = self.games_done % 2 == 0
        pending: List[Tuple[np.ndarray, np.ndarray, int]] = []  # obs, visits, mover
        moves = 0
        stats_states = stats_hits = 0
        deepest = 0

        while True:
            mover = env.current_player
            learner_to_move = opponent_act is None or (mover == 1) == learner_is_p1
            legal = env.legal_actions()

            if learner_to_move:
                self.unique_states_seen.add(hash(obs.tobytes()))
                result = self.search.run(obs)
                stats_states += result.nodes_evaluated
                stats_hits += result.cache_hits
                deepest = max(deepest, result.max_depth_reached)
                self.search_moves += 1

                # Where the search spent its attention IS the move head's
                # target. Early in a game the move is sampled from it so games
                # differ; later the best is taken.
                temperature = 1.0 if moves < self.cfg.sampling_moves else 0.0
                visit_policy = result.visit_policy(temperature=1.0)
                pending.append((obs.copy(), visit_policy, mover))

                if temperature > 0 and visit_policy.sum() > 0:
                    action = int(self.rng.choice(len(visit_policy), p=visit_policy))
                    if action not in legal:  # numerical safety
                        action = result.best_action_by_visits
                else:
                    # The most-attended move, as AlphaZero selects. Not the
                    # highest backed-up value: a visit count reflects
                    # sustained preference across the search, while a mean
                    # can rest on one or two visits.
                    action = result.best_action_by_visits
            else:
                action = opponent_act(obs, legal)

            step = env.step(action)
            moves += 1
            obs = step.next_obs
            if step.done:
                break

        self.search_states_examined += stats_states
        self.search_cache_hits += stats_hits
        self.search_deepest = max(self.search_deepest, deepest)

        # The value head's target is how the game actually ended, seen from
        # each position's own mover. No estimate of an estimate: the outcome.
        winner = step.winner
        for obs_i, visits_i, mover_i in pending:
            if winner is None or winner == 0:
                outcome = 0.0
            else:
                outcome = 1.0 if mover_i == winner else -1.0
            self.buffer.append((obs_i, visits_i, outcome))
            if self.cfg.mirror_augmentation:
                self.buffer.append((np.flip(obs_i, axis=2).copy(),
                                    np.flip(visits_i).copy(), outcome))

        learner_side = 1 if learner_is_p1 else -1
        learner_result = 0
        if winner is not None and winner != 0:
            learner_result = 1 if winner == learner_side else -1

        self.games_done += 1
        stats = GameStats(moves=moves, winner=winner, learner_result=learner_result,
                          search_states=stats_states, search_cache_hits=stats_hits,
                          deepest=deepest)
        for _ in range(self.cfg.train_steps_per_game):
            step_losses = self._train_step()
            if step_losses is not None:
                (policy_loss, value_loss, total, grad_norm, value_abs,
                 weight_delta) = step_losses
                stats.policy_losses.append(policy_loss)
                stats.value_losses.append(value_loss)
                stats.total_losses.append(total)
                stats.grad_norms.append(grad_norm)
                stats.value_abs_means.append(value_abs)
                stats.weight_deltas.append(weight_delta)
        return stats

    # -- learning -------------------------------------------------------

    def _train_step(self) -> Optional[Tuple[float, float, float, float, float]]:
        if len(self.buffer) < self.cfg.batch_size:
            return None
        batch = random.sample(self.buffer, self.cfg.batch_size)
        obs = torch.from_numpy(np.stack([b[0] for b in batch])).float()
        target_policy = torch.from_numpy(np.stack([b[1] for b in batch])).float()
        target_value = torch.tensor([b[2] for b in batch], dtype=torch.float32)

        self.net.train()
        logits, value = self.net(obs)

        # Move head: match where the search spent its attention.
        policy_loss = -(target_policy * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        # Value head: match how the game actually ended.
        value_loss = F.mse_loss(value, target_value)
        total = policy_loss + self.cfg.value_loss_weight * value_loss

        self.optimizer.zero_grad()
        total.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)

        # How far the weights actually move, measured either side of the step.
        # Distinct from the gradient norm: the optimiser can scale a large
        # gradient down to almost nothing, so only this answers "is the
        # network still changing?".
        before = torch.cat([p.detach().reshape(-1).clone()
                            for p in self.net.parameters()])
        self.optimizer.step()
        after = torch.cat([p.detach().reshape(-1) for p in self.net.parameters()])
        weight_delta = float((after - before).abs().mean().item())

        return (float(policy_loss.item()), float(value_loss.item()),
                float(total.item()), float(grad_norm),
                float(value.detach().abs().mean().item()), weight_delta)

    # -- diagnostics ----------------------------------------------------

    def dormant_fraction(self, threshold: float = 0.01) -> Dict[str, float]:
        out = {}
        for name, param in self.net.named_parameters():
            if param.dim() > 1:
                out[name] = float((param.abs() < threshold).float().mean().item())
        return out

    # -- serialization --------------------------------------------------

    def scripted(self) -> torch.jit.ScriptModule:
        cpu_net = PolicyValueNet(self.rows, self.cols, self.n_actions,
                                 self.cfg.channels, self.cfg.hidden,
                                 self.cfg.conv_layers)
        cpu_net.load_state_dict({k: v.cpu() for k, v in self.net.state_dict().items()})
        cpu_net.eval()
        example = torch.zeros(1, 2, self.rows, self.cols, dtype=torch.float32)
        return torch.jit.trace(cpu_net, example)


def move_scores_from(module: torch.jit.ScriptModule):
    """Move preference from a saved two-headed policy, for scoring it alone."""
    def q(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits, _ = module(torch.from_numpy(obs).float().unsqueeze(0))
            return logits.squeeze(0).numpy()
    return q


def policy_value_from(module: torch.jit.ScriptModule):
    """Both heads from a saved policy, for searching with it."""
    def fn(obs: np.ndarray) -> Tuple[np.ndarray, float]:
        with torch.no_grad():
            logits, value = module(torch.from_numpy(obs).float().unsqueeze(0))
            return logits.squeeze(0).numpy(), float(value.item())
    return fn
