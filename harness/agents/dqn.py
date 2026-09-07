"""DQN trainer — the dumbest complete learner that closed the loop in
connect4-rl, ported with three deliberate simplifications and one fix:

- Uniform replay instead of prioritized (dumb baseline; PER is a registered
  Phase-later experiment, not a Phase 0 assumption).
- No BatchNorm/Dropout: deterministic given a seed, so the reproducibility
  tests can demand bit-identical runs.
- The negamax Bellman target is kept: next_obs is the OPPONENT's perspective,
  so target = r - gamma * max_a' Q(next_obs, a').
- FIX vs connect4-rl: loss attribution is explicit. The episode collector
  buffers each game's transitions and, at game end, writes reward -1 and
  done=True onto the loser's final transition — no post-hoc buffer patching.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..interfaces import Environment, QFunction


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def shaping_delta(my_wins, opp_wins, action: int, penalty: float) -> float:
    """Tactical shaping (exp-011): penalty for the two decision-critical
    blunders. ``my_wins``/``opp_wins`` are the columns that would win
    immediately for the mover / the opponent, computed BEFORE the move.

    - Mover had a winning move and did not take it: -penalty (a foregone
      win otherwise produces no signal at all).
    - Opponent has an immediate win and the mover did not block it:
      -penalty, delivered NOW rather than only via the -1 at game end
      (and delivered even if the opponent then misses the win).
    Both can apply to one move. Taking a win already earns the env's +1.
    """
    delta = 0.0
    if my_wins and action not in my_wins:
        delta -= penalty
    if opp_wins and action not in opp_wins:
        delta -= penalty
    return delta


class QNet(nn.Module):
    """Small conv net over (2, R, C) canonical boards -> Q per column."""

    def __init__(
        self,
        rows: int,
        cols: int,
        n_actions: int,
        channels: int = 32,
        hidden: int = 128,
        conv_layers: int = 2,
    ):
        super().__init__()
        layers: list = []
        in_ch = 2
        for _ in range(conv_layers):
            layers += [nn.Conv2d(in_ch, channels, kernel_size=3, padding=1), nn.LeakyReLU(0.01)]
            in_ch = channels
        self.conv = nn.Sequential(*layers)
        self.fc1 = nn.Linear(channels * rows * cols, hidden)
        self.act = nn.LeakyReLU(0.01)
        self.out = nn.Linear(hidden, n_actions)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        h = self.act(self.fc1(h))
        return self.out(h)


# transition: (obs, action, reward, next_obs, next_legal_mask, done)
Transition = Tuple[np.ndarray, int, float, np.ndarray, np.ndarray, bool]


class ReplayBuffer:
    """Uniform replay, optionally guaranteeing a fraction of each batch is
    terminal transitions (``terminal_fraction`` > 0). Sparse +-1 rewards only
    exist at terminals; uniform sampling starves the value function of them —
    the connect4-rl precursor diagnosed this and used a 30% terminal quota."""

    def __init__(self, capacity: int, seed: int, terminal_fraction: float = 0.0):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.terminals: Deque[Transition] = deque(maxlen=capacity)
        self.terminal_fraction = terminal_fraction
        self.rng = random.Random(seed)

    def add(self, t: Transition) -> None:
        self.buffer.append(t)
        if t[5]:
            self.terminals.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        k = 0
        if self.terminal_fraction > 0 and self.terminals:
            k = min(int(round(batch_size * self.terminal_fraction)), len(self.terminals))
        rest = self.rng.sample(self.buffer, batch_size - k)
        if k:
            rest = rest + self.rng.sample(self.terminals, k)
        return rest

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class TrainerConfig:
    seed: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 64
    buffer_capacity: int = 20000
    train_steps_per_episode: int = 2
    target_update_every: int = 50  # episodes
    eps_start: float = 0.5
    eps_end: float = 0.1
    eps_decay: float = 0.999
    channels: int = 32
    hidden: int = 128
    # Upgrades, each default-off (the dumb baseline) and flipped only inside
    # a registered experiment, one at a time:
    store_opponent_transitions: bool = False  # exp-004: off-policy data from both seats
    mirror_augmentation: bool = False  # exp-005: left-right symmetry, 2x data
    terminal_fraction: float = 0.0  # exp-007: terminal quota per batch (precursor used 0.3)
    conv_layers: int = 2  # exp-010+: network depth
    # Tactical shaping (exp-011): immediate penalty on the two decision-
    # critical blunders — skipping your own winning move, and failing to
    # block the opponent's. Training-signal-only: the task reward, the
    # benchmark, and all evaluation stay pure win/lose.
    tactical_shaping: bool = False
    shaping_penalty: float = 0.5


@dataclass
class EpisodeStats:
    moves: int
    winner: Optional[int]
    losses: List[float] = field(default_factory=list)


class DQNTrainer:
    """Owns the learning apparatus (net, target net, buffer, optimizer).
    The Policy artifact is only the resulting Q-function — the trainer is
    never archived."""

    def __init__(self, rows: int, cols: int, n_actions: int, config: TrainerConfig):
        self.cfg = config
        self.rows, self.cols, self.n_actions = rows, cols, n_actions
        seed_everything(config.seed)
        self.net = QNet(rows, cols, n_actions, config.channels, config.hidden, config.conv_layers)
        self.target = QNet(rows, cols, n_actions, config.channels, config.hidden, config.conv_layers)
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.buffer = ReplayBuffer(
            config.buffer_capacity, config.seed, config.terminal_fraction
        )
        self.eps = config.eps_start
        self.episodes_done = 0
        self.rng = np.random.default_rng(config.seed)

    # -- inference ------------------------------------------------------

    def q_function(self) -> QFunction:
        """Deterministic eval-mode Q-function view of the current net."""

        def q(obs: np.ndarray) -> np.ndarray:
            self.net.eval()
            with torch.no_grad():
                t = torch.from_numpy(obs).float().unsqueeze(0)
                return self.net(t).squeeze(0).numpy()

        return q

    def load_weights_from(self, module: torch.nn.Module) -> None:
        self.net.load_state_dict(module.state_dict())
        self.target.load_state_dict(module.state_dict())

    def select_action(self, obs: np.ndarray, legal: List[int]) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.choice(legal))
        q = self.q_function()(obs)
        masked = np.full(self.n_actions, -np.inf, dtype=np.float64)
        masked[legal] = q[legal]
        return int(np.argmax(masked))

    # -- experience -----------------------------------------------------

    def play_episode(
        self,
        env: Environment,
        opponent_q: Optional[QFunction],
        opponent_act: Optional[Callable[[np.ndarray, List[int]], int]] = None,
    ) -> EpisodeStats:
        """One self-play game. The learner plays epsilon-greedy; the opponent
        is either a frozen Q-function played greedily (champion) or a
        scripted ``opponent_act``. Learner's side alternates by episode.

        By default only the learner's own transitions are stored (dumb
        baseline). With ``store_opponent_transitions`` the opponent's moves
        are stored too — valid off-policy data: a transition is a fact about
        the game regardless of who chose the move, and the Bellman target is
        computed by OUR net (exp-004).
        Loss attribution (the connect4-rl post-hoc-patch fix): at game end
        the LOSING side's final transition is rewritten to reward -1,
        done True, whichever side that is, before entering the buffer.
        """
        learner_is_p1 = self.episodes_done % 2 == 0
        obs = env.reset()
        pending: List[Transition] = []
        movers: List[int] = []  # +1 / -1, aligned with pending
        moves = 0

        def legal_mask() -> np.ndarray:
            m = np.zeros(self.n_actions, dtype=np.float32)
            m[env.legal_actions()] = 1.0
            return m

        while True:
            legal = env.legal_actions()
            mover = env.current_player
            learner_to_move = (mover == 1) == learner_is_p1
            will_store = learner_to_move or self.cfg.store_opponent_transitions
            my_wins = opp_wins = None
            if self.cfg.tactical_shaping and will_store and hasattr(env, "winning_moves"):
                my_wins = env.winning_moves(mover)
                opp_wins = env.winning_moves(-mover)
            if learner_to_move:
                action = self.select_action(obs, legal)
            elif opponent_act is not None:
                action = opponent_act(obs, legal)
            else:
                q = opponent_q(obs)
                masked = np.full(self.n_actions, -np.inf, dtype=np.float64)
                masked[legal] = q[legal]
                action = int(np.argmax(masked))
            result = env.step(action)
            if will_store:
                reward = result.reward
                if my_wins is not None:
                    reward += shaping_delta(my_wins, opp_wins, action, self.cfg.shaping_penalty)
                pending.append(
                    (obs, action, reward, result.next_obs, legal_mask(), result.done)
                )
                movers.append(mover)
            moves += 1
            obs = result.next_obs
            if result.done:
                break

        # Explicit loss attribution: rewrite the loser's final transition.
        if result.winner is not None and result.winner != 0:
            loser = -result.winner
            for i in range(len(pending) - 1, -1, -1):
                if movers[i] == loser:
                    o, a, _, no, m, _ = pending[i]
                    pending[i] = (o, a, -1.0, no, m, True)
                    break

        for t in pending:
            self.buffer.add(t)
            if self.cfg.mirror_augmentation:
                self.buffer.add(self._mirror(t))

        self.episodes_done += 1
        self.eps = max(self.cfg.eps_end, self.eps * self.cfg.eps_decay)

        stats = EpisodeStats(moves=moves, winner=result.winner)
        for _ in range(self.cfg.train_steps_per_episode):
            loss = self._train_step()
            if loss is not None:
                stats.losses.append(loss)
        if self.episodes_done % self.cfg.target_update_every == 0:
            self.target.load_state_dict(self.net.state_dict())
        return stats

    def _mirror(self, t: Transition) -> Transition:
        """Left-right reflection: Connect 4 is symmetric, so every transition
        teaches its mirror for free (exp-005)."""
        o, a, r, no, m, d = t
        return (
            np.flip(o, axis=2).copy(),
            self.n_actions - 1 - a,
            r,
            np.flip(no, axis=2).copy(),
            np.flip(m).copy(),
            d,
        )

    # -- learning -------------------------------------------------------

    def _train_step(self) -> Optional[float]:
        if len(self.buffer) < self.cfg.batch_size:
            return None
        batch = self.buffer.sample(self.cfg.batch_size)
        obs = torch.from_numpy(np.stack([t[0] for t in batch])).float()
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        next_obs = torch.from_numpy(np.stack([t[3] for t in batch])).float()
        masks = torch.from_numpy(np.stack([t[4] for t in batch])).float()
        dones = torch.tensor([float(t[5]) for t in batch], dtype=torch.float32)

        self.net.train()
        q = self.net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_obs)
            next_q = next_q.masked_fill(masks == 0, -1e9)
            next_max = next_q.max(dim=1).values
        # Negamax: next_obs is the opponent's perspective, hence the minus.
        target = rewards - self.cfg.gamma * next_max * (1.0 - dones)
        loss = nn.functional.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())

    # -- diagnostics ----------------------------------------------------

    def dormant_neuron_fraction(self, threshold: float = 0.01) -> Dict[str, float]:
        """Fraction of near-zero weights per layer — Phase 4 plasticity
        telemetry, collected from day one per the roadmap."""
        out = {}
        for name, p in self.net.named_parameters():
            if p.dim() > 1:
                out[name] = float((p.abs() < threshold).float().mean().item())
        return out

    # -- serialization --------------------------------------------------

    def scripted(self) -> torch.jit.ScriptModule:
        cpu_net = QNet(
            self.rows, self.cols, self.n_actions,
            self.cfg.channels, self.cfg.hidden, self.cfg.conv_layers,
        )
        cpu_net.load_state_dict({k: v.cpu() for k, v in self.net.state_dict().items()})
        cpu_net.eval()
        example = torch.zeros(1, 2, self.rows, self.cols, dtype=torch.float32)
        return torch.jit.trace(cpu_net, example)
