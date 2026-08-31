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


class QNet(nn.Module):
    """Small conv net over (2, R, C) canonical boards -> Q per column."""

    def __init__(self, rows: int, cols: int, n_actions: int, channels: int = 32, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
        )
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
    def __init__(self, capacity: int, seed: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, t: Transition) -> None:
        self.buffer.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return self.rng.sample(self.buffer, batch_size)

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
        self.net = QNet(rows, cols, n_actions, config.channels, config.hidden)
        self.target = QNet(rows, cols, n_actions, config.channels, config.hidden)
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.buffer = ReplayBuffer(config.buffer_capacity, config.seed)
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

        Both players' transitions from the LEARNER's net's point of view are
        not stored — only the learner's own transitions are (dumb baseline;
        symmetric/both-player storage is a registered later experiment).
        Loss attribution: if the opponent wins, the learner's last transition
        is rewritten to reward -1, done True, before entering the buffer.
        """
        learner_is_p1 = self.episodes_done % 2 == 0
        obs = env.reset()
        pending: List[Transition] = []
        moves = 0

        def legal_mask() -> np.ndarray:
            m = np.zeros(self.n_actions, dtype=np.float32)
            m[env.legal_actions()] = 1.0
            return m

        while True:
            legal = env.legal_actions()
            learner_to_move = (env.current_player == 1) == learner_is_p1
            if learner_to_move:
                action = self.select_action(obs, legal)
                result = env.step(action)
                pending.append(
                    (obs, action, result.reward, result.next_obs, legal_mask(), result.done)
                )
            else:
                if opponent_act is not None:
                    action = opponent_act(obs, legal)
                else:
                    q = opponent_q(obs)
                    masked = np.full(self.n_actions, -np.inf, dtype=np.float64)
                    masked[legal] = q[legal]
                    action = int(np.argmax(masked))
                result = env.step(action)
            moves += 1
            obs = result.next_obs
            if result.done:
                break

        # Explicit loss attribution (the connect4-rl post-hoc-patch fix):
        learner_player = 1 if learner_is_p1 else -1
        if result.winner is not None and result.winner != 0 and result.winner != learner_player:
            if pending:
                o, a, _, no, m, _ = pending[-1]
                pending[-1] = (o, a, -1.0, no, m, True)

        for t in pending:
            self.buffer.add(t)

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
        cpu_net = QNet(self.rows, self.cols, self.n_actions, self.cfg.channels, self.cfg.hidden)
        cpu_net.load_state_dict({k: v.cpu() for k, v in self.net.state_dict().items()})
        cpu_net.eval()
        example = torch.zeros(1, 2, self.rows, self.cols, dtype=torch.float32)
        return torch.jit.trace(cpu_net, example)
