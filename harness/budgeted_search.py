"""Budget-limited, value-guided lookahead.

The budget is a number of STATES the policy may explore per move — not a
search depth. The policy decides where to spend it: a promising line gets
followed deep while a line its value function dislikes is abandoned after one
look. This is the shape asked for in the learning rules (docs/learning-rules.md):

    "a fixed budget for number of states to explore ... the policy can prune
     parts of the tree however it wants ... the value function informs pruning"

Depth-limited search spends its budget uniformly and cannot do that; a
depth-3 search examines ~7^3 states whether or not any of them matter.
At an equal number of states examined, a budgeted best-first search reaches
far deeper along the lines that decide the game.

Three properties make this reusable beyond Connect 4:

1. **The evaluator is an interface.** `QNetEvaluator` wraps the kind of
   network we have now (a value per column); `PolicyValueEvaluator` wraps an
   AlphaZero-style network (a move distribution plus one position value).
   The search is identical either way, so swapping the learner does not
   change the search.
2. **The model is an interface.** `PerfectModel` applies the real game rules
   — available for Connect 4 and free. For a task with no rulebook, a learned
   model implements the same two methods and reports a CONFIDENCE for each
   predicted transition. The search multiplies confidence along a line and
   discounts values by it, so an unreliable model is automatically searched
   shallowly. That is the honest handling of the simulation-to-reality gap:
   trust the model exactly as far as it has earned.
3. **It is a drop-in QFunction.** `search(obs) -> value per action`, so the
   benchmark, the evaluator, the cards, and interactive play accept it
   unchanged.

Selection is PUCT (Rosin 2011; Silver et al. 2017): each step down the tree
takes the action maximizing

    Q(a)  +  c_puct * prior(a) * sqrt(N_parent) / (1 + N(a))

where Q(a) is the negamax-backed value of that action and the second term
decays as an action is explored. The prior concentrates the search on moves
the network likes, which is what makes it pruning rather than enumeration.

Sign convention, applied consistently: every stored value is from the
perspective of the player **to move at that node**. Backing a value up one
level negates it. Entering a terminal state returns a reward for the player
who just moved, so it is stored negated and comes back out with the right
sign for the parent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

ILLEGAL = -1e9


# ---------------------------------------------------------------------------
# interfaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transition:
    """Result of applying an action inside the search's model."""

    state: object
    reward: float  # for the player who just moved
    done: bool
    confidence: float = 1.0  # 1.0 = exact (real rules); < 1.0 = learned model


class SearchModel(Protocol):
    """What the search needs in order to look ahead. A perfect rulebook and a
    learned world model satisfy the same interface."""

    def root_state(self, obs: np.ndarray) -> object: ...
    def legal_actions(self, state: object) -> List[int]: ...
    def observation(self, state: object) -> np.ndarray: ...
    def apply(self, state: object, action: int) -> Transition: ...


@dataclass(frozen=True)
class NodeEval:
    """One network opinion about a position: how good it is for the player to
    move, and how promising each action looks."""

    value: float
    priors: np.ndarray  # length n_actions, zero on illegal moves, sums to 1


class Evaluator(Protocol):
    def evaluate(self, obs: np.ndarray, legal: Sequence[int]) -> NodeEval: ...

    @property
    def n_actions(self) -> int: ...


# ---------------------------------------------------------------------------
# evaluators
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        out = np.zeros_like(x)
        out[int(np.argmax(x))] = 1.0
        return out
    z = (x - np.max(x)) / temperature
    e = np.exp(z)
    return e / e.sum()


class QNetEvaluator:
    """Adapts a Q-function (one value per action) to the search.

    The position's value is the best value available to the mover; the priors
    are a softmax over the action values, so the search follows the moves the
    network rates highly. Lets the existing networks be searched over without
    retraining.
    """

    def __init__(self, q_function, n_actions: int, prior_temperature: float = 0.5):
        self.q_function = q_function
        self._n_actions = n_actions
        self.prior_temperature = prior_temperature

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def evaluate(self, obs: np.ndarray, legal: Sequence[int]) -> NodeEval:
        q = np.asarray(self.q_function(obs), dtype=np.float64)
        q = np.nan_to_num(q, nan=0.0, posinf=1.0, neginf=-1.0)
        legal = list(legal)
        value = float(np.max(q[legal])) if legal else 0.0
        priors = np.zeros(self._n_actions, dtype=np.float64)
        if legal:
            priors[legal] = _softmax(q[legal], self.prior_temperature)
        return NodeEval(value=value, priors=priors)


class PolicyValueEvaluator:
    """Adapts an AlphaZero-style network: a move distribution and a single
    bounded value for the position."""

    def __init__(self, policy_value_fn, n_actions: int):
        self.policy_value_fn = policy_value_fn
        self._n_actions = n_actions

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def evaluate(self, obs: np.ndarray, legal: Sequence[int]) -> NodeEval:
        logits, value = self.policy_value_fn(obs)
        logits = np.asarray(logits, dtype=np.float64)
        logits = np.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        legal = list(legal)
        priors = np.zeros(self._n_actions, dtype=np.float64)
        if legal:
            priors[legal] = _softmax(logits[legal], 1.0)
        return NodeEval(value=float(np.clip(value, -1.0, 1.0)), priors=priors)


# ---------------------------------------------------------------------------
# configuration, tree, result
# ---------------------------------------------------------------------------


@dataclass
class SearchConfig:
    """The thinking budget and how it is spent.

    node_budget — states the policy may explore for one move. This is the
      compute currency; equal budgets make two policies comparable.
    c_puct — willingness to look at moves the network does not already like.
      Low: deep and narrow. High: broad and shallow.
    gamma — per-ply discount, matching the learner's.
    max_depth — optional hard ceiling, normally unset: the budget governs.
    model_confidence_decay — extra per-ply trust decay for a learned model
      (1.0 = no decay, correct for exact rules).
    dirichlet_noise / dirichlet_alpha — root-only randomization used during
      self-play so training games vary; always 0 at evaluation.
    """

    node_budget: int = 200
    c_puct: float = 1.4
    gamma: float = 0.99
    max_depth: Optional[int] = None
    prior_temperature: float = 0.5
    model_confidence_decay: float = 1.0
    dirichlet_noise: float = 0.0
    dirichlet_alpha: float = 1.0
    seed: int = 0


class _Node:
    __slots__ = ("state", "depth", "visits", "value_sum", "children", "priors",
                 "legal", "terminal", "terminal_value", "confidence", "expanded")

    def __init__(self, state: object, depth: int, confidence: float):
        self.state = state
        self.depth = depth
        self.confidence = confidence
        self.visits = 0
        self.value_sum = 0.0
        self.children: Dict[int, "_Node"] = {}
        self.priors: Optional[np.ndarray] = None
        self.legal: List[int] = []
        self.terminal = False
        self.terminal_value = 0.0  # from THIS node's mover's perspective
        self.expanded = False

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class SearchResult:
    """What one move's thinking produced."""

    action_values: np.ndarray  # negamax value per action, ILLEGAL where unplayable
    visit_counts: np.ndarray  # how the budget was spent per root action
    root_value: float
    nodes_evaluated: int  # budget actually consumed
    max_depth_reached: int
    tree_size: int
    best_action: int

    def visit_policy(self, temperature: float = 1.0) -> np.ndarray:
        """The search's move preference, as a distribution. This is the
        training target for a policy head: the network learns to predict what
        the search concluded, which is how deliberation becomes instinct."""
        counts = self.visit_counts.astype(np.float64)
        if counts.sum() <= 0:
            out = np.zeros_like(counts)
            out[self.best_action] = 1.0
            return out
        if temperature <= 0:
            out = np.zeros_like(counts)
            out[int(np.argmax(counts))] = 1.0
            return out
        scaled = counts ** (1.0 / temperature)
        return scaled / scaled.sum()


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------


class BudgetedSearch:
    """Best-first lookahead under a fixed budget of states explored."""

    def __init__(self, evaluator: Evaluator, model: SearchModel,
                 config: Optional[SearchConfig] = None):
        self.evaluator = evaluator
        self.model = model
        self.cfg = config or SearchConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        # cumulative diagnostics across all moves this object has thought about
        self.total_nodes_evaluated = 0
        self.total_moves = 0
        self.deepest_reached = 0

    # -- QFunction interface ------------------------------------------------

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.run(obs).action_values

    # -- main loop ----------------------------------------------------------

    def run(self, obs: np.ndarray) -> SearchResult:
        n_actions = self.evaluator.n_actions
        root = _Node(self.model.root_state(obs), depth=0, confidence=1.0)
        nodes_evaluated = 0
        max_depth = 0
        tree_size = 1

        # First unit of budget: look at the position we are actually in.
        nodes_evaluated += self._expand(root)
        if self.cfg.dirichlet_noise > 0 and root.legal and root.priors is not None:
            noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(root.legal))
            mix = self.cfg.dirichlet_noise
            for i, a in enumerate(root.legal):
                root.priors[a] = (1 - mix) * root.priors[a] + mix * noise[i]

        while nodes_evaluated < self.cfg.node_budget and root.legal:
            # SELECT: walk down by PUCT to a node we have not looked at yet.
            node = root
            path: List[_Node] = [root]
            while node.expanded and not node.terminal:
                depth_capped = (self.cfg.max_depth is not None
                                and node.depth >= self.cfg.max_depth)
                if depth_capped:
                    break
                action = self._select_action(node)
                if action is None:
                    break
                child = node.children.get(action)
                if child is None:
                    child = self._make_child(node, action)
                    node.children[action] = child
                    tree_size += 1
                node = child
                path.append(node)

            # EVALUATE: one unit of budget. Terminal states are exact and need
            # no network call, but still count as a state explored.
            if node.terminal:
                value = node.terminal_value
                nodes_evaluated += 1
            elif not node.expanded:
                nodes_evaluated += self._expand(node)
                value = node.value_sum  # set by _expand as the raw estimate
                node.value_sum = 0.0  # cleared; the backup below records it
            else:
                # Depth-capped or no legal moves: reuse what we know.
                value = node.mean_value
                nodes_evaluated += 1

            max_depth = max(max_depth, node.depth)

            # BACKUP: negate at every level, discount per ply, and shrink by
            # how much the model is trusted at this depth.
            self._backup(path, value)

        return self._result(root, n_actions, nodes_evaluated, max_depth, tree_size)

    # -- pieces -------------------------------------------------------------

    def _expand(self, node: _Node) -> int:
        """Look at a position: ask the network, record priors. Returns the
        budget consumed (1)."""
        node.legal = list(self.model.legal_actions(node.state))
        ev = self.evaluator.evaluate(self.model.observation(node.state), node.legal)
        node.priors = ev.priors
        node.value_sum = ev.value  # handed straight to the backup by the caller
        node.expanded = True
        return 1

    def _make_child(self, node: _Node, action: int) -> _Node:
        t = self.model.apply(node.state, action)
        confidence = node.confidence * t.confidence * self.cfg.model_confidence_decay
        child = _Node(t.state, depth=node.depth + 1, confidence=confidence)
        if t.done:
            child.terminal = True
            child.expanded = True
            # Reward is for the player who just moved (the parent's mover).
            # Stored negated so one backup negation yields the parent's view.
            child.terminal_value = -float(t.reward)
            child.legal = []
        return child

    def _select_action(self, node: _Node) -> Optional[int]:
        if not node.legal or node.priors is None:
            return None
        sqrt_parent = math.sqrt(max(node.visits, 1))
        best_score, best_action = -math.inf, None
        for a in node.legal:
            child = node.children.get(a)
            if child is None or child.visits == 0:
                q = 0.0
                child_visits = 0
            else:
                # Child values are from the opponent's perspective.
                q = -self.cfg.gamma * child.mean_value
                child_visits = child.visits
            u = self.cfg.c_puct * float(node.priors[a]) * sqrt_parent / (1 + child_visits)
            score = q + u
            if score > best_score:
                best_score, best_action = score, a
        return best_action

    def _backup(self, path: List[_Node], leaf_value: float) -> None:
        value = leaf_value
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value * node.confidence
            value = -self.cfg.gamma * value

    def _result(self, root: _Node, n_actions: int, nodes_evaluated: int,
                max_depth: int, tree_size: int) -> SearchResult:
        action_values = np.full(n_actions, ILLEGAL, dtype=np.float64)
        visit_counts = np.zeros(n_actions, dtype=np.int64)
        for a in root.legal:
            child = root.children.get(a)
            if child is None:
                # Never looked at: fall back to the network's prior opinion.
                action_values[a] = -1.0 + float(root.priors[a]) if root.priors is not None else -1.0
                continue
            visit_counts[a] = child.visits
            if child.terminal:
                action_values[a] = -child.terminal_value
            elif child.visits:
                action_values[a] = -self.cfg.gamma * child.mean_value
            else:
                action_values[a] = -1.0 + float(root.priors[a]) if root.priors is not None else -1.0
        best_action = int(np.argmax(action_values))

        self.total_nodes_evaluated += nodes_evaluated
        self.total_moves += 1
        self.deepest_reached = max(self.deepest_reached, max_depth)
        return SearchResult(
            action_values=action_values,
            visit_counts=visit_counts,
            root_value=root.mean_value,
            nodes_evaluated=nodes_evaluated,
            max_depth_reached=max_depth,
            tree_size=tree_size,
            best_action=best_action,
        )


# ---------------------------------------------------------------------------
# Connect 4 model (exact rules — confidence 1.0)
# ---------------------------------------------------------------------------


class PerfectConnect4Model:
    """The real rules as a search model. Confidence is always 1.0: there is
    no simulation-to-reality gap when the simulation IS the rules."""

    def __init__(self, rows: int = 6, cols: int = 7):
        self.rows, self.cols = rows, cols

    def root_state(self, obs: np.ndarray) -> Tuple[np.ndarray, int]:
        # Canonical obs: plane 0 = mover's pieces, plane 1 = opponent's.
        board = (obs[0] - obs[1]).astype(np.int8)
        return (board, 1)

    def legal_actions(self, state) -> List[int]:
        board, _ = state
        return [c for c in range(self.cols) if board[0, c] == 0]

    def observation(self, state) -> np.ndarray:
        board, player = state
        out = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        out[0] = board == player
        out[1] = board == -player
        return out

    def apply(self, state, action: int) -> Transition:
        from .tasks.connect4 import Connect4Env

        board, player = state
        env = Connect4Env(rows=self.rows, cols=self.cols)
        env.board = board.copy()
        env.current_player = player
        env.done = False
        env.winner = None
        result = env.step(action)
        return Transition(
            state=(env.board, env.current_player),
            reward=float(result.reward),
            done=bool(result.done),
            confidence=1.0,
        )


def budgeted_q(q_function, node_budget: int, n_actions: int = 7,
               **kwargs) -> BudgetedSearch:
    """Wrap an existing Q-function in budgeted search. Drop-in QFunction."""
    cfg = SearchConfig(node_budget=node_budget, **kwargs)
    return BudgetedSearch(
        QNetEvaluator(q_function, n_actions, cfg.prior_temperature),
        PerfectConnect4Model(),
        cfg,
    )
