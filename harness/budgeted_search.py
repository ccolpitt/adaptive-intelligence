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

    def __init__(self, q_function, n_actions: int, prior_temperature: float = 0.5,
                 uniform_priors: bool = False):
        self.q_function = q_function
        self._n_actions = n_actions
        self.prior_temperature = prior_temperature
        # When set, the network's ranking of moves is not used to decide where
        # to look — only its valuation of positions. Isolates the value of the
        # ranking itself.
        self.uniform_priors = uniform_priors

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
            priors[legal] = (np.full(len(legal), 1.0 / len(legal))
                             if self.uniform_priors
                             else _softmax(q[legal], self.prior_temperature))
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
    # Re-visiting a position already paid for is free, because the budget
    # buys looks by the NETWORK and a repeat needs none. But free is not
    # unlimited: in a position where every reachable line is already paid for
    # — a nearly full board, or a forced sequence — nothing would ever advance
    # the budget and the loop would not end. So the number of walks down the
    # tree is also capped, at this multiple of the budget. It is a termination
    # guard, not a second budget: it should rarely bind.
    max_simulations_per_budget: int = 20
    # Which statistic names the move once the search is done.
    #   "visits" — the most-attended move, as AlphaZero does. Steadier,
    #              because a move earns visits only by continuing to look
    #              good as it is examined further.
    #   "value"  — the highest backed-up value. Noisier at small budgets,
    #              where a mean can rest on one or two visits.
    # "value" is the default only because experiments already recorded used
    # it; new work should set "visits".
    select_by: str = "value"


class _Node:
    __slots__ = ("state", "depth", "visits", "value_sum", "children", "priors",
                 "legal", "terminal", "terminal_value", "confidence", "expanded",
                 "counted")

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
        # Has this position already been charged to the budget? A position is
        # paid for once per move, however often the search returns to it.
        self.counted = False

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class SearchResult:
    """What one move's thinking produced."""

    action_values: np.ndarray  # negamax value per action, ILLEGAL where unplayable
    visit_counts: np.ndarray  # how the budget was spent per root action
    root_value: float
    nodes_evaluated: int  # UNIQUE positions examined — the budget consumed
    max_depth_reached: int
    tree_size: int
    best_action: int  # by backed-up value
    cache_hits: int = 0  # repeats served without spending budget

    @property
    def best_action_by_visits(self) -> int:
        """The move the search attended to most.

        This is how AlphaZero picks its move — not the highest backed-up
        value (Silver et al., 2017: the move is drawn from the visit counts,
        with the temperature driven to zero at play time, which selects the
        most-visited move). The value appears only inside the selection rule
        while searching.

        Visit count is the steadier statistic. A move accumulates visits only
        by continuing to look good as it is examined more deeply, so it
        reflects sustained preference. A backed-up mean can rest on very few
        visits, or be carried by one fortunate deep line, and is therefore
        far noisier at small budgets.
        """
        if self.visit_counts.sum() <= 0:
            return self.best_action
        return int(np.argmax(self.visit_counts))

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
        self.total_cache_hits = 0
        self._cache: Dict[bytes, NodeEval] = {}
        self.cache_hits = 0

    # -- QFunction interface ------------------------------------------------

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Scores per move, so the search can stand in for a policy anywhere
        a policy is expected. Under "visits" the scores are the visit counts,
        so taking the largest selects the most-attended move; illegal moves
        stay excluded."""
        result = self.run(obs)
        if self.cfg.select_by == "visits" and result.visit_counts.sum() > 0:
            scores = result.visit_counts.astype(np.float64)
            scores[result.action_values <= ILLEGAL / 2] = ILLEGAL
            return scores
        return result.action_values

    # -- main loop ----------------------------------------------------------

    def run(self, obs: np.ndarray) -> SearchResult:
        n_actions = self.evaluator.n_actions
        root = _Node(self.model.root_state(obs), depth=0, confidence=1.0)
        nodes_evaluated = 0
        max_depth = 0
        tree_size = 1

        # Positions already looked at during THIS move. Two different move
        # orders often reach the same position, and looking at it twice is
        # both wasted compute and a budget charged twice for one position, so
        # a repeat is served from here and costs nothing. The cache is
        # discarded when the move is over, because the weights may change
        # between moves and a stale opinion would then be worse than none.
        self._cache: Dict[bytes, NodeEval] = {}
        self.cache_hits = 0

        # First unit of budget: look at the position we are actually in.
        nodes_evaluated += self._expand(root)
        if self.cfg.dirichlet_noise > 0 and root.legal and root.priors is not None:
            noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(root.legal))
            mix = self.cfg.dirichlet_noise
            for i, a in enumerate(root.legal):
                root.priors[a] = (1 - mix) * root.priors[a] + mix * noise[i]

        simulations = 0
        simulation_cap = max(1, self.cfg.node_budget
                             * self.cfg.max_simulations_per_budget)
        while (nodes_evaluated < self.cfg.node_budget and root.legal
               and simulations < simulation_cap):
            simulations += 1
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

            # EVALUATE: at most one unit of budget.
            #
            # The budget buys LOOKS AT POSITIONS BY THE NETWORK. A position
            # already looked at this move is free (the cache), and so is a
            # terminal position after the first time: the rules give its value
            # exactly, forever, with no network call. Re-visiting is not waste
            # — it is how the search expresses confidence, and a lopsided
            # visit count is exactly the training signal the move head needs
            # — so the right treatment is to stop CHARGING for it, not to
            # stop doing it.
            #
            # Charging every re-visit made the declared budget meaningless in
            # precisely the positions that decide games: a budget of 200 in a
            # win-in-one position bought 14 distinct positions and a search
            # two plies deep, because 187 units went on re-confirming the same
            # known win (R-008).
            if node.terminal:
                value = node.terminal_value
                if not node.counted:
                    node.counted = True
                    nodes_evaluated += 1
            elif not node.expanded:
                nodes_evaluated += self._expand(node)
                node.counted = True
                value = node.value_sum  # set by _expand as the raw estimate
                node.value_sum = 0.0  # cleared; the backup below records it
            else:
                # Depth-capped or no legal moves: reuse what we know, and
                # charge only the first time, for the same reason.
                value = node.mean_value
                if not node.counted:
                    node.counted = True
                    nodes_evaluated += 1

            max_depth = max(max_depth, node.depth)

            # BACKUP: negate at every level, discount per ply, and shrink by
            # how much the model is trusted at this depth.
            self._backup(path, value)

        return self._result(root, n_actions, nodes_evaluated, max_depth, tree_size)

    # -- pieces -------------------------------------------------------------

    def _expand(self, node: _Node) -> int:
        """Look at a position. Returns the budget consumed: 1 for a position
        not yet seen this move, 0 for one already looked at (served from the
        cache), so the budget counts UNIQUE positions examined."""
        node.legal = list(self.model.legal_actions(node.state))
        observation = self.model.observation(node.state)
        key = self._position_key(node.state, observation)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            node.priors = cached.priors
            node.value_sum = cached.value
            node.expanded = True
            return 0
        ev = self.evaluator.evaluate(observation, node.legal)
        self._cache[key] = ev
        node.priors = ev.priors
        node.value_sum = ev.value  # handed straight to the backup by the caller
        node.expanded = True
        return 1

    @staticmethod
    def _position_key(state, observation: np.ndarray) -> bytes:
        """Identifies a position for caching. Uses the canonical observation,
        which already encodes whose turn it is by whose pieces are in which
        plane, so two states that look identical to the network share a key."""
        return observation.tobytes()

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
        self.total_cache_hits += self.cache_hits
        # The cache is per-move: discard it so the next move cannot act on an
        # opinion formed by weights that have since changed.
        self._cache = {}
        return SearchResult(
            action_values=action_values,
            visit_counts=visit_counts,
            root_value=root.mean_value,
            nodes_evaluated=nodes_evaluated,
            max_depth_reached=max_depth,
            tree_size=tree_size,
            best_action=best_action,
            cache_hits=self.cache_hits,
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


# ---------------------------------------------------------------------------
# spending rules
# ---------------------------------------------------------------------------
#
# Given an equal budget of states to examine, these are different rules for
# deciding which states to spend it on. Because the budget is identical, the
# rules are directly comparable, and which one learns fastest is a fair
# question for an experiment.

SPEND_RULES = ("puct", "ucb", "greedy", "uniform")


def make_search(
    q_function,
    budget: int,
    spend_rule: str = "puct",
    n_actions: int = 7,
    rows: int = 6,
    cols: int = 7,
    c_puct: float = 1.4,
    prior_temperature: float = 0.5,
    gamma: float = 0.99,
    dirichlet_noise: float = 0.0,
    seed: int = 0,
    model_confidence_decay: float = 1.0,
    select_by: str = "value",
) -> BudgetedSearch:
    """Build a budgeted search that spends its budget by a named rule.

    puct    — follow the moves the network rates highly, with the pull of each
              move fading as it is examined. Concentrates fastest when the
              network has useful opinions.
    ucb     — ignore the network's ranking when deciding where to look, and
              choose by the best value found so far plus a bonus for the
              less-examined. Tests whether the network's move ranking is worth
              anything beyond its position values.
    greedy  — always extend the line that currently looks best. Deepest reach,
              and most easily fooled by one bad estimate.
    uniform — spread the budget evenly, which is what a fixed-depth search
              does. The control that isolates the value of choosing.
    """
    if spend_rule not in SPEND_RULES:
        raise ValueError(f"spend_rule must be one of {SPEND_RULES}, got {spend_rule!r}")

    exploration = {"puct": c_puct, "ucb": c_puct, "greedy": 0.0, "uniform": 50.0}[spend_rule]
    cfg = SearchConfig(
        node_budget=budget,
        c_puct=exploration,
        gamma=gamma,
        prior_temperature=prior_temperature,
        dirichlet_noise=dirichlet_noise,
        seed=seed,
        model_confidence_decay=model_confidence_decay,
        select_by=select_by,
    )
    evaluator = QNetEvaluator(
        q_function, n_actions, prior_temperature,
        # ucb and uniform deliberately discard the network's move ranking, so
        # that what is being compared is the spending rule and not the prior.
        uniform_priors=spend_rule in ("ucb", "uniform"),
    )
    return BudgetedSearch(evaluator, PerfectConnect4Model(rows, cols), cfg)
