"""Connect 4 cycle renderer, version 1.

Draws one training cycle. Every panel answers a question that has come up
while diagnosing this project:

  1. Is the error it is trained on going down?
  2. Is the size of the weight corrections stable?
  3. Are its value estimates staying inside the range the rewards span?
  4. Is it seeing new positions, or grinding the same ones?
  5. Is it winning more of its training games as the cycle goes on?
  6. How was the lookahead budget actually used?
  7. Does it know a winning move when it sees one, and a losing one?
  8. Is any part of the network dying or blowing up?
  9. What did the cycle conclude — score, promotion, and what to fix next.

Never edited once records exist that name it. Changes go into version 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

VERSION = "connect4_v1"
TASK = "connect4"

# What this renderer expects to find in record.series. A record may carry
# more; anything missing is drawn as "not measured".
EXPECTED_SERIES = (
    "loss",            # training error per game
    "grad_norm",       # size of the weight correction per game
    "q_mean_abs",      # average magnitude of value estimates
    "result",          # +1 win / 0 draw / -1 loss, per training game
    "unique_states",   # cumulative distinct positions seen
    "epsilon",         # chance of a random move
    "search_states",   # states examined by lookahead, per game
)


def _smooth(values: Sequence[Optional[float]], window: int = 25
            ) -> Tuple[List[int], List[float]]:
    points = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(points) < window:
        return [p[0] for p in points], [float(p[1]) for p in points]
    idx = [p[0] for p in points]
    vals = [float(p[1]) for p in points]
    out = [sum(vals[i:i + window]) / window for i in range(len(vals) - window + 1)]
    return idx[window - 1:], out


def render_cycle(record, out_dir: Union[str, Path, None] = None, show: bool = False):
    """Draw the cycle. Returns the path written, or the figure if not saving."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = record.series or {}

    def get(name: str) -> List:
        return series.get(name, []) or []

    fig, axes = plt.subplots(3, 3, figsize=(19, 12))
    promoted = ("PROMOTED" if record.promoted
                else ("parity" if record.parity else "rejected"))
    score = record.evaluation.get("score")
    fig.suptitle(
        f"{record.lineage_id}  cycle {record.cycle}  —  {promoted}"
        f"{f'   score {score:.3f}' if isinstance(score, (int, float)) else ''}"
        f"   [{record.rubric_id}]",
        fontsize=14, fontweight="bold",
    )

    def blank(ax, title, note="not measured in this cycle"):
        ax.set_title(title)
        ax.text(0.5, 0.5, note, ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=9)

    # 1: training error
    ax = axes[0, 0]
    loss = get("loss")
    if any(v is not None for v in loss):
        ax.plot([i for i, v in enumerate(loss) if v is not None],
                [v for v in loss if v is not None], alpha=0.25, color="tab:blue")
        sx, sy = _smooth(loss)
        ax.plot(sx, sy, color="tab:blue", label="smoothed")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.set_title("Training error per game (should fall)")
    else:
        blank(ax, "Training error per game")
    ax.set_xlabel("training game")

    # 2: size of weight corrections
    ax = axes[0, 1]
    grads = get("grad_norm")
    if any(v is not None for v in grads):
        sx, sy = _smooth(grads)
        ax.plot(sx, sy, color="tab:purple")
        ax.set_yscale("log")
        ax.set_title("Weight-correction size (steady is healthy)")
    else:
        blank(ax, "Weight-correction size")
    ax.set_xlabel("training game")

    # 3: are value estimates inside the range rewards span?
    ax = axes[0, 2]
    qs = get("q_mean_abs")
    if any(v is not None for v in qs):
        sx, sy = _smooth(qs)
        ax.plot(sx, sy, color="tab:red", label="average magnitude")
        ax.axhline(1.0, color="black", linestyle=":", label="limit of the reward scale")
        ax.legend(fontsize=8)
        ax.set_title("Value estimates (must stay under 1)")
    else:
        blank(ax, "Value estimates")
    ax.set_xlabel("training game")

    # 4: exploration coverage
    ax = axes[1, 0]
    states = get("unique_states")
    if states:
        ax.plot(states, color="tab:green")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_title("Distinct positions seen (cumulative)")
    else:
        blank(ax, "Distinct positions seen")
    ax.set_xlabel("training game")

    # 5: winning more of its own games?
    ax = axes[1, 1]
    results = get("result")
    if results:
        scores = [(r + 1) / 2 for r in results]  # 1 win, 0.5 draw, 0 loss
        window = min(100, max(len(scores) // 5, 2))
        sx, sy = _smooth(scores, window)
        ax.plot(sx, sy, color="tab:olive")
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="even")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.set_title(f"Win rate in training games (rolling {window})")
    else:
        blank(ax, "Win rate in training games")
    ax.set_xlabel("training game")

    # 6: how the lookahead budget was used
    ax = axes[1, 2]
    if record.search_budget > 0:
        spent = get("search_states")
        if any(v is not None for v in spent):
            sx, sy = _smooth(spent)
            ax.plot(sx, sy, color="tab:cyan", label="states examined per move")
        ax.axhline(record.search_budget, color="black", linestyle=":",
                   label=f"budget ({record.search_budget})")
        ax.legend(fontsize=8)
        ax.set_title(f"Lookahead use — deepest {record.search_deepest_ply} moves ahead")
        ax.set_xlabel("training game")
    else:
        blank(ax, "Lookahead use", "this cycle trained without lookahead")

    # 7: does it recognise decided positions?
    ax = axes[2, 0]
    probes = record.probes or {}
    labels, values, colors, targets = [], [], [], []
    for key, label, target in (
        ("q_of_game_winning_moves", "a move that wins", 1.0),
        ("q_at_solver_proven_wins", "a position proven won", 1.0),
        ("q_of_not_blocking", "failing to block", -1.0),
        ("q_at_solver_proven_losses", "a position proven lost", -1.0),
    ):
        if isinstance(probes.get(key), (int, float)):
            labels.append(label)
            values.append(probes[key])
            targets.append(target)
            colors.append("tab:green" if target > 0 else "tab:red")
    if labels:
        y = range(len(labels))
        ax.barh(list(y), values, color=colors, alpha=0.75)
        for i, t in enumerate(targets):
            ax.plot([t], [i], marker="|", markersize=18, color="black")
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(-1.4, 1.4)
        ax.axvline(0, color="gray", alpha=0.4)
        ax.set_title("Value of decided positions (bars should reach the marks)")
    else:
        blank(ax, "Value of decided positions")

    # 8: network health
    ax = axes[2, 1]
    layers = probes.get("weight_distribution") or {}
    if layers:
        names = list(layers)
        dead = [layers[n].get("dead_fraction", 0) for n in names]
        big = [layers[n].get("exploding_fraction", 0) for n in names]
        x = range(len(names))
        ax.bar([i - 0.2 for i in x], dead, width=0.4, label="near zero", color="black")
        ax.bar([i + 0.2 for i in x], big, width=0.4, label="very large", color="tab:red")
        ax.set_xticks(list(x))
        ax.set_xticklabels([n.replace(".weight", "") for n in names],
                           rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.set_title("Weights by layer (dying or blowing up)")
    else:
        blank(ax, "Weights by layer")

    # 9: what the cycle concluded
    ax = axes[2, 2]
    ax.axis("off")
    lines = [
        f"games this cycle: {record.games_trained:,}",
        f"games in lineage: {record.cumulative_games:,}",
        f"parameters: {record.parameter_count:,}",
        f"score: {score if not isinstance(score, (int, float)) else f'{score:.3f}'}",
        f"vs champion: {record.head_to_head.get('score', '—')}",
        f"outcome: {promoted}",
    ]
    if record.evaluation.get("frontier_depth") is not None:
        lines.append(f"next opponent to beat: depth {record.evaluation['frontier_depth']}")
    if record.wall_seconds:
        lines.append(f"took: {record.wall_seconds / 60:.1f} min")
    if record.gate_reason:
        reason = record.gate_reason
        lines.append("")
        lines.append("why: " + (reason[:150] + "…" if len(reason) > 150 else reason))
    if record.diagnosis:
        lines.append("")
        lines.append("what to fix next:")
        for s in record.diagnosis[:3]:
            finding = str(s.get("finding", ""))[:110]
            lines.append(f"  • {finding}")
    ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9,
            family="monospace", transform=ax.transAxes)
    ax.set_title("Cycle outcome")

    for row in axes:
        for a in row:
            a.grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if out_dir is None:
        if show:
            return fig
        out_dir = Path("store") / "renders" / record.lineage_id
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cycle-{record.cycle:04d}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def describe() -> Dict:
    """What this renderer version draws — recorded so a reader a year from
    now knows what was and was not being measured."""
    return {
        "version": VERSION,
        "task": TASK,
        "expected_series": list(EXPECTED_SERIES),
        "panels": [
            "training error per game",
            "weight-correction size per game",
            "value-estimate magnitude against the reward scale",
            "distinct positions seen",
            "win rate in training games",
            "lookahead budget use and depth reached",
            "value of decided positions against their true values",
            "weights by layer: dying and blowing up",
            "cycle outcome, gate reason, and diagnosis",
        ],
    }
