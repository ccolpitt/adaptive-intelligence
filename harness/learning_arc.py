"""The learning arc: is this line of policies actually getting stronger?

A cycle record answers "what happened during one round of training". This
answers the question that matters more and is easy to lose sight of while
staring at individual runs: **across many rounds, is the policy improving?**

It reads every cycle record for a lineage and reports the trend — the score
of the policy being trained, the score of the champion after each promotion,
how many games it has cost so far, and how the network and the rubric
changed along the way. Deliberately separate from the per-cycle view, because
a cycle can look healthy while the arc is flat, and the arc is the thing the
project is trying to bend upward.

Like cycle records, the arc is data first: ``summarize`` returns plain
numbers, and plotting is a separate step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from .cycle_record import CycleStore
from .lineage import LineageStore, RubricStore


def summarize(store_root: Union[str, Path], lineage_id: str) -> Dict:
    """The arc of one lineage, as numbers."""
    root = Path(store_root)
    cycles = CycleStore(root / "cycles").load_all(lineage_id)
    lineages = LineageStore(root / "lineages")
    lineage = lineages.get(lineage_id)
    rubrics = RubricStore(root / "rubrics")

    points = []
    champion_score, cumulative = None, 0
    for record in cycles:
        cumulative += record.games_trained
        score = record.evaluation.get("score")
        if record.promoted and isinstance(score, (int, float)):
            champion_score = score
        points.append({
            "cycle": record.cycle,
            "cumulative_games": record.cumulative_games or cumulative,
            "challenger_score": score,
            "champion_score": champion_score,
            "promoted": record.promoted,
            "parity": record.parity,
            "parameter_count": record.parameter_count,
            "search_budget": record.search_budget,
            "rubric_id": record.rubric_id,
            "frontier_depth": record.evaluation.get("frontier_depth"),
        })

    scores = [p["challenger_score"] for p in points
              if isinstance(p["challenger_score"], (int, float))]
    promoted_scores = [p["champion_score"] for p in points
                       if p["promoted"] and isinstance(p["champion_score"], (int, float))]

    # Is it still climbing, or has it levelled off? Compare the best score in
    # the most recent third with the best before it.
    trend = "insufficient data"
    if len(scores) >= 6:
        cut = len(scores) * 2 // 3
        before, recent = max(scores[:cut]), max(scores[cut:])
        if recent > before + 0.02:
            trend = "improving"
        elif recent < before - 0.02:
            trend = "declining"
        else:
            trend = "flat"

    rubric_ids = sorted({p["rubric_id"] for p in points if p["rubric_id"]})
    return {
        "lineage_id": lineage_id,
        "task_id": lineage.task_id if lineage else None,
        "description": lineage.description if lineage else "",
        "parent_lineage": lineage.parent_lineage if lineage else None,
        "cycles": len(points),
        "total_games": cumulative,
        "best_challenger_score": max(scores) if scores else None,
        "current_champion_score": champion_score,
        "promotions": sum(1 for p in points if p["promoted"]),
        "parity_cycles": sum(1 for p in points if p["parity"] and not p["promoted"]),
        "trend": trend,
        "rubrics_used": rubric_ids,
        "rubric_summaries": {r: (rubrics.get(r).summary() if rubrics.get(r) else r)
                             for r in rubric_ids},
        "points": points,
    }


def save_summary(store_root: Union[str, Path], lineage_id: str) -> str:
    """Write the arc as JSON. Data, so a later renderer can redraw it."""
    summary = summarize(store_root, lineage_id)
    out = Path(store_root) / "arcs"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{lineage_id}.json"
    path.write_text(json.dumps(summary, indent=2))
    return str(path)


def render_text(summary: Dict) -> str:
    lines = [
        f"LEARNING ARC — {summary['lineage_id']}  ({summary['task_id']})",
        "=" * 76,
        summary["description"],
        "",
        f"cycles: {summary['cycles']}    games trained: {summary['total_games']:,}    "
        f"promotions: {summary['promotions']}    trend: {summary['trend'].upper()}",
        f"best score by any challenger: {summary['best_challenger_score']}",
        f"current champion score: {summary['current_champion_score']}",
    ]
    if summary["parent_lineage"]:
        lines.append(f"restarted from: {summary['parent_lineage']}")
    for rubric_id, text in summary["rubric_summaries"].items():
        lines.append(f"rubric: {text}")
    lines += ["", f"{'cycle':>6} {'games':>9} {'challenger':>11} {'champion':>9} "
                  f"{'next opp':>9}  outcome"]
    for p in summary["points"]:
        ch = p["challenger_score"]
        cp = p["champion_score"]
        outcome = "PROMOTED" if p["promoted"] else ("parity" if p["parity"] else "")
        lines.append(
            f"{p['cycle']:>6} {p['cumulative_games']:>9,} "
            f"{(f'{ch:.3f}' if isinstance(ch, (int, float)) else '—'):>11} "
            f"{(f'{cp:.3f}' if isinstance(cp, (int, float)) else '—'):>9} "
            f"{str(p['frontier_depth'] or '—'):>9}  {outcome}"
        )
    return "\n".join(lines)


def render_plot(summary: Dict, out_path: Union[str, Path, None] = None) -> str:
    """Three panels: strength over cycles, strength per game spent (the
    sample-efficiency view), and the lookahead budget in force."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = summary["points"]
    if not points:
        raise ValueError(f"no cycles recorded for {summary['lineage_id']}")

    cycles = [p["cycle"] for p in points]
    games = [p["cumulative_games"] for p in points]
    challenger = [p["challenger_score"] for p in points]
    champion = [p["champion_score"] for p in points]
    promoted = [p["cycle"] for p in points if p["promoted"]]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 12),
                                        gridspec_kw={"height_ratios": [3, 3, 1]})
    fig.suptitle(
        f"Learning arc — {summary['lineage_id']}   "
        f"({summary['cycles']} cycles, {summary['total_games']:,} games, "
        f"trend: {summary['trend']})",
        fontsize=13, fontweight="bold",
    )

    ax1.plot(cycles, challenger, marker="o", markersize=4, color="tab:blue",
             alpha=0.7, label="policy being trained")
    ax1.step(cycles, champion, where="post", color="tab:orange", linewidth=2,
             label="champion")
    for c in promoted:
        ax1.axvline(c, color="green", linestyle=":", alpha=0.5)
    ax1.axhline(0.90, color="gray", linestyle="--", alpha=0.6, label="mastery")
    ax1.set_ylabel("score on the task")
    ax1.set_xlabel("cycle")
    ax1.legend(fontsize=9)
    ax1.set_title("Strength over cycles (green lines mark promotions)")

    valid = [(g, s) for g, s in zip(games, champion) if isinstance(s, (int, float))]
    if valid:
        ax2.plot([g for g, _ in valid], [s for _, s in valid],
                 marker="o", markersize=4, color="tab:green")
    ax2.set_xlabel("training games spent (cumulative)")
    ax2.set_ylabel("champion score")
    ax2.set_title("Strength per game spent — the sample-efficiency view")

    ax3.plot(cycles, [p["search_budget"] for p in points], drawstyle="steps-post",
             color="tab:cyan")
    ax3.set_ylabel("lookahead\nbudget")
    ax3.set_xlabel("cycle")
    ax3.set_title("States the policy was allowed to examine per move while training")

    for ax in (ax1, ax2, ax3):
        ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if out_path is None:
        out_path = Path("store") / "arcs" / f"{summary['lineage_id']}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return str(out_path)
