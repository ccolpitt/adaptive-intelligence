"""Experiment cards — one visual, auditable page per experiment.

    python -m harness exp-card --experiment-id exp-012-signal-package
    python -m harness exp-card --all

Joins the three stores an experiment leaves traces in:

- the EXPERIMENT REGISTRY (the "if I do X, I expect Y" hypothesis, the one
  change, the frozen conditions — task@v, benchmark@v, seed, full config —
  prior art, and the immutable verdict once recorded),
- the TELEMETRY file (per-iteration benchmark / head-to-head / loss /
  gradient-norm / dormant-neuron series the loop records while running),
- the POLICY ARCHIVE (which policies the experiment consumed and produced,
  and which were promoted).

Rendered as a self-contained HTML card with inline SVG charts (no image
files, no plotting dependency — a card is a few KB of text and renders in
any browser) plus a terminal text summary. `--all` also writes an index page
sorted newest-first. Cards are derived artifacts under
``store/cards/experiments/`` — regenerable at will, never authoritative.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# data gathering
# ---------------------------------------------------------------------------


def gather(store, experiment_id: str) -> Dict:
    exp = store.experiments.get(experiment_id)
    if exp is None:
        raise ValueError(f"unknown experiment: {experiment_id}")
    telemetry: Dict = {}
    tpath = Path(store.telemetry_dir) / f"{experiment_id}.json"
    if tpath.exists():
        telemetry = json.loads(tpath.read_text())

    outputs = exp.policy_refs.get("outputs", [])
    iterations = telemetry.get("iterations", [])
    promoted = [it["challenger"] for it in iterations if it.get("promoted")]

    return {
        "experiment_id": experiment_id,
        "hypothesis": exp.hypothesis,
        "changes": exp.changes,
        "conditions": exp.conditions,
        "prior_art": exp.prior_art,
        "tags": exp.tags,
        "revisit_when": exp.revisit_when,
        "inputs": exp.policy_refs.get("inputs", []),
        "outputs": outputs,
        "promoted": promoted,
        "verdict": exp.verdict,
        "tldr": exp.tldr,
        "results": exp.results,
        "superseded_by": exp.superseded_by,
        "telemetry": telemetry,
    }


# ---------------------------------------------------------------------------
# SVG chart (dependency-free)
# ---------------------------------------------------------------------------

PALETTE = ["#4f9cf9", "#f97b4f", "#59c26b", "#c95fd6", "#e6b93f", "#5fd6c9"]


def svg_chart(
    series: Dict[str, Sequence[Optional[float]]],
    title: str,
    width: int = 340,
    height: int = 150,
    hline: Optional[float] = None,
    markers: Optional[Sequence[int]] = None,
) -> str:
    """Multi-series line chart as inline SVG. ``markers`` are x indices to
    highlight (e.g. promoted iterations)."""
    pad_l, pad_r, pad_t, pad_b = 42, 8, 22, 18
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    values = [v for s in series.values() for v in s if v is not None]
    if not values:
        return (f'<svg width="{width}" height="{height}"><text x="8" y="20" '
                f'class="ct">{html.escape(title)}</text>'
                f'<text x="{pad_l}" y="{height // 2}" class="cm">no data</text></svg>')
    lo = min(values + ([hline] if hline is not None else []))
    hi = max(values + ([hline] if hline is not None else []))
    if hi == lo:
        hi, lo = hi + 1e-9, lo - 1e-9
    span = hi - lo
    lo, hi = lo - 0.05 * span, hi + 0.05 * span
    n = max(len(s) for s in series.values())

    def x(i: int) -> float:
        return pad_l + (plot_w * i / max(n - 1, 1))

    def y(v: float) -> float:
        return pad_t + plot_h * (1 - (v - lo) / (hi - lo))

    parts = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append(f'<text x="8" y="14" class="ct">{html.escape(title)}</text>')
    parts.append(f'<text x="4" y="{pad_t + 8}" class="cm">{hi:.3g}</text>')
    parts.append(f'<text x="4" y="{height - pad_b}" class="cm">{lo:.3g}</text>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" class="ax"/>')
    parts.append(f'<line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" '
                 f'y2="{height - pad_b}" class="ax"/>')
    if hline is not None and lo <= hline <= hi:
        parts.append(f'<line x1="{pad_l}" y1="{y(hline):.1f}" x2="{width - pad_r}" '
                     f'y2="{y(hline):.1f}" class="hl"/>')
    for idx, (name, s) in enumerate(series.items()):
        color = PALETTE[idx % len(PALETTE)]
        pts = [(x(i), y(v)) for i, v in enumerate(s) if v is not None]
        if len(pts) == 1:
            parts.append(f'<circle cx="{pts[0][0]:.1f}" cy="{pts[0][1]:.1f}" r="3" fill="{color}"/>')
        elif pts:
            path = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
            parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="1.6"/>')
        if len(series) > 1:
            parts.append(f'<text x="{pad_l + 6 + 92 * idx}" y="{height - 4}" fill="{color}" '
                         f'class="cm">{html.escape(name)}</text>')
    for m in markers or []:
        parts.append(f'<line x1="{x(m):.1f}" y1="{pad_t}" x2="{x(m):.1f}" '
                     f'y2="{height - pad_b}" class="mk"/>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# renderers
# ---------------------------------------------------------------------------

_CSS = """
body { background:#14171c; color:#d8dee6; font:14px/1.5 -apple-system, 'Segoe UI', sans-serif;
       max-width: 1000px; margin: 24px auto; padding: 0 16px; }
h1 { font-size: 20px; } h2 { font-size: 15px; color:#9fb2c8; margin: 22px 0 6px;
     text-transform: uppercase; letter-spacing: .08em; }
.badge { display:inline-block; padding:2px 10px; border-radius:10px; font-weight:600; }
.supported { background:#1d4a2a; color:#7fe39a; } .refuted { background:#542222; color:#f2938d; }
.inconclusive { background:#4a431d; color:#e3d47f; } .pending { background:#2a3442; color:#9fb2c8; }
.hyp { background:#1b2129; border-left:3px solid #4f9cf9; padding:10px 14px; border-radius:4px; }
table { border-collapse: collapse; width:100%; font-size:13px; }
td, th { padding: 4px 8px; border-bottom: 1px solid #262d36; text-align:left; }
th { color:#9fb2c8; } tr.promoted td { background:#1d2e22; }
.grid { display:flex; flex-wrap:wrap; gap:14px; }
.card { background:#1b2129; border-radius:6px; padding:8px; }
.ct { fill:#9fb2c8; font-size:11px; } .cm { fill:#68788c; font-size:9px; }
.ax { stroke:#333c47; stroke-width:1; } .hl { stroke:#8895a5; stroke-dasharray:4 3; }
.mk { stroke:#59c26b; stroke-dasharray:2 3; }
code, pre { background:#1b2129; border-radius:4px; padding:1px 5px; font-size:12.5px; }
pre { padding:10px; overflow-x:auto; }
.kv td:first-child { color:#9fb2c8; white-space:nowrap; vertical-align:top; }
"""


def _fmt(v: Optional[float], spec: str = ".3f") -> str:
    """None-tolerant number formatting — early telemetry schemas lack fields."""
    return format(v, spec) if isinstance(v, (int, float)) else "—"


def render_html(data: Dict) -> str:
    t = data["telemetry"]
    iterations = t.get("iterations", [])
    verdict = data["verdict"] or "pending"
    e = html.escape

    bench = {"challenger": [it.get("benchmark_score") for it in iterations],
             "champion": [it.get("champion_benchmark") for it in iterations]}
    h2h = {"h2h": [it.get("h2h_score_vs_champion") for it in iterations]}
    dormant: Dict[str, List[Optional[float]]] = {}
    for it in iterations:
        for layer, frac in (it.get("dormant_neurons") or {}).items():
            dormant.setdefault(layer, [None] * len(iterations))
    for i, it in enumerate(iterations):
        for layer, frac in (it.get("dormant_neurons") or {}).items():
            dormant[layer][i] = frac
    promoted_idx = [i for i, it in enumerate(iterations) if it.get("promoted")]

    charts = [
        svg_chart(bench, "benchmark score (absolute yardstick)", markers=promoted_idx),
        svg_chart(h2h, "head-to-head vs champion", hline=0.5, markers=promoted_idx),
        svg_chart({"loss": t.get("episode_loss_means", [])}, "training loss (MSE, per-iteration mean)"),
    ]
    if any(x is not None for x in t.get("episode_grad_norm_means", [])):
        charts.append(svg_chart({"norm": t.get("episode_grad_norm_means", [])},
                                "gradient norm (pre-clip L2)"))
    if dormant:
        charts.append(svg_chart(dormant, "dormant weights by layer"))
    if any("epsilon" in it for it in iterations):
        charts.append(svg_chart({"eps": [it.get("epsilon") for it in iterations]},
                                "exploration (epsilon)"))

    cond = data["conditions"]
    setup_rows = "".join(
        f"<tr><td>{e(str(k))}</td><td><code>{e(json.dumps(v, sort_keys=True, default=str))}</code></td></tr>"
        for k, v in cond.items() if k != "config"
    )
    config_pre = e(json.dumps(cond.get("config", {}), indent=1, sort_keys=True, default=str))

    iter_rows = "".join(
        f'<tr class="{"promoted" if it.get("promoted") else ""}">'
        f'<td>{it.get("iteration")}</td><td><code>{e(str(it.get("challenger")))}</code></td>'
        f'<td>{_fmt(it.get("h2h_score_vs_champion"))}</td>'
        f'<td>{_fmt(it.get("benchmark_score"))}</td>'
        f'<td>{"★ promoted" if it.get("promoted") else "rejected"}</td>'
        f'<td>{e(str(it.get("reason", "")))}</td></tr>'
        for it in iterations
    )

    results_block = ""
    if data["results"]:
        results_block = f"<h2>Recorded results</h2><pre>{e(json.dumps(data['results'], indent=1))}</pre>"

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{e(data['experiment_id'])}</title><style>{_CSS}</style></head><body>
<h1>{e(data['experiment_id'])} <span class="badge {verdict}">{verdict}</span></h1>
{f'<p><b>tl;dr</b> — {e(data["tldr"])}</p>' if data['tldr'] else ''}
{f'<p>superseded by <code>{e(str(data["superseded_by"]))}</code></p>' if data['superseded_by'] else ''}

<h2>Hypothesis (if X, then Y)</h2>
<div class="hyp">{e(data['hypothesis'])}</div>
<p><b>the one change:</b> {e(data['changes'])}<br>
<b>prior art:</b> {e(json.dumps(data['prior_art']))}
{f"<br><b>revisit when:</b> {e(data['revisit_when'])}" if data['revisit_when'] else ''}</p>

<h2>Setup (auditable conditions)</h2>
<table class="kv">
<tr><td>input policies (warm start)</td><td><code>{e(json.dumps(data['inputs']))}</code></td></tr>
{setup_rows}
</table>
<details><summary>full config</summary><pre>{config_pre}</pre></details>

<h2>Result curves</h2>
<div class="grid">{''.join(f'<div class="card">{c}</div>' for c in charts)}</div>

<h2>Iterations ({len(promoted_idx)} promotion{'s' if len(promoted_idx) != 1 else ''},
{len(data['outputs'])} polic{'ies' if len(data['outputs']) != 1 else 'y'} archived)</h2>
<table><tr><th>#</th><th>challenger</th><th>h2h</th><th>bench</th><th>gate</th><th>reason</th></tr>
{iter_rows}</table>
{results_block}
</body></html>"""


def render_text(data: Dict) -> str:
    t = data["telemetry"]
    iterations = t.get("iterations", [])
    verdict = (data["verdict"] or "PENDING").upper()
    lines = [
        f"EXPERIMENT CARD — {data['experiment_id']}  [{verdict}]",
        "=" * 72,
        f"hypothesis: {data['hypothesis']}",
        f"change:     {data['changes']}",
        f"prior art:  {json.dumps(data['prior_art'])}",
        f"conditions: task={data['conditions'].get('task_ref')} "
        f"benchmark={data['conditions'].get('benchmark_ref')} "
        f"seed={data['conditions'].get('seed')}",
        f"inputs:     {json.dumps(data['inputs'])}",
    ]
    if data["tldr"]:
        lines.append(f"tl;dr:      {data['tldr']}")
    if iterations:
        first, last = iterations[0], iterations[-1]
        promoted = [it for it in iterations if it.get("promoted")]
        lines += [
            "-" * 72,
            f"iterations: {len(iterations)}, promotions: {len(promoted)}",
            f"benchmark:  {_fmt(first.get('benchmark_score'))} -> {_fmt(last.get('benchmark_score'))} "
            f"(champion held: {_fmt(last.get('champion_benchmark'))})",
        ]
        for it in iterations:
            mark = "★" if it.get("promoted") else " "
            lines.append(
                f"  {mark} iter {it.get('iteration'):>2}  h2h={_fmt(it.get('h2h_score_vs_champion'), '.2f')}  "
                f"bench={_fmt(it.get('benchmark_score'))}  {it.get('challenger')}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# saving + index
# ---------------------------------------------------------------------------


def save_experiment_card(store, experiment_id: str) -> Dict[str, str]:
    data = gather(store, experiment_id)
    out_dir = Path(store.root) / "cards" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{experiment_id}.html"
    html_path.write_text(render_html(data))
    return {"html": str(html_path), "text": render_text(data)}


def save_all(store) -> str:
    """Render every registered experiment plus an index page, newest last in
    the registry order (append-only means registry order IS chronology)."""
    ids = store.experiments.all_ids()
    rows = []
    for exp_id in ids:
        save_experiment_card(store, exp_id)
        data = gather(store, exp_id)
        iterations = data["telemetry"].get("iterations", [])
        final_bench = iterations[-1].get("champion_benchmark") if iterations else None
        verdict = data["verdict"] or "pending"
        rows.append(
            f'<tr><td><a href="{html.escape(exp_id)}.html">{html.escape(exp_id)}</a></td>'
            f'<td><span class="badge {verdict}">{verdict}</span></td>'
            f'<td>{f"{final_bench:.3f}" if final_bench is not None else "—"}</td>'
            f'<td>{sum(1 for it in iterations if it.get("promoted"))}</td>'
            f'<td>{html.escape((data["tldr"] or data["hypothesis"])[:140])}</td></tr>'
        )
    index = (f"<!doctype html><html><head><meta charset='utf-8'><title>experiments</title>"
             f"<style>{_CSS}</style></head><body><h1>Experiment cards ({len(ids)})</h1>"
             f"<table><tr><th>experiment</th><th>verdict</th><th>final champion bench</th>"
             f"<th>promotions</th><th>tl;dr / hypothesis</th></tr>"
             f"{''.join(reversed(rows))}</table></body></html>")
    out_dir = Path(store.root) / "cards" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.html"
    index_path.write_text(index)
    return str(index_path)


# ---------------------------------------------------------------------------
# matplotlib dashboard (optional — fine-grained per-episode curves)
# ---------------------------------------------------------------------------


def _smooth(xs: Sequence[Optional[float]], window: int = 25) -> Tuple[List[int], List[float]]:
    """Moving average over the non-None points, returning (x, y)."""
    pts = [(i, v) for i, v in enumerate(xs) if v is not None]
    if len(pts) < window:
        return [p[0] for p in pts], [p[1] for p in pts]
    idx = [p[0] for p in pts]
    vals = [p[1] for p in pts]
    kernel = [1.0 / window] * window
    smoothed = []
    for i in range(len(vals) - window + 1):
        smoothed.append(sum(vals[i:i + window]) / window)
    return idx[window - 1:], smoothed


def save_matplotlib_dashboard(store, experiment_id: str) -> str:
    """3x3 PNG dashboard in the spirit of connect4-rl's training dashboard,
    from stored telemetry alone (no re-training, no probing). Requires
    matplotlib; raises a clear error if it is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as err:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for --plots (pip install matplotlib)"
        ) from err

    data = gather(store, experiment_id)
    t = data["telemetry"]
    ep = t.get("episodes", {})
    iterations = t.get("iterations", [])
    it_x = list(range(len(iterations)))
    promoted_x = [i for i, it in enumerate(iterations) if it.get("promoted")]

    fig, axes = plt.subplots(3, 3, figsize=(20, 12))
    verdict = (data["verdict"] or "pending").upper()
    fig.suptitle(f"{experiment_id}  [{verdict}]", fontsize=15, fontweight="bold")

    def no_data(ax, title):
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray")

    # 1: loss per episode
    ax = axes[0, 0]
    if any(v is not None for v in ep.get("loss", [])):
        raw = ep["loss"]
        ax.plot([i for i, v in enumerate(raw) if v is not None],
                [v for v in raw if v is not None], alpha=0.25, color="tab:blue")
        sx, sy = _smooth(raw)
        ax.plot(sx, sy, color="tab:blue", label="smoothed (25 ep)")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.set_title("Training loss per episode (MSE)")
    else:
        no_data(ax, "Training loss per episode (MSE)")
    ax.set_xlabel("episode")

    # 2: gradient norm per episode
    ax = axes[0, 1]
    if any(v is not None for v in ep.get("grad_norm", [])):
        sx, sy = _smooth(ep["grad_norm"])
        ax.plot(sx, sy, color="tab:purple")
        ax.set_yscale("log")
        ax.set_title("Gradient norm per episode (pre-clip L2, smoothed)")
    else:
        no_data(ax, "Gradient norm per episode")
    ax.set_xlabel("episode")

    # 3: rolling training score (free win-rate signal vs training opponents)
    ax = axes[0, 2]
    results = ep.get("result", [])
    if results:
        score = [(r + 1) / 2 for r in results]  # +1/0/-1 -> 1/0.5/0
        sx, sy = _smooth(score, window=min(100, max(len(score) // 5, 2)))
        ax.plot(sx, sy, color="tab:olive")
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="even (0.5)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.set_title("Rolling training score vs training opponents")
    else:
        no_data(ax, "Rolling training score")
    ax.set_xlabel("episode")

    # 4: unique states explored
    ax = axes[1, 0]
    if ep.get("unique_states"):
        ax.plot(ep["unique_states"], color="tab:green")
        ax.set_title("Unique states explored (cumulative)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    else:
        no_data(ax, "Unique states explored")
    ax.set_xlabel("episode")

    # 5: benchmark per iteration (the absolute yardstick)
    ax = axes[1, 1]
    if iterations:
        ax.plot(it_x, [it.get("benchmark_score") for it in iterations],
                marker="o", label="challenger", color="tab:blue")
        ax.plot(it_x, [it.get("champion_benchmark") for it in iterations],
                marker="s", label="champion", color="tab:orange")
        for px in promoted_x:
            ax.axvline(px, color="green", linestyle=":", alpha=0.6)
        ax.legend(fontsize=8)
        ax.set_title("Benchmark score per iteration (green = promotion)")
    else:
        no_data(ax, "Benchmark score per iteration")
    ax.set_xlabel("iteration")

    # 6: head-to-head per iteration
    ax = axes[1, 2]
    h2h = [it.get("h2h_score_vs_champion") for it in iterations]
    if any(v is not None for v in h2h):
        ax.plot(it_x, h2h, marker="o", color="gold")
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.6)
        ax.set_ylim(0, 1)
        ax.set_title("Challenger vs champion (head-to-head)")
    else:
        no_data(ax, "Challenger vs champion (head-to-head)")
    ax.set_xlabel("iteration")

    # 7: dormant weights by layer
    ax = axes[2, 0]
    layers: Dict[str, List[Optional[float]]] = {}
    for i, it in enumerate(iterations):
        for layer, frac in (it.get("dormant_neurons") or {}).items():
            layers.setdefault(layer, [None] * len(iterations))[i] = frac
    if layers:
        for layer, series in layers.items():
            ax.plot(it_x, series, marker=".", label=layer)
        ax.legend(fontsize=7)
        ax.set_title("Dormant weight fraction by layer (plasticity)")
    else:
        no_data(ax, "Dormant weight fraction by layer")
    ax.set_xlabel("iteration")

    # 8: epsilon + Q calibration (both live near [0, 1])
    ax = axes[2, 1]
    has_eps = bool(ep.get("epsilon"))
    has_q = any(v is not None for v in ep.get("q_mean_abs", []))
    if has_eps or has_q:
        if has_eps:
            ax.plot(ep["epsilon"], color="tab:gray", label="epsilon")
        if has_q:
            sx, sy = _smooth(ep["q_mean_abs"])
            ax.plot(sx, sy, color="tab:red", label="mean |Q| (smoothed)")
            ax.axhline(1.0, color="tab:red", linestyle=":", alpha=0.5, label="|Q| target ceiling")
        ax.legend(fontsize=8)
        ax.set_title("Exploration (epsilon) & Q calibration (mean |Q| ≤ 1)")
        ax.set_xlabel("episode")
    else:
        no_data(ax, "Exploration & Q calibration")

    # 9: summary panel
    ax = axes[2, 2]
    ax.axis("off")
    last_bench = iterations[-1].get("champion_benchmark") if iterations else None
    lines = [
        f"change: {data['changes'][:90]}",
        f"promotions: {len(promoted_x)} / {len(iterations)} iterations",
        f"final champion benchmark: {_fmt(last_bench)}",
        f"policies archived: {len(data['outputs'])}",
        f"verdict: {verdict}",
    ]
    if data["tldr"]:
        lines.append(f"tl;dr: {data['tldr'][:120]}")
    ax.text(0.02, 0.95, "\n\n".join(lines), va="top", fontsize=10,
            family="monospace", wrap=True, transform=ax.transAxes)

    for row in axes:
        for a in row:
            a.grid(True, alpha=0.25)

    out_dir = Path(store.root) / "cards" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{experiment_id}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    return str(png_path)


# ---------------------------------------------------------------------------
# campaign story — the levers/expected/observed ledger, generated from the
# registry so it can never drift out of date
# ---------------------------------------------------------------------------


def _prediction_excerpt(hypothesis: str, limit: int = 500) -> str:
    """Pull the PREDICTION paragraph out of a hypothesis; fall back to the
    hypothesis itself for early experiments written as one sentence."""
    marker = "PREDICTION"
    idx = hypothesis.find(marker)
    if idx >= 0:
        text = hypothesis[idx + len(marker):].lstrip(": ").strip()
        end = text.find("\n\n")
        if end > 0:
            text = text[:end]
    else:
        text = hypothesis.strip()
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


def render_story(store) -> str:
    """Markdown ledger: one block per experiment — the lever(s) pulled, what
    we expected to observe, what we observed, and the verdict."""
    ids = store.experiments.all_ids()
    lines = [
        "# Campaign story — every lever pulled, expectation, and result",
        "",
        "Auto-generated from the experiment registry "
        "(`python -m harness story`); do not edit by hand.",
        "",
        f"Experiments: {len(ids)}. Verdict legend: SUPPORTED = prediction held; "
        "REFUTED = prediction failed (knowledge gained, lever closed); "
        "INCONCLUSIVE = effect below what the sample size can detect; "
        "PENDING = still running or unjudged.",
        "",
    ]
    for exp_id in ids:
        exp = store.experiments.get(exp_id)
        verdict = (exp.verdict or "pending").upper()
        lines += [
            f"## {exp_id} — {verdict}",
            "",
            f"**Lever(s) pulled:** {exp.changes}",
            "",
            f"**Expected:** {_prediction_excerpt(exp.hypothesis)}",
            "",
            f"**Observed:** {exp.tldr or '(no verdict recorded yet)'}",
            "",
        ]
        if exp.results:
            lines += [f"**Key numbers:** `{json.dumps(exp.results, sort_keys=True)}`", ""]
        if isinstance(exp.prior_art, list) and exp.prior_art:
            lines += [f"*Builds on: {', '.join(exp.prior_art)}*", ""]
    return "\n".join(lines)


def save_story(store) -> str:
    out_dir = Path(store.root) / "cards" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "story.md"
    path.write_text(render_story(store))
    return str(path)


# ---------------------------------------------------------------------------
# campaign overview plot — one picture of the whole program
# ---------------------------------------------------------------------------


def save_campaign_plot(store) -> str:
    """Two panels: (top) best ladder score each experiment reached, colored by
    verdict; (bottom) the champion's ladder score after every promotion —
    the lineage the gate actually produced. Ladder-benchmark experiments only
    (connect4@v2); the early vs-random benchmark is a different scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .jsonl_store import read_jsonl

    colors = {"supported": "#59c26b", "refuted": "#f2938d", "inconclusive": "#e3d47f",
              None: "#9fb2c8"}
    labels, bests, cols = [], [], []
    for exp_id in store.experiments.all_ids():
        tpath = Path(store.telemetry_dir) / f"{exp_id}.json"
        if not tpath.exists():
            continue
        t = json.loads(tpath.read_text())
        if t.get("task_ref") != "connect4@v2":
            continue
        its = t.get("iterations", [])
        if not its:
            continue
        exp = store.experiments.get(exp_id)
        labels.append(exp_id.replace("exp-", "").replace("-seed", "/s"))
        bests.append(max(it["benchmark_score"] for it in its))
        cols.append(colors.get(exp.verdict, colors[None]))

    promos = [e for e in read_jsonl(Path(store.root) / "archive" / "events.jsonl")
              if e.get("kind") == "promotion"]
    lineage = []
    for e in promos:
        try:
            _, meta = store.archive.load_policy(e["policy_id"])
        except Exception:
            continue
        bench = next((r["score"] for r in meta.get("eval_records", [])
                      if "connect4-benchmark@v1" in str(r.get("benchmark_ref"))), None)
        if bench is not None:
            lineage.append((e["policy_id"], bench))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, 0.42 * len(labels)), 11),
                                   gridspec_kw={"height_ratios": [3, 2]})
    x = range(len(labels))
    ax1.bar(x, bests, color=cols)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax1.set_ylabel("best challenger ladder score (bare inference)")
    ax1.set_title("Every ladder experiment: best score reached  "
                  "(green=supported, red=refuted, yellow=inconclusive)")
    ax1.axhline(0.90, color="gray", linestyle=":", alpha=0.6, label="mastery (0.90 per rung)")
    if lineage:
        ax1.axhline(lineage[-1][1], color="tab:blue", linestyle="--", alpha=0.7,
                    label=f"current champion {lineage[-1][1]:.3f}")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    if lineage:
        ys = [b for _, b in lineage]
        ax2.step(range(len(ys)), ys, where="post", color="tab:blue", linewidth=2)
        ax2.scatter(range(len(ys)), ys, color="tab:blue")
        for i, (pid, b) in enumerate(lineage):
            ax2.annotate(pid.replace("connect4-pol-", "pol-"), (i, b), fontsize=7,
                         textcoords="offset points", xytext=(0, 6), ha="center")
        ax2.set_xlabel("promotion # (champion lineage, ladder benchmark v1 only)")
        ax2.set_ylabel("champion ladder score")
        ax2.set_title("What the promotion gate produced: champion strength after each promotion")
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(store.root) / "cards" / "campaign.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return str(out)
