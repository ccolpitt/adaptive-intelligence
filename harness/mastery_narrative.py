"""Mastery narrative — "explain how we did it".

Given a policy ID, walk its lineage and the linked experiments and tell the
story: each ancestor's hypothesis, verdict, tldr, scores, and epitaph, in
order. A pure read over archive + registry; if it cannot be generated, the
metadata links are broken.
"""

from __future__ import annotations

from typing import List

from .improvement_loop import Store


def narrative(store: Store, policy_id: str) -> str:
    chain = store.archive.lineage(policy_id)
    if not chain:
        raise ValueError(f"unknown policy: {policy_id}")

    lines: List[str] = [f"How we got to {policy_id} ({len(chain)} generations)", "=" * 60]
    seen_experiments = set()
    for entry in chain:
        pid = entry["policy_id"]
        lines.append(f"\n{pid}  (gen {entry.get('generation', '?')}, {entry.get('archived_at', '')[:10]})")
        for rec in entry.get("eval_records", []):
            lines.append(f"  score {rec['score']:.3f}  vs {rec.get('opponent', rec['benchmark_ref'])}")
        epitaph = store.archive.epitaph(pid)
        if epitaph:
            lines.append(f"  epitaph: {epitaph}")
        exp_id = entry.get("experiment_id")
        if exp_id and exp_id not in seen_experiments:
            seen_experiments.add(exp_id)
            exp = store.experiments.get(exp_id)
            if exp is not None:
                lines.append(f"  experiment {exp_id}:")
                lines.append(f"    hypothesis: {exp.hypothesis}")
                lines.append(f"    changes:    {exp.changes}")
                verdict = exp.verdict or "pending"
                lines.append(f"    verdict:    {verdict}" + (f" — {exp.tldr}" if exp.tldr else ""))
                if exp.superseded_by:
                    lines.append(f"    superseded by: {exp.superseded_by}")
    return "\n".join(lines)
