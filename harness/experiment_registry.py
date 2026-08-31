"""Experiment registry — experiments are first-class citizens.

Append-only JSONL (roadmap Phase 0: formalizes connect4-rl's
``experiments.json``; "start dumb — greppable").

Experiment lifecycle preserves append-only + verdict immutability:
- ``register`` appends a ``registration`` record (hypothesis BEFORE the run).
  Registration REQUIRES ``prior_art``: a list of related experiment IDs, or
  the explicit string "none" — the anti-regression mechanism.
- ``record_verdict`` appends exactly one ``verdict`` record per experiment.
  A second verdict for the same experiment is refused: revisiting a
  hypothesis under new conditions is a NEW experiment citing the old one via
  ``prior_art`` (supersession, not mutation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .jsonl_store import append_jsonl, now_iso, read_jsonl

VERDICTS = ("supported", "refuted", "inconclusive")

_REGISTRATION_REQUIRED = (
    "experiment_id",
    "hypothesis",
    "changes",
    "conditions",
    "policy_refs",
    "prior_art",
)
_CONDITIONS_REQUIRED = ("task_ref", "benchmark_ref", "config", "seed")


@dataclass
class Experiment:
    experiment_id: str
    hypothesis: str
    changes: str
    conditions: Dict
    policy_refs: Dict = field(default_factory=dict)  # {"inputs": [...], "outputs": [...]}
    prior_art: Union[List[str], str] = "none"
    tags: List[str] = field(default_factory=list)
    revisit_when: Optional[str] = None
    results: Optional[Dict] = None
    verdict: Optional[str] = None
    tldr: Optional[str] = None
    superseded_by: Optional[str] = None


class ExperimentRegistry:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)

    # -- write ----------------------------------------------------------

    def register(
        self,
        experiment_id: str,
        hypothesis: str,
        changes: str,
        conditions: Dict,
        prior_art: Union[List[str], str],
        input_policies: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        revisit_when: Optional[str] = None,
    ) -> str:
        """Append a registration record. Hypothesis must contain a measurable
        prediction — enforced socially, but its presence is enforced here."""
        if self._registration(experiment_id) is not None:
            raise ValueError(f"experiment {experiment_id} already registered (append-only)")
        if not hypothesis.strip():
            raise ValueError("hypothesis is required, registered BEFORE the run")
        if prior_art != "none" and not isinstance(prior_art, list):
            raise ValueError('prior_art must be a list of experiment IDs or the string "none"')
        if isinstance(prior_art, list):
            known = {r["experiment_id"] for r in self._records("registration")}
            missing = [p for p in prior_art if p not in known]
            if missing:
                raise ValueError(f"prior_art references unknown experiments: {missing}")
        for key in _CONDITIONS_REQUIRED:
            if key not in conditions:
                raise ValueError(f"conditions missing required field: {key}")
        record = {
            "kind": "registration",
            "experiment_id": experiment_id,
            "hypothesis": hypothesis,
            "changes": changes,
            "conditions": conditions,
            "policy_refs": {"inputs": input_policies or [], "outputs": []},
            "prior_art": prior_art,
            "tags": tags or [],
            "revisit_when": revisit_when,
            "registered_at": now_iso(),
        }
        for key in _REGISTRATION_REQUIRED:
            assert key in record
        append_jsonl(self.path, record)
        return experiment_id

    def add_output_policy(self, experiment_id: str, policy_id: str) -> None:
        """Append an output-policy link event (experiment -> policy)."""
        self._require_registered(experiment_id)
        append_jsonl(
            self.path,
            {
                "kind": "output_policy",
                "experiment_id": experiment_id,
                "policy_id": policy_id,
                "at": now_iso(),
            },
        )

    def record_verdict(
        self,
        experiment_id: str,
        verdict: str,
        tldr: str,
        results: Dict,
    ) -> None:
        """Exactly one verdict per experiment, scoped to its registered
        conditions, immutable once written."""
        self._require_registered(experiment_id)
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
        if not tldr.strip():
            raise ValueError("tldr is required — one line, what we learned")
        if any(
            r["experiment_id"] == experiment_id
            for r in self._records("verdict")
        ):
            raise ValueError(
                f"experiment {experiment_id} already has a verdict (immutable); "
                "register a NEW experiment citing it via prior_art"
            )
        append_jsonl(
            self.path,
            {
                "kind": "verdict",
                "experiment_id": experiment_id,
                "verdict": verdict,
                "tldr": tldr,
                "results": results,
                "at": now_iso(),
            },
        )

    def mark_superseded(self, old_experiment_id: str, new_experiment_id: str) -> None:
        self._require_registered(old_experiment_id)
        self._require_registered(new_experiment_id)
        append_jsonl(
            self.path,
            {
                "kind": "superseded",
                "experiment_id": old_experiment_id,
                "superseded_by": new_experiment_id,
                "at": now_iso(),
            },
        )

    # -- read -----------------------------------------------------------

    def get(self, experiment_id: str) -> Optional[Experiment]:
        reg = self._registration(experiment_id)
        if reg is None:
            return None
        exp = Experiment(
            experiment_id=reg["experiment_id"],
            hypothesis=reg["hypothesis"],
            changes=reg["changes"],
            conditions=reg["conditions"],
            policy_refs=dict(reg["policy_refs"]),
            prior_art=reg["prior_art"],
            tags=reg.get("tags", []),
            revisit_when=reg.get("revisit_when"),
        )
        outputs = list(exp.policy_refs.get("outputs", []))
        for rec in read_jsonl(self.path):
            if rec.get("experiment_id") != experiment_id:
                continue
            if rec["kind"] == "output_policy":
                outputs.append(rec["policy_id"])
            elif rec["kind"] == "verdict":
                exp.verdict = rec["verdict"]
                exp.tldr = rec["tldr"]
                exp.results = rec["results"]
            elif rec["kind"] == "superseded":
                exp.superseded_by = rec["superseded_by"]
        exp.policy_refs["outputs"] = outputs
        return exp

    def search(self, text: str) -> List[Experiment]:
        """Dumb index: substring match over hypothesis, changes, tags, tldr.
        The mandatory pre-registration prior-art search runs through this."""
        needle = text.lower()
        out = []
        for reg in self._records("registration"):
            exp = self.get(reg["experiment_id"])
            haystack = " ".join(
                [exp.hypothesis, exp.changes, " ".join(exp.tags), exp.tldr or ""]
            ).lower()
            if needle in haystack:
                out.append(exp)
        return out

    def all_ids(self) -> List[str]:
        return [r["experiment_id"] for r in self._records("registration")]

    # -- internals ------------------------------------------------------

    def _records(self, kind: str) -> List[Dict]:
        return [r for r in read_jsonl(self.path) if r.get("kind") == kind]

    def _registration(self, experiment_id: str) -> Optional[Dict]:
        for rec in self._records("registration"):
            if rec["experiment_id"] == experiment_id:
                return rec
        return None

    def _require_registered(self, experiment_id: str) -> None:
        if self._registration(experiment_id) is None:
            raise ValueError(f"unknown experiment: {experiment_id}")
