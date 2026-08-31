"""Archive — the append-only policy store (the fossil record).

Extends connect4-rl's TorchScript ``_extra_files`` pattern: each policy file
embeds its full metadata as JSON, so any fossil is interpretable standalone,
without the index. The index (index.jsonl) and promotion history
(events.jsonl) are append-only; the "current champion" is DERIVED from
promotion events, never a mutated pointer file (connect4-rl's overwritten
champion_current.pt was a known trap).

Death means "stops receiving compute", never deletion — there is no delete
API on purpose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

from .jsonl_store import append_jsonl, now_iso, read_jsonl

METADATA_KEY = "metadata.json"


class Archive:
    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self.events_path = self.root / "events.jsonl"
        self.index_path.touch(exist_ok=True)
        self.events_path.touch(exist_ok=True)

    # -- write ----------------------------------------------------------

    def next_policy_id(self, task_id: str) -> str:
        prefix = f"{task_id}-pol-"
        n = sum(
            1 for r in read_jsonl(self.index_path)
            if str(r.get("policy_id", "")).startswith(prefix)
        )
        return f"{prefix}{n:05d}"

    def save_policy(
        self,
        policy_id: str,
        scripted_module: torch.jit.ScriptModule,
        metadata: Dict,
    ) -> Path:
        """Append-only: refuses an existing policy_id. ``metadata`` must carry
        the roadmap-required fields (lineage, experiment link, eval records)."""
        required = ("task_ref", "experiment_id", "parent_id", "training_config", "eval_records")
        for key in required:
            if key not in metadata:
                raise ValueError(f"policy metadata missing required field: {key}")
        if self.get_entry(policy_id) is not None:
            raise ValueError(f"policy {policy_id} already archived (append-only)")
        parent = metadata["parent_id"]
        if parent is not None and self.get_entry(parent) is None:
            raise ValueError(f"lineage broken: parent {parent} not in archive")
        for rec in metadata["eval_records"]:
            for key in ("task_ref", "benchmark_ref", "score"):
                if key not in rec:
                    raise ValueError(f"eval record missing required field: {key}")

        meta = dict(metadata)
        meta["policy_id"] = policy_id
        meta["archived_at"] = now_iso()

        path = self.root / f"{policy_id}.pt"
        torch.jit.save(scripted_module, str(path), _extra_files={METADATA_KEY: json.dumps(meta)})
        append_jsonl(self.index_path, meta)
        return path

    def record_promotion(self, task_id: str, policy_id: str, reason: str) -> None:
        if self.get_entry(policy_id) is None:
            raise ValueError(f"cannot promote unarchived policy {policy_id}")
        append_jsonl(
            self.events_path,
            {"kind": "promotion", "task_id": task_id, "policy_id": policy_id,
             "reason": reason, "at": now_iso()},
        )

    def record_epitaph(self, policy_id: str, epitaph: str) -> None:
        """One line — why it lived or died."""
        if self.get_entry(policy_id) is None:
            raise ValueError(f"unknown policy {policy_id}")
        append_jsonl(
            self.events_path,
            {"kind": "epitaph", "policy_id": policy_id, "epitaph": epitaph, "at": now_iso()},
        )

    # -- read -----------------------------------------------------------

    def load_policy(self, policy_id: str) -> Tuple[torch.jit.ScriptModule, Dict]:
        """Architecture-agnostic load: TorchScript needs no class at load time."""
        path = self.root / f"{policy_id}.pt"
        extra = {METADATA_KEY: ""}
        module = torch.jit.load(str(path), map_location="cpu", _extra_files=extra)
        module.eval()
        raw = extra[METADATA_KEY]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return module, json.loads(raw)

    def get_entry(self, policy_id: str) -> Optional[Dict]:
        for rec in read_jsonl(self.index_path):
            if rec.get("policy_id") == policy_id:
                return rec
        return None

    def current_champion(self, task_id: str) -> Optional[str]:
        """Derived from promotion events — the pointer is a fold over
        append-only history, not mutable state."""
        champion = None
        for rec in read_jsonl(self.events_path):
            if rec.get("kind") == "promotion" and rec.get("task_id") == task_id:
                champion = rec["policy_id"]
        return champion

    def lineage(self, policy_id: str) -> List[Dict]:
        """Root-first chain of index entries ending at ``policy_id``."""
        chain = []
        entry = self.get_entry(policy_id)
        while entry is not None:
            chain.append(entry)
            parent = entry.get("parent_id")
            entry = self.get_entry(parent) if parent else None
        return list(reversed(chain))

    def epitaph(self, policy_id: str) -> Optional[str]:
        text = None
        for rec in read_jsonl(self.events_path):
            if rec.get("kind") == "epitaph" and rec.get("policy_id") == policy_id:
                text = rec["epitaph"]
        return text

    def all_entries(self) -> List[Dict]:
        return read_jsonl(self.index_path)
