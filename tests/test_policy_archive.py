"""Archive invariants: append-only, save/load round-trip, lineage integrity,
champion pointer derived from events (never mutation), epitaphs."""

import numpy as np
import pytest
import torch

from harness.agents.dqn import DQNTrainer, TrainerConfig
from harness.policy_archive import Archive


def make_scripted(seed=0):
    trainer = DQNTrainer(6, 7, 7, TrainerConfig(seed=seed))
    return trainer.scripted()


def metadata(parent=None, gen=0, score=0.5):
    return {
        "task_ref": "connect4@v1",
        "experiment_id": "exp-test",
        "parent_id": parent,
        "generation": gen,
        "training_config": {},
        "eval_records": [
            {"task_ref": "connect4@v1", "benchmark_ref": "connect4-benchmark@v0", "score": score}
        ],
    }


def test_append_only_refuses_duplicate_id(tmp_path):
    archive = Archive(tmp_path)
    archive.save_policy("p-0", make_scripted(), metadata())
    with pytest.raises(ValueError, match="append-only"):
        archive.save_policy("p-0", make_scripted(1), metadata())


def test_no_delete_api():
    assert not any("delete" in name.lower() or "remove" in name.lower() for name in dir(Archive))


def test_round_trip_weights_and_metadata(tmp_path):
    archive = Archive(tmp_path)
    scripted = make_scripted(seed=42)
    archive.save_policy("p-0", scripted, metadata(score=0.77))
    loaded, meta = archive.load_policy("p-0")
    x = torch.zeros(1, 2, 6, 7)
    assert torch.equal(scripted(x), loaded(x))
    x = torch.rand(4, 2, 6, 7)
    assert torch.allclose(scripted(x), loaded(x))
    assert meta["policy_id"] == "p-0"
    assert meta["eval_records"][0]["score"] == 0.77
    assert meta["task_ref"] == "connect4@v1"


def test_lineage_integrity(tmp_path):
    archive = Archive(tmp_path)
    with pytest.raises(ValueError, match="lineage broken"):
        archive.save_policy("orphan", make_scripted(), metadata(parent="ghost"))
    archive.save_policy("p-0", make_scripted(), metadata())
    archive.save_policy("p-1", make_scripted(1), metadata(parent="p-0", gen=1))
    archive.save_policy("p-2", make_scripted(2), metadata(parent="p-1", gen=2))
    chain = archive.lineage("p-2")
    assert [e["policy_id"] for e in chain] == ["p-0", "p-1", "p-2"]


def test_metadata_required_fields(tmp_path):
    archive = Archive(tmp_path)
    bad = metadata()
    del bad["experiment_id"]
    with pytest.raises(ValueError, match="experiment_id"):
        archive.save_policy("p-0", make_scripted(), bad)
    bad2 = metadata()
    del bad2["eval_records"][0]["benchmark_ref"]
    with pytest.raises(ValueError, match="benchmark_ref"):
        archive.save_policy("p-0", make_scripted(), bad2)


def test_champion_pointer_is_event_derived(tmp_path):
    archive = Archive(tmp_path)
    assert archive.current_champion("connect4") is None
    archive.save_policy("p-0", make_scripted(), metadata())
    archive.record_promotion("connect4", "p-0", "initial")
    assert archive.current_champion("connect4") == "p-0"
    archive.save_policy("p-1", make_scripted(1), metadata(parent="p-0", gen=1))
    archive.record_promotion("connect4", "p-1", "beat champion")
    assert archive.current_champion("connect4") == "p-1"
    # History preserved: both promotion events remain on disk.
    events = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len([e for e in events if "promotion" in e]) == 2
    # Promoting an unarchived policy is refused.
    with pytest.raises(ValueError):
        archive.record_promotion("connect4", "ghost", "nope")


def test_epitaph(tmp_path):
    archive = Archive(tmp_path)
    archive.save_policy("p-0", make_scripted(), metadata())
    archive.record_epitaph("p-0", "gen 0: untrained seed champion")
    assert "untrained" in archive.epitaph("p-0")
