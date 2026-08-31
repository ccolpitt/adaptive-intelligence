"""Registry invariants: append-only, verdict immutability, schema validation,
cross-reference resolution, prior-art requirement, supersession."""

import pytest

from harness.experiment_registry import ExperimentRegistry
from harness.task_registry import TaskRegistry


def conditions():
    return {
        "task_ref": "connect4@v1",
        "benchmark_ref": "connect4-benchmark@v0",
        "config": {"episodes": 100},
        "seed": 0,
    }


# -- task registry --------------------------------------------------------

def test_task_register_and_get(tmp_path):
    reg = TaskRegistry(tmp_path / "tasks.jsonl")
    definition = {"task_id": "connect4", "version": 1, "benchmark_ref": "connect4-benchmark@v0"}
    reg.register(definition)
    assert reg.get("connect4", 1)["benchmark_ref"] == "connect4-benchmark@v0"
    assert reg.get("connect4", 2) is None


def test_task_reregister_identical_is_noop(tmp_path):
    reg = TaskRegistry(tmp_path / "tasks.jsonl")
    d = {"task_id": "connect4", "version": 1, "benchmark_ref": "b@v0"}
    reg.register(d)
    reg.register(d)
    assert len(reg.all()) == 1


def test_task_changed_definition_same_version_refused(tmp_path):
    reg = TaskRegistry(tmp_path / "tasks.jsonl")
    reg.register({"task_id": "connect4", "version": 1, "benchmark_ref": "b@v0"})
    with pytest.raises(ValueError, match="bump the version"):
        reg.register({"task_id": "connect4", "version": 1, "benchmark_ref": "b@v1"})


# -- experiment registry ---------------------------------------------------

def test_register_requires_hypothesis_and_conditions(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp.jsonl")
    with pytest.raises(ValueError, match="hypothesis"):
        reg.register("e1", "  ", "changes", conditions(), "none")
    with pytest.raises(ValueError, match="conditions missing"):
        reg.register("e1", "if X then Y", "changes", {"task_ref": "t"}, "none")


def test_duplicate_registration_refused(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp.jsonl")
    reg.register("e1", "if X then Y", "X", conditions(), "none")
    with pytest.raises(ValueError, match="append-only"):
        reg.register("e1", "if X then Z", "X", conditions(), "none")


def test_prior_art_must_resolve(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp.jsonl")
    with pytest.raises(ValueError, match="unknown experiments"):
        reg.register("e1", "if X then Y", "X", conditions(), prior_art=["ghost"])
    reg.register("e1", "if X then Y", "X", conditions(), "none")
    reg.register("e2", "if X' then Y'", "X'", conditions(), prior_art=["e1"])
    assert reg.get("e2").prior_art == ["e1"]


def test_verdict_once_and_immutable(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp.jsonl")
    reg.register("e1", "if X then Y", "X", conditions(), "none")
    with pytest.raises(ValueError, match="verdict must be one of"):
        reg.record_verdict("e1", "maybe", "tldr", {})
    with pytest.raises(ValueError, match="tldr"):
        reg.record_verdict("e1", "supported", "  ", {})
    reg.record_verdict("e1", "refuted", "X did not produce Y at this scale", {"y": 0.1})
    with pytest.raises(ValueError, match="immutable"):
        reg.record_verdict("e1", "supported", "changed my mind", {})
    exp = reg.get("e1")
    assert exp.verdict == "refuted"
    assert exp.results == {"y": 0.1}


def test_policy_refs_bidirectional_links(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp.jsonl")
    reg.register("e1", "if X then Y", "X", conditions(), "none", input_policies=["pol-0"])
    reg.add_output_policy("e1", "pol-1")
    reg.add_output_policy("e1", "pol-2")
    exp = reg.get("e1")
    assert exp.policy_refs["inputs"] == ["pol-0"]
    assert exp.policy_refs["outputs"] == ["pol-1", "pol-2"]
    with pytest.raises(ValueError, match="unknown experiment"):
        reg.add_output_policy("ghost", "pol-3")


def test_supersession_not_mutation(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp.jsonl")
    reg.register("e1", "if X then Y", "X", conditions(), "none", revisit_when="curriculum exists")
    reg.record_verdict("e1", "refuted", "no effect at 100 episodes", {})
    reg.register("e2", "if X then Y (with curriculum)", "X+curriculum", conditions(), prior_art=["e1"])
    reg.mark_superseded("e1", "e2")
    old = reg.get("e1")
    assert old.verdict == "refuted"  # verdict untouched
    assert old.superseded_by == "e2"
    assert old.revisit_when == "curriculum exists"


def test_search_dumb_index(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp.jsonl")
    reg.register("e1", "wider FC layer improves win rate", "fc 128->256", conditions(), "none",
                 tags=["architecture"])
    reg.register("e2", "lower LR stabilizes loss", "lr 1e-3->1e-4", conditions(), "none")
    assert [e.experiment_id for e in reg.search("wider fc")] == ["e1"]
    assert [e.experiment_id for e in reg.search("architecture")] == ["e1"]
    assert [e.experiment_id for e in reg.search("LR")] == ["e2"]
    assert reg.search("ecology") == []
