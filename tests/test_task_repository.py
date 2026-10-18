"""The task repository: tasks are editable commentary, evaluation definitions
are frozen contracts."""

import json

import pytest

from harness.task_repository import (
    Evaluation,
    Task,
    TaskRepository,
    connect4_evaluation_v1,
    seed_repository,
)


@pytest.fixture
def repo(tmp_path):
    return TaskRepository(tmp_path / "tasks")


def test_register_and_read_a_task(repo):
    repo.register_task(Task(task_id="t1", name="Test task", description="d",
                            status="active"))
    task = repo.get_task("t1")
    assert task.name == "Test task" and task.status == "active"
    assert repo.get_task("missing") is None


def test_task_description_is_editable_but_created_at_is_kept(repo):
    repo.register_task(Task(task_id="t1", name="A", description="first"))
    created = repo.get_task("t1").created_at
    repo.register_task(Task(task_id="t1", name="A", description="revised"))
    assert repo.get_task("t1").description == "revised"
    assert repo.get_task("t1").created_at == created


def test_evaluation_requires_its_task(repo):
    with pytest.raises(ValueError, match="register the task"):
        repo.register_evaluation(connect4_evaluation_v1())


def test_publishing_an_evaluation_sets_it_current(repo):
    repo.register_task(Task(task_id="connect4", name="c4", description="d"))
    repo.register_evaluation(connect4_evaluation_v1())
    assert repo.get_task("connect4").current_evaluation == 1
    ev = repo.get_evaluation("connect4")
    assert ev.ref == "connect4-eval@v1"
    assert ev.scorer_kind == "exact_opponent"
    assert ev.mastery["per_opponent"] == 0.90
    assert "reads the position once" in ev.inference_rule


def test_republishing_the_same_definition_is_a_no_op(repo):
    repo.register_task(Task(task_id="connect4", name="c4", description="d"))
    repo.register_evaluation(connect4_evaluation_v1())
    repo.register_evaluation(connect4_evaluation_v1())  # must not raise
    assert len(repo.list_evaluations("connect4")) == 1


def test_changing_a_published_definition_is_refused(repo):
    """The core guarantee: a score keeps its meaning forever."""
    repo.register_task(Task(task_id="connect4", name="c4", description="d"))
    repo.register_evaluation(connect4_evaluation_v1())
    altered = connect4_evaluation_v1()
    altered.mastery = {"per_opponent": 0.50}  # moving the goalposts
    with pytest.raises(ValueError, match="already published with a different"):
        repo.register_evaluation(altered)


def test_a_new_version_may_differ_and_both_remain_readable(repo):
    repo.register_task(Task(task_id="connect4", name="c4", description="d"))
    repo.register_evaluation(connect4_evaluation_v1())
    v2 = connect4_evaluation_v1()
    v2.version = 2
    v2.procedure = dict(v2.procedure, games_per_opponent=112, games_total=672)
    v2.noise_floor = 0.045
    repo.register_evaluation(v2)
    assert repo.get_task("connect4").current_evaluation == 2
    assert repo.get_evaluation("connect4", 1).procedure["games_total"] == 168
    assert repo.get_evaluation("connect4", 2).procedure["games_total"] == 672
    assert [e.version for e in repo.list_evaluations("connect4")] == [1, 2]


def test_rejects_unknown_scorer_kind_and_status(repo):
    repo.register_task(Task(task_id="t", name="t", description="d"))
    bad = connect4_evaluation_v1()
    bad.task_id, bad.scorer_kind = "t", "vibes"
    with pytest.raises(ValueError, match="scorer_kind"):
        repo.register_evaluation(bad)
    with pytest.raises(ValueError, match="status"):
        repo.register_task(Task(task_id="t2", name="t", description="d", status="maybe"))


def test_seeded_repository_declares_the_task_ladder(repo):
    seed_repository(repo)
    tasks = {t.task_id: t for t in repo.list_tasks()}
    assert tasks["connect4"].status == "active"
    # Future tasks are declared before they are possible, on purpose.
    for future in ("connect4-variants", "chess", "flight-booking",
                   "restaurant-reservation", "online-trading"):
        assert tasks[future].status == "declared"
    # Only the active task has a published way to score it.
    assert repo.get_evaluation("connect4") is not None
    assert repo.get_evaluation("flight-booking") is None


def test_everything_on_disk_is_plain_json(repo, tmp_path):
    seed_repository(repo)
    blob = json.loads((tmp_path / "tasks" / "connect4" / "evaluations" / "eval-v1.json").read_text())
    assert blob["ref"] == "connect4-eval@v1"
    assert blob["scoring"]["per_game"].startswith("win = 1")
