"""Loop integration: one tiny end-to-end run produces an archived,
metadata-rich policy lineage, an experiment record with bidirectional policy
links, and a generatable mastery narrative."""

from harness.agents.dqn import TrainerConfig
from harness.improvement_loop import LoopConfig, Store, run_loop
from harness.mastery_narrative import narrative
from harness.tasks.connect4 import Connect4Task


def tiny_config(experiment_id="exp-smoke", seed=0, iterations=2):
    return LoopConfig(
        experiment_id=experiment_id,
        hypothesis="if the loop runs, it archives a lineage and registry record",
        changes="none — harness integration fixture",
        seed=seed,
        iterations=iterations,
        episodes_per_iteration=15,
        gate_games=10,
        trainer=TrainerConfig(seed=seed, batch_size=16, buffer_capacity=2000),
    )


def test_end_to_end_produces_lineage_and_records(tmp_path):
    store = Store(tmp_path)
    task = Connect4Task(benchmark_games=20)
    summary = run_loop(task, store, tiny_config())

    # A champion exists and is archived.
    champion = store.archive.current_champion("connect4")
    assert champion is not None
    assert summary["champion"] == champion

    # gen-0 + one challenger per iteration were archived, win or lose.
    entries = store.archive.all_entries()
    assert len(entries) == 3  # 1 seed + 2 challengers
    for e in entries:
        assert e["task_ref"] == "connect4@v1"
        assert e["experiment_id"] == "exp-smoke"
        assert e["eval_records"], "every archived policy carries eval records"

    # Task registered; experiment registered with bidirectional policy links.
    assert store.tasks.get("connect4", 1) is not None
    exp = store.experiments.get("exp-smoke")
    assert exp is not None
    assert exp.hypothesis.startswith("if the loop runs")
    assert len(exp.policy_refs["outputs"]) == 3
    assert exp.verdict is None  # loop never auto-records verdicts

    # Every archived policy has an epitaph.
    for e in entries:
        assert store.archive.epitaph(e["policy_id"])

    # Mastery narrative generates from any leaf.
    story = narrative(store, entries[-1]["policy_id"])
    assert "hypothesis" in story
    assert "exp-smoke" in story


def test_loop_resumes_from_archived_champion(tmp_path):
    store = Store(tmp_path)
    task = Connect4Task(benchmark_games=20)
    run_loop(task, store, tiny_config("exp-a", iterations=1))
    first_champion = store.archive.current_champion("connect4")

    # Second experiment must warm-start from (and cite) the incumbent.
    run_loop(task, store, tiny_config("exp-b", iterations=1))
    exp_b = store.experiments.get("exp-b")
    assert exp_b.policy_refs["inputs"] == [first_champion]

    # Archive only ever grew.
    assert len(store.archive.all_entries()) == 3  # (seed + ch) + ch


def test_verdict_flow(tmp_path):
    store = Store(tmp_path)
    task = Connect4Task(benchmark_games=20)
    run_loop(task, store, tiny_config())
    store.experiments.record_verdict(
        "exp-smoke", "supported", "loop closed end-to-end", {"promotions": 0}
    )
    assert store.experiments.get("exp-smoke").verdict == "supported"
