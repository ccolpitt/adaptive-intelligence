"""One-command harness CLI.

  python -m harness run     --store store --experiment-id exp1 --hypothesis "..." ...
  python -m harness report  --store store --policy connect4-pol-00003
  python -m harness verdict --store store --experiment-id exp1 \
      --verdict supported --tldr "..." 
"""

from __future__ import annotations

import argparse
import json

from .agents.dqn import TrainerConfig
from .improvement_loop import LoopConfig, Store, run_loop
from .mastery_narrative import narrative
from .tasks.connect4 import Connect4Task


def main() -> None:
    parser = argparse.ArgumentParser(prog="harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the improvement loop on Connect 4")
    p_run.add_argument("--store", default="store")
    p_run.add_argument("--experiment-id", required=True)
    p_run.add_argument("--hypothesis", required=True)
    p_run.add_argument("--changes", required=True)
    p_run.add_argument("--prior-art", nargs="*", default=None,
                       help="related experiment IDs; omit to assert none exists")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--iterations", type=int, default=5)
    p_run.add_argument("--episodes-per-iteration", type=int, default=200)
    p_run.add_argument("--gate-games", type=int, default=50)
    p_run.add_argument("--benchmark-games", type=int, default=200)

    p_rep = sub.add_parser("report", help="mastery narrative for a policy")
    p_rep.add_argument("--store", default="store")
    p_rep.add_argument("--policy", required=True)

    p_play = sub.add_parser("play-solver", help="play Connect 4 vs the solver in the terminal")
    p_play.add_argument("--depth", type=int, default=8,
                        help="search depth in plies (try 2 vs 10; change in-game with 'd N')")
    p_play.add_argument("--first", choices=["human", "solver"], default="human")

    p_ver = sub.add_parser("verdict", help="record an experiment verdict (once, immutable)")
    p_ver.add_argument("--store", default="store")
    p_ver.add_argument("--experiment-id", required=True)
    p_ver.add_argument("--verdict", choices=["supported", "refuted", "inconclusive"], required=True)
    p_ver.add_argument("--tldr", required=True)
    p_ver.add_argument("--results", default="{}", help="JSON dict of result metrics")

    args = parser.parse_args()

    if args.command == "play-solver":
        from .interactive_play import play_solver

        play_solver(depth=args.depth, human_first=args.first == "human")
        return

    store = Store(args.store)

    if args.command == "run":
        cfg = LoopConfig(
            experiment_id=args.experiment_id,
            hypothesis=args.hypothesis,
            changes=args.changes,
            prior_art=args.prior_art if args.prior_art else "none",
            seed=args.seed,
            iterations=args.iterations,
            episodes_per_iteration=args.episodes_per_iteration,
            gate_games=args.gate_games,
            trainer=TrainerConfig(seed=args.seed),
        )
        task = Connect4Task(benchmark_games=args.benchmark_games)
        summary = run_loop(task, store, cfg)
        print(json.dumps(summary, indent=2))
    elif args.command == "report":
        print(narrative(store, args.policy))
    elif args.command == "verdict":
        store.experiments.record_verdict(
            args.experiment_id, args.verdict, args.tldr, json.loads(args.results)
        )
        print(f"verdict recorded for {args.experiment_id}: {args.verdict}")


if __name__ == "__main__":
    main()
