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
    p_run.add_argument("--gate-games", type=int, default=200)
    p_run.add_argument("--task-version", type=int, choices=[1, 2], default=2,
                       help="2 = solver-ladder benchmark (default), 1 = legacy vs-random")
    p_run.add_argument("--benchmark-games", type=int, default=200,
                       help="games for the v1 legacy benchmark only")
    p_run.add_argument("--frontier-fraction", type=float, default=0.0,
                       help="fraction of training episodes vs the frontier-depth solver")

    p_rep = sub.add_parser("report", help="mastery narrative for a policy")
    p_rep.add_argument("--store", default="store")
    p_rep.add_argument("--policy", required=True)

    p_card = sub.add_parser("card", help="policy performance card (diagnostics, not just win/loss)")
    p_card.add_argument("--store", default="store")
    p_card.add_argument("--policy", required=True)
    p_card.add_argument("--games", type=int, default=100, help="probe games vs seeded random")
    p_card.add_argument("--seed", type=int, default=0)
    p_card.add_argument("--json", action="store_true", help="print JSON instead of text")

    p_play = sub.add_parser("play-solver", help="play Connect 4 vs the solver in the terminal")
    p_play.add_argument("--depth", type=int, default=8,
                        help="search depth in plies (try 2 vs 10; change in-game with 'd N')")
    p_play.add_argument("--first", choices=["human", "solver"], default="human")

    p_playp = sub.add_parser("play", help="play Connect 4 vs an archived policy (default: current champion)")
    p_playp.add_argument("--store", default="store")
    p_playp.add_argument("--policy", default=None, help="policy id (default: current champion)")
    p_playp.add_argument("--first", choices=["human", "policy"], default="human")

    p_exp = sub.add_parser("exp-card", help="visual experiment card (hypothesis, setup, result curves)")
    p_exp.add_argument("--store", default="store")
    p_exp.add_argument("--experiment-id", default=None, help="one experiment; omit with --all")
    p_exp.add_argument("--all", action="store_true", help="render every experiment + index.html")
    p_exp.add_argument("--plots", action="store_true",
                       help="also render a matplotlib PNG dashboard (per-episode curves)")

    p_cp = sub.add_parser("campaign-plot", help="one picture: best score per experiment + champion lineage")
    p_cp.add_argument("--store", default="store")

    p_story = sub.add_parser("story", help="campaign ledger: every lever pulled, expected vs observed")
    p_story.add_argument("--store", default="store")

    p_status = sub.add_parser("status", help="is an experiment running? champion, last activity, pending verdicts")
    p_status.add_argument("--store", default="store")

    p_diag = sub.add_parser("diagnose", help="read a policy's card and emit hypothesis candidates")
    p_diag.add_argument("--store", default="store")
    p_diag.add_argument("--policy", required=True)
    p_diag.add_argument("--games", type=int, default=60)

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
            frontier_opponent_fraction=args.frontier_fraction,
            trainer=TrainerConfig(seed=args.seed),
        )
        if args.task_version == 2:
            from .tasks.connect4_ladder import Connect4LadderTask

            task = Connect4LadderTask()
        else:
            task = Connect4Task(benchmark_games=args.benchmark_games)
        summary = run_loop(task, store, cfg)
        print(json.dumps(summary, indent=2))
    elif args.command == "play":
        from .interactive_play import play_policy

        play_policy(store, args.policy, human_first=args.first == "human")
    elif args.command == "report":
        print(narrative(store, args.policy))
    elif args.command == "card":
        from .performance_card import build_card, save_card

        card = build_card(store, args.policy, games=args.games, seed=args.seed)
        paths = save_card(store, card)
        print(json.dumps(card.to_dict(), indent=2) if args.json else card.render())
        print(f"\nsaved: {paths['txt']}  {paths['json']}")
    elif args.command == "exp-card":
        from .experiment_card import save_all, save_experiment_card

        if args.all:
            index = save_all(store)
            print(f"rendered {len(store.experiments.all_ids())} cards\nindex: {index}")
            if args.plots:
                from .experiment_card import save_matplotlib_dashboard

                for exp_id in store.experiments.all_ids():
                    print(f"  plot: {save_matplotlib_dashboard(store, exp_id)}")
        elif args.experiment_id:
            out = save_experiment_card(store, args.experiment_id)
            print(out["text"])
            print(f"\nsaved: {out['html']}")
            if args.plots:
                from .experiment_card import save_matplotlib_dashboard

                print(f"plot:  {save_matplotlib_dashboard(store, args.experiment_id)}")
        else:
            print("provide --experiment-id or --all")
    elif args.command == "diagnose":
        from .diagnosis import diagnose_run, render_diagnosis
        from .performance_card import build_card, save_card

        card = build_card(store, args.policy, games=args.games)
        save_card(store, card)
        telemetry = {}
        exp_id = store.archive.get_entry(args.policy).get("experiment_id")
        tpath = store.telemetry_dir / f"{exp_id}.json"
        if tpath.exists():
            telemetry = json.loads(tpath.read_text())
        print(card.render())
        print()
        print(render_diagnosis(diagnose_run(card, telemetry)))
    elif args.command == "status":
        import subprocess
        from datetime import datetime, timezone

        from .jsonl_store import read_jsonl

        procs = subprocess.run(
            ["pgrep", "-fl", "experiments/exp"], capture_output=True, text=True
        ).stdout.strip()
        print("experiment processes:")
        print(f"  {procs}" if procs else "  none running")

        events = read_jsonl(store.root / "archive" / "events.jsonl")
        if events:
            last = events[-1]
            at = datetime.fromisoformat(last["at"])
            age_min = (datetime.now(timezone.utc) - at).total_seconds() / 60
            beat = "LIVE (an iteration completed recently)" if age_min < 20 else "quiet"
            print(f"archive heartbeat: last event {age_min:.0f} min ago "
                  f"({last.get('kind')} {last.get('policy_id', '')}) — {beat}")

        champ = store.archive.current_champion("connect4")
        if champ:
            entry = store.archive.get_entry(champ)
            _, meta = store.archive.load_policy(champ)
            bench = next((r["score"] for r in reversed(meta.get("eval_records", []))
                          if "benchmark" in str(r.get("benchmark_ref"))), None)
            print(f"champion: {champ} (gen {entry.get('generation')}, "
                  f"bench {bench:.3f})" if bench is not None else f"champion: {champ}")

        pending = [i for i in store.experiments.all_ids()
                   if store.experiments.get(i).verdict is None]
        print(f"experiments awaiting verdict: {pending if pending else 'none'}")
    elif args.command == "campaign-plot":
        from .experiment_card import save_campaign_plot

        print(f"saved: {save_campaign_plot(store)}")
    elif args.command == "story":
        from .experiment_card import render_story, save_story

        print(render_story(store))
        print(f"\nsaved: {save_story(store)}")
    elif args.command == "verdict":
        store.experiments.record_verdict(
            args.experiment_id, args.verdict, args.tldr, json.loads(args.results)
        )
        print(f"verdict recorded for {args.experiment_id}: {args.verdict}")


if __name__ == "__main__":
    main()
