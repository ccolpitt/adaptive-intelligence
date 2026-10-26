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

    p_playp = sub.add_parser("play", help="play Connect 4 vs an archived policy (default: latest champion)")
    p_playp.add_argument("--store", default="store")
    p_playp.add_argument("--policy", default=None, help="policy id (default: latest champion)")
    p_playp.add_argument("--lineage", default=None,
                         help="play that lineage's champion, e.g. connect4-L5")
    p_playp.add_argument("--first", choices=["human", "policy"], default="human")
    p_playp.add_argument("--lookahead", type=int, default=0,
                         help="distinct positions the policy may examine per move "
                              "(0 = how it is scored; try 200 for a real fight)")

    p_pols = sub.add_parser("policies", help="list archived policies you can play")
    p_pols.add_argument("--store", default="store")
    p_pols.add_argument("--lineage", default=None, help="only this lineage")
    p_pols.add_argument("--all", action="store_true",
                        help="every policy, not just the ones that were promoted")
    p_pols.add_argument("--limit", type=int, default=40)

    p_ref = sub.add_parser("reference",
                           help="what a benchmark score is worth, against players of known strength")
    p_ref.add_argument("--store", default="store")
    p_ref.add_argument("--measure", action="store_true",
                       help="re-measure the reference players (minutes); otherwise read the stored board")
    p_ref.add_argument("--score", type=float, default=None,
                       help="place a specific score on the scale")
    p_ref.add_argument("--policy", default=None,
                       help="place an archived policy's recorded score on the scale")

    p_exp = sub.add_parser("exp-card", help="visual experiment card (hypothesis, setup, result curves)")
    p_exp.add_argument("--store", default="store")
    p_exp.add_argument("--experiment-id", default=None, help="one experiment; omit with --all")
    p_exp.add_argument("--all", action="store_true", help="render every experiment + index.html")
    p_exp.add_argument("--plots", action="store_true",
                       help="also render a matplotlib PNG dashboard (per-episode curves)")

    p_rev = sub.add_parser("review-checks", help="should the expectation checks be trimmed or extended?")
    p_rev.add_argument("--store", default="store")
    p_rev.add_argument("--lineage", default=None, help="default: every lineage")

    p_tac = sub.add_parser("tactics", help="does it take wins and block losses? the triage view")
    p_tac.add_argument("--store", default="store")
    p_tac.add_argument("--lineage", required=True)

    p_geo = sub.add_parser("tactics-by-geometry",
                           help="split those rates by line direction and by where the positions came from")
    p_geo.add_argument("--store", default="store")
    p_geo.add_argument("--policy", default=None, help="default: the latest champion")
    p_geo.add_argument("--games-off", type=int, default=400,
                       help="games of random play (positions the policy never trains on)")
    p_geo.add_argument("--games-in", type=int, default=60,
                       help="games of the policy's own play (slower: it searches)")
    p_geo.add_argument("--budget", type=int, default=200)
    p_geo.add_argument("--seed", type=int, default=0)
    p_geo.add_argument("--json", action="store_true")

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

        play_policy(store, args.policy, human_first=args.first == "human",
                    lookahead=args.lookahead, lineage=args.lineage)
    elif args.command == "policies":
        from .interactive_play import list_policies

        list_policies(store, lineage=args.lineage,
                      champions_only=not args.all, limit=args.limit)
    elif args.command == "reference":
        from .benchmark_reference import (
            ReferenceBoardStore, measure_reference_board, render,
        )
        from .tasks.connect4_ladder import SolverLadderBenchmark

        benchmark = SolverLadderBenchmark()
        boards = ReferenceBoardStore(store.root / "reference")
        board = boards.load(benchmark.ref)
        if args.measure or board is None:
            if board is not None:
                print("re-measuring the reference players; this takes minutes")
            else:
                print("no stored reference board; measuring, which takes minutes")
            board = measure_reference_board(benchmark)
            print(f"\nstored: {boards.save(board)}\n")
        print(render(board))

        score = args.score
        label = "that score"
        if args.policy:
            _, meta = store.archive.load_policy(args.policy)
            score = next((r["score"] for r in reversed(meta.get("eval_records", []))
                          if "benchmark" in str(r.get("benchmark_ref", ""))
                          and "lookahead" not in str(r.get("benchmark_ref", ""))), None)
            label = args.policy
            if score is None:
                print(f"\n{args.policy} has no recorded benchmark score")
        if score is not None:
            print("\n" + board.describe(score, label))
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

        from .cycle_record import CycleStore

        # The process table is the ground truth for "is something running".
        # A training run writes nothing for minutes at a time while it plays
        # its games, so file timestamps alone cannot answer this.
        procs = [line for line in subprocess.run(
            ["pgrep", "-fl", "experiments/exp"], capture_output=True, text=True
        ).stdout.strip().splitlines() if "caffeinate" not in line]

        if procs:
            print("RUNNING:")
            for line in procs:
                pid, _, cmd = line.partition(" ")
                print(f"  pid {pid}  {cmd.replace('python3 -u ', '')}")
        else:
            print("RUNNING: nothing — no experiment process is alive")

        # Progress per line of policies, newest cycle first. This is the
        # useful view: a run's champion belongs to ITS lineage, so reading one
        # global champion reports whichever line happened to write last.
        cycles_store = CycleStore(store.root / "cycles")
        lineage_dirs = sorted((store.root / "cycles").glob("*"))
        if lineage_dirs:
            print("\nlines of policies:")
        for directory in lineage_dirs:
            if not directory.is_dir():
                continue
            records = cycles_store.load_all(directory.name)
            if not records:
                continue
            last = records[-1]
            age_min = (datetime.now(timezone.utc)
                       - datetime.fromisoformat(last.created_at)).total_seconds() / 60
            best = max((r.evaluation.get("score") or 0) for r in records)
            champion_score = next(
                (r.evaluation.get("score") for r in reversed(records) if r.promoted), None)
            # A cycle takes minutes, so "no new cycle yet" is normal while a
            # run is alive; it only means something is wrong if nothing is.
            # An experiment id is hyphenated and its script file is not, so
            # the comparison has to ignore the difference — otherwise a live
            # run is reported as having just stopped.
            def _flat(text: str) -> str:
                return text.replace("-", "").replace("_", "").lower()

            alive = any(_flat(directory.name) in _flat(p)
                        or _flat(last.experiment_id or "\0") in _flat(p)
                        for p in procs)
            if alive:
                state = f"training (last cycle finished {age_min:.0f} min ago)"
            elif age_min < 30:
                state = f"just stopped ({age_min:.0f} min ago)"
            else:
                state = f"idle ({age_min / 60:.1f} h since last cycle)"
            print(f"  {directory.name:16s} cycle {last.cycle:>3}  "
                  f"{last.cumulative_games:>7,} games  "
                  f"best {best:.3f}  champion "
                  f"{(f'{champion_score:.3f}' if champion_score is not None else '—')}"
                  f"  [{state}]")
            # Findings are recomputed rather than read from the records. A
            # check that was wrong when a cycle ran would otherwise keep
            # reporting a defect that has since been fixed (R-006, R-007).
            from .sanity import CONTRADICTION, recheck, superseded_findings

            faults, superseded = [], []
            for i, r in enumerate(records):
                faults += [f for f in recheck(r, records[:i])
                           if f.severity == CONTRADICTION]
                superseded += superseded_findings(r, records[:i])
            if faults:
                checks = sorted({f.check for f in faults})
                print(f"    {len(faults)} CONTRADICTION(S) — investigate "
                      f"before trusting these numbers (learning rule 11): "
                      f"{'; '.join(checks)}")
            if superseded:
                stale = sorted({f.get("check", "?") for f in superseded})
                print(f"    {len(superseded)} finding(s) recorded during the "
                      f"run no longer hold under the current checks "
                      f"(not faults): {'; '.join(stale)}")

        pending = [i for i in store.experiments.all_ids()
                   if store.experiments.get(i).verdict is None]
        if pending:
            print(f"\nverdicts owed: {', '.join(pending)}")
            print("  (an experiment without a verdict is an unfinished thought)")
        else:
            print("\nverdicts owed: none")
    elif args.command == "review-checks":
        from .cycle_record import CycleStore
        from .sanity import render_review, review_checks

        cycles_store = CycleStore(store.root / "cycles")
        if args.lineage:
            records = cycles_store.load_all(args.lineage)
        else:
            records = [r for d in sorted((store.root / "cycles").glob("*"))
                       if d.is_dir() for r in cycles_store.load_all(d.name)]
        print(render_review(review_checks(records)))
    elif args.command == "tactics-by-geometry":
        from .interactive_play import find_champion
        from .tactical_probe import probe_policy, render

        policy_id = args.policy or find_champion(store)
        if policy_id is None:
            print("no policy to probe; try: python -m harness policies --all")
            return
        module, _ = store.archive.load_policy(policy_id)
        breakdowns = probe_policy(module, games_off=args.games_off,
                                  games_in=args.games_in, budget=args.budget,
                                  seed=args.seed)
        if args.json:
            print(json.dumps([b.to_dict() for b in breakdowns], indent=2))
        else:
            print(render(breakdowns, label=policy_id))
    elif args.command == "tactics":
        from .cycle_record import CycleStore

        records = CycleStore(store.root / "cycles").load_all(args.lineage)
        if not records:
            print(f"no cycles recorded for {args.lineage}")
            return
        print(f"TACTICAL COMPETENCE — {args.lineage}")
        print("A policy that beats a 6-move perfect-play opponent should take an")
        print("available win and block a forced loss essentially every time. Where")
        print("it does not, the gap says the learning path is losing something the")
        print("search already found — which makes these the best triage numbers we")
        print("have. Measured on the policy alone, with no lookahead.")
        print()
        print(f"{'cycle':>6} {'games':>8} {'score':>7} {'takes win':>10} "
              f"{'blocks loss':>12}  flag")
        for r in records:
            take = r.probes.get("take_the_win_rate")
            block = r.probes.get("block_the_loss_rate")
            score = r.evaluation.get("score")
            flag = ""
            if isinstance(score, (int, float)) and score > 0.4:
                if isinstance(take, (int, float)) and take < 0.9:
                    flag += " win-blind"
                if isinstance(block, (int, float)) and block < 0.9:
                    flag += " block-blind"
            print(f"{r.cycle:>6} {r.cumulative_games:>8,} "
                  f"{(f'{score:.3f}' if isinstance(score, (int, float)) else '—'):>7} "
                  f"{(f'{take:.2f}' if isinstance(take, (int, float)) else '—'):>10} "
                  f"{(f'{block:.2f}' if isinstance(block, (int, float)) else '—'):>12}"
                  f"  {flag}")
        takes = [r.probes.get("take_the_win_rate") for r in records[-5:]]
        blocks = [r.probes.get("block_the_loss_rate") for r in records[-5:]]
        takes = [t for t in takes if isinstance(t, (int, float))]
        blocks = [b for b in blocks if isinstance(b, (int, float))]
        if takes and blocks:
            import statistics

            t_now, b_now = statistics.mean(takes), statistics.mean(blocks)
            print(f"\nlast five cycles: takes a win {t_now:.0%}, blocks a loss {b_now:.0%}")
            if t_now < 0.9 or b_now < 0.9:
                print("VERDICT: tactically incomplete. The search sees these positions,")
                print("so what it found is not fully reaching the weights. Candidates:")
                print("  - raise the search budget (a coarse visit distribution is a")
                print("    coarse training target)")
                print("  - weight the move-head loss toward decided positions")
                print("  - oversample decision-critical positions when training")
                print("    (TrainerConfig.tactical_fraction, built and never tested)")
            else:
                print("VERDICT: tactically sound.")
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
