"""exp-010: capacity probe - can the current network even REPRESENT tactics?

A diagnostic, not a loop run: train networks SUPERVISED on solver-labeled
positions, removing the RL signal question entirely. If the current net
cannot fit the training set, capacity/optimization binds. If it fits easily,
capacity is NOT the bottleneck and attention shifts to the training signal.

Run:  python3 experiments/exp_010_capacity_probe.py
"""

import json

import numpy as np
import torch
import torch.nn as nn
from _common import Store, _REPO_ROOT

from harness.agents.dqn import QNet, seed_everything
from harness.tasks.connect4 import Connect4Env
from harness.tasks.connect4_solver import Connect4Solver

SEED = 10
N_POSITIONS = 4000
LABEL_DEPTH = 6
EPOCHS = 150
TRAIN_FRAC = 0.8

HYPOTHESIS = """\
CONTEXT: Nine loop experiments (exp-002..009) moved the ladder only 0.030 ->
0.083; rung-1 win rate plateaus ~0.42. Two candidate bottlenecks remain:
the training SIGNAL (sparse win/lose reward) and network CAPACITY (2 conv
layers x 32 filters + 128-unit FC, ~200k params). This diagnostic isolates
capacity by removing RL entirely: supervised training on solver-labeled
best moves.

HYPOTHESIS: The current architecture (conv_layers 2, channels 32, hidden 128)
has insufficient capacity to represent Connect 4 tactics. Tested against a
larger variant (conv_layers 3, channels 64, hidden 256, ~1M params) on
identical data: 4000 positions from seeded random playouts, each labeled with
the depth-6 solver's best move, 3200 train / 800 held out, 150 epochs of
Adam lr 1e-3 batch 128, cross-entropy on the best-move class.

PREDICTION: If capacity binds, the current net's TRAIN accuracy stays under
90% while the larger net reaches 90%+. If BOTH fit the training set (>=90%),
capacity is NOT the binding constraint at this scale and the bottleneck is
the reward signal (exp-011's territory). Held-out accuracy measures
generalization separately; the capacity question is about TRAIN fit.

DECISION CRITERIA: thresholds pre-registered above; single seed, so ~2-3pp
noise on accuracies - only gaps larger than that count. This experiment
produces no policy and no promotion; its output is a fact about architecture."""


def generate_labeled_positions(rng):
    solver = Connect4Solver(max_depth=LABEL_DEPTH)
    seen, X, y = set(), [], []
    while len(X) < N_POSITIONS:
        env = Connect4Env()
        env.reset()
        while not env.done and len(X) < N_POSITIONS:
            key = (env.board.tobytes(), env.current_player)
            if key not in seen and env.board.any():
                seen.add(key)
                X.append(env.obs().copy())
                y.append(solver.best_move(env.board, env.current_player))
            env.step(int(rng.choice(env.legal_actions())))
    return np.stack(X), np.array(y)


def fit(name, net, Xtr, ytr, Xva, yva):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)
    n = len(Xtr_t)
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        net.train()
        for i in range(0, n, 128):
            idx = perm[i : i + 128]
            loss = nn.functional.cross_entropy(net(Xtr_t[idx]), ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        train_acc = float((net(Xtr_t).argmax(1) == ytr_t).float().mean())
        val_acc = float((net(Xva_t).argmax(1) == yva_t).float().mean())
    params = sum(p.numel() for p in net.parameters())
    print(f"{name}: params={params:,} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
    return {"params": params, "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4)}


def main() -> None:
    store = Store("store")
    store.experiments.register(
        experiment_id="exp-010-capacity-probe",
        hypothesis=HYPOTHESIS,
        changes="diagnostic only - no loop run; compares conv_layers/channels/hidden "
        "2/32/128 vs 3/64/256 supervised on 4000 solver-d6-labeled positions",
        conditions={
            "task_ref": "connect4@v2",
            "benchmark_ref": "supervised-probe-solver-d6-labels",
            "config": {
                "n_positions": N_POSITIONS,
                "label_depth": LABEL_DEPTH,
                "epochs": EPOCHS,
                "train_frac": TRAIN_FRAC,
                "arch_small": "2conv-32ch-128fc",
                "arch_large": "3conv-64ch-256fc",
            },
            "seed": SEED,
        },
        prior_art=["exp-009-big-swing"],
        tags=["diagnostic", "capacity", "architecture", "supervised-probe"],
    )

    seed_everything(SEED)
    rng = np.random.default_rng(SEED)
    print("generating solver-labeled positions...")
    X, y = generate_labeled_positions(rng)
    split = int(len(X) * TRAIN_FRAC)
    results = {
        "small": fit("small 2conv-32ch-128fc", QNet(6, 7, 7, 32, 128, 2), X[:split], y[:split], X[split:], y[split:]),
        "large": fit("large 3conv-64ch-256fc", QNet(6, 7, 7, 64, 256, 3), X[:split], y[:split], X[split:], y[split:]),
    }
    out = _REPO_ROOT / "store" / "telemetry" / "exp-010-capacity-probe.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
