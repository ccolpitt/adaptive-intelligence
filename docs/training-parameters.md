# Every Tunable Parameter (as of 2026-10-04)

Plain-language reference for what can be changed when training a policy,
where it lives in the code, and what it is set to. "Baseline" = the default
value. "Best-known setup" = the four changes that together produced the
program's best training results (shaping + cold start + both players' data
+ varied opponents) — what I had been calling "the recipe".

## A. The network itself (`harness/agents/dqn.py`, class `QNet`)

| Parameter | Default | Plain meaning |
|---|---|---|
| input | 2 planes × 6 rows × 7 cols | my pieces / opponent's pieces, always from the mover's view |
| `conv_layers` | 2 | how many 3×3 pattern-detector layers (each layer widens the view by 2 cells: 2 layers see 5×5, 3 see 7×7) |
| `channels` | 32 | pattern detectors per conv layer |
| `hidden` | 128 | neurons in the one fully-connected layer that sees the whole board |
| output | 7 | one estimated value per column (the Q values) |
| activation | LeakyReLU(0.01) | fixed; not a lever today |
| weight init | Kaiming normal | fixed; standard for ReLU-family nets |
| total parameters | ≈183,000 | small by design |

## B. How it learns (`TrainerConfig`)

| Parameter | Default | Plain meaning |
|---|---|---|
| `lr` | 0.001 | learning rate: how big each weight update is (Adam optimizer) |
| `weight_decay` | 0.0001 | gentle pull of all weights toward zero (regularization) |
| `gamma` | 0.99 | discount: how much a result N moves away is worth today (0.99^N) |
| `batch_size` | 64 | transitions per gradient step |
| `train_steps_per_episode` | 2 | gradient steps after every game |
| `buffer_capacity` | 20,000 (50,000 in best-known setup) | how many past transitions are remembered for replay |
| `target_update_every` | 50 games | how often the "target network" copy is refreshed (stabilizes learning) |
| gradient clip | 1.0 (hard-coded) | caps the size of any single update |
| loss | MSE (hard-coded) | squared error between Q and its target |
| `eps_start` / `eps_end` / `eps_decay` | 0.5 / 0.1 / 0.999 | exploration: chance of a random move, decaying per game |
| `seed` | 0 | starting point for ALL randomness (weights, replay sampling, exploration, random opponent) |

## C. What it learns FROM (`TrainerConfig`, each default off)

| Parameter | Default | Plain meaning | Evidence |
|---|---|---|---|
| `store_opponent_transitions` | off (on in best-known) | also store the opponent's moves as training examples (2× data/game) | needed; multiplies with diverse opponents |
| `mirror_augmentation` | off | add the left-right mirror of every transition | no gain (exp-005) |
| `terminal_fraction` | 0.0 | guarantee this share of each batch is game-ending moves | no gain alone (exp-007) |
| `tactical_fraction` | 0.0 | guarantee this share is "win available / must block" positions | untested |
| `tactical_shaping` / `shaping_penalty` | off / 0.5 (on in best-known) | −0.5 immediately for skipping a win or failing a block (training only) | needs cold start + rich data |
| `mc_targets` | off | label every move with the discounted final outcome instead of 0 | hurt (exp-019) |
| `train_search_depth` | 0 | during TRAINING games, the net looks k moves ahead before choosing | no gain at k=2 (exp-022 A) |
| `search_targets` | off | train toward the looked-ahead value instead of the one-step estimate | hurt; scale inflation (exp-022 B) |

## D. The improvement loop around it (`LoopConfig`, `harness/improvement_loop.py`)

| Parameter | Default | Plain meaning |
|---|---|---|
| `iterations` | 5 (experiments use 15–30) | train→evaluate→gate cycles per experiment |
| `episodes_per_iteration` | 200 (experiments use 300) | training games per cycle |
| `cold_start` | off (on in best-known) | start the challenger from random weights instead of copying the champion |
| `diverse_opponents` | off (on in best-known) | training games rotate champion / champion / scripted one-step-win / random |
| `frontier_opponent_fraction` | 0.0 | share of training games vs the solver at the champion's frontier depth (no gain, exp-003) |
| `gate_games` | 200 | head-to-head games challenger vs champion per cycle |
| `gate_temperature` | 0.3 | small move randomness in head-to-head so 200 games aren't 2 repeated games |
| `gate_rule` | `strict` | promotion rule: beat champion AND improve ladder (ADR-007). `either`/`double` kept for history |
| `z_promote` / `z_regress` | 1.645 | statistical confidence (one-sided 95%) for "beats" / "regressed" |
| `plateau_patience` | 8 | **patience** = how many consecutive cycles the best ladder score may fail to improve before we call it stuck |
| `plateau_min_delta` | 0.02 | an improvement smaller than this doesn't count (it's benchmark noise) |
| `plateau_min_iterations` | 12 | never judge "stuck" before this many cycles (cold starts are flat for ~8) |
| `plateau_action` | `flag` (`stop` in experiments) | on plateau: just note it, or end the run and save the compute |
| `auto_card` / `auto_card_games` | on / 60 | after the run, probe the best challenger and write a diagnosis |
| epsilon bump on promotion | +0.1 (hard-coded) | re-explore against the new champion |

## E. How it is judged (fixed — these define the yardstick, change = new benchmark version)

| Component | Setting | Plain meaning |
|---|---|---|
| ladder opponents | negamax (minimax+alpha-beta) at depths 1,2,3,4,5,6 | perfect-play-to-depth-k opponents |
| games per depth | 28 = 14 fixed openings × both seats | identical every time, forever |
| total benchmark games | 168 | |
| per-depth score | wins + ½ draws, / 28 | |
| **ladder score** | average of the six per-depth scores | the headline number (0.0–1.0) |
| mastery | ≥ 0.90 at every depth | |
| frontier depth | shallowest depth scored < 0.90 | "the next opponent to beat" |
| evaluation inference | **naked network, no search** (ADR-006) | one forward pass, pick the largest Q |
| noise floor | ≈ 0.09 at 168 games | differences smaller than this are not distinguishable in one run |
