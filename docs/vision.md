# Adaptive, Sample-Efficient Learning: A Framework for an Evolving Intelligence System

*Revision 2 — restructured from the original essay. The original argument is preserved;
critiques, verdicts, and research references have been folded in. This is the canonical
framework document for this repository.*

---

## 1. Motivation: The World Is Non-Stationary

Today, deployed artificial neural networks are static. We formulate learning as fitting the
weights of a network to minimize loss against a training and test dataset — an image
labeling set, a DQN replay buffer, next-token prediction during pre-training. Whatever
the domain, the common characteristic is that the network is frozen after training.

Reality is not frozen. Self-replicating systems abide by evolutionary forces that change
the environment over time. A strategy that worked in the past usually stops working:
penicillin resistance, financial arbitrage decay, technological disruption, the rise and
fall of governance systems, music styles, business models. Evolution stems from (a)
random variation in replication and (b) differential reproductive rates that favor
adaptive variants. We take it as self-evident that systems must adapt or lapse into
irrelevance.

For designed intelligence, this means: the environment will evolve; a problem relevant
today will be trivial tomorrow; a problem-solving strategy that works today will fail
tomorrow. Adapting may require updating not just network weights but the architecture
itself.

The most adaptive intelligence in existence is the human brain. Its gross structure is
fixed within a lifetime — the product of millions of years of evolution — but its
connections are plastic: the strength of connections, which neurons connect, and how
much neural real estate is allocated to a task all change with use. Structure follows
capability and adapts to environmental demand.

Observing how biological intelligence adapts across timescales gives us the layers of
the framework:

1. **Evolutionary timescale** → new neural architectures (*architectural evolution*)
2. **Cultural timescale** → new curricula and training methods (*pedagogical evolution*)
3. **Lifetime timescale** → changing connection strengths, growing and pruning
   connections (*continual plasticity*)

To these three, this revision adds two more components that the original essay treated
implicitly or not at all — and one of them turns out to be the foundation everything
else stands on.

---

## 2. The Five Foundational Components

### Component 0: Evaluation and Measurement — the foundation

*(New in this revision. Absent from the original, and its absence was the document's
biggest flaw.)*

Every other component in this framework depends on answering one question reliably:
**"is this variant better?"** Better architecture, better curriculum, better plasticity
rule, which policy survives. If measurement is broken, every loop above it optimizes
noise.

The failure mode to design against is **self-referential measurement**. A self-play
improvement loop that only measures challengers against its own champions can promote
thirty-two generations in a row while going nowhere in absolute terms — strategy spaces
are non-transitive (rock-paper-scissors dynamics), so self-play can cycle rather than
climb (Balduzzi et al., 2019). AlphaStar needed a league of diverse exploiter agents,
not just "latest vs. latest," for exactly this reason (Vinyals et al., 2019). We
observed this failure firsthand in the connect4-rl precursor project: a v32 champion
with no idea of its absolute strength, because nothing ever measured it against an
external yardstick.

**Principles:**

- **Every task must carry at least one fixed, external, never-changing benchmark.**
  For Connect 4 this is free: the game is solved (Allis, 1988; Allen, 1989) and a
  minimax/perfect solver gives an absolute scale. For unsolved domains, use frozen
  reference opponents, held-out test suites, or fixed scripted baselines — anything
  that does not move when the population moves.
- **Relative metrics (Elo, tournament rank, champion win-rate) are necessary but never
  sufficient.** They order the population; they do not locate it on an absolute scale.
- **The transfer hypothesis gets a defined metric.** The central claim of this whole
  program (Section 3) is measured as sample efficiency: episodes-to-threshold on task B
  with and without prior training on task A, fixed seeds, matched compute. Standard
  practice in the continual/transfer learning literature (Taylor & Stone, 2009).
- **Negative results are recorded with the same care as positive ones.** The experiment
  registry is append-only.

**Verdict: build first. Nothing else is interpretable without it.**

### Component 1: Architectural Evolution — pursue as diagnosis-driven design; undirected search last

The original essay proposed architecture evolution as the outer loop, with each
generation learning from the performance of the last, and a "tabula rasa protocol"
expressive enough to span all architectures. Honest assessment after engaging with the
literature:

**This is a real field with sobering results.** Neural Architecture Search has been
explored intensively (Zoph & Le, 2017; Real et al., 2019 "Regularized Evolution";
Elsken et al., 2019 survey). Neuroevolution of topologies goes back further — NEAT
(Stanley & Miikkulainen, 2002) and HyperNEAT evolve network structure directly but have
never scaled beyond small networks. AutoML-Zero (Real et al., 2020) evolved learning
algorithms from primitives — fascinating, and brutally compute-hungry. The field's
uncomfortable headline finding is the Bitter Lesson (Sutton, 2019): scaling a good
fixed architecture with more compute and data beats clever search for novel structure,
almost everywhere it's been tried.

**The compute economics are the binding constraint.** Evaluating one architecture
candidate costs one full training run. Evolution needs populations times generations.
On a single machine this yields a few dozen candidate evaluations per week — that is a
slow grid search, not open-ended evolution. The connect4-rl experiments made this
concrete: one architecture variant (wider FC layer) took a full 5k-episode run to
evaluate and lost to the baseline.

**The search space design is itself the hard research problem.** An "infinitely
expressive" encoding makes search intractable; a narrow one reduces to hyperparameter
tuning. There is no known free lunch here.

**Why did automated search fail where humans succeeded?** Not because humans had
deeper math (attention is three matrices and a softmax) or neuroscience (convolutions
are the exception: Hubel & Wiesel → Fukushima, 1980 → LeCun). The dominant pattern in
real breakthroughs is **diagnosis of a specific failure followed by a targeted fix**:

- *Attention*: Bahdanau et al. (2014) diagnosed the seq2seq fixed-vector bottleneck
  and let the decoder attend over all encoder states; Vaswani et al. (2017) diagnosed
  that recurrence blocks parallelism and long-range gradient flow, and removed it.
- *ResNets*: He et al. (2016) diagnosed that deeper nets trained worse *on training
  data* — an optimization pathology — and made identity the default via skips.
- *LSTM*: Hochreiter analyzed vanishing gradients mathematically, then designed gating
  to protect the error signal (Hochreiter & Schmidhuber, 1997).
- *Mixture-of-Experts*: diagnosed that capacity scaling forces compute scaling; fixed
  it with sparse specialization (Shazeer et al., 2017; Fedus et al., 2022).

What humans possessed was (1) rich diagnostic signal about *how* training failed, not
just a scalar fitness; (2) causal reasoning from symptom to mechanism; (3) breadth of
analogy. Classic NAS/neuroevolution had none of these: uninformed mutation, one-bit
feedback per expensive training run, human-bounded primitive sets. That is why it
failed — not because machine-discovered architecture is impossible.

**The bet this project makes: diagnosis-driven architecture evolution.** Replace the
random mutation operator with an informed proposer (LLM/agent + human) that reads
training diagnostics and the literature, proposes a targeted change, and registers it
as a falsifiable experiment. This is where the field is heading: FunSearch
(Romera-Paredes et al., 2024) and AlphaEvolve (Novikov et al., 2025) replace random
mutation with LLM-guided proposal and have produced genuinely new algorithms. The
connect4-rl precursor already ran this loop at the hyperparameter level (its
experiment registry entries follow diagnosis → hypothesis → change → result); this
project extends the same loop to architecture. Honesty requires noting that
machine-discovered *architecture* results to date are incremental, not
attention-level; the realistic win is domain-appropriate fixes, measured.

**Tenets for architectural change** (every proposed change must satisfy all):

1. **Diagnosed, not divined.** The change responds to a specific, instrumented failure
   in a previous run — evidence, not vibes. This makes diagnostics (Component 0)
   load-bearing: the instrumentation must be able to *disambiguate* candidate causes.
2. **Scales with compute** (Bitter Lesson-compliant). No fix that only works small.
3. **Names its enrichment.** State what new information routing the change enables —
   convolutions let spatial neighbors enrich each other; attention lets tokens enrich
   each other by content; memory lets the past enrich the present; MoE enriches via
   specialization. A change that cannot answer "what can now inform what?" is suspect.
4. **Makes a falsifiable prediction** about which diagnostic improves, registered
   before the run.
5. **One change at a time** (framework principle 3).

**Hand-designed architecture experiments are welcome at any phase** — they are
ordinary registered experiments, not Phase 5 work. Example queued: a Thousand
Brains-inspired cortical-column experiment (Hawkins, 2021) — N small parallel networks,
each observing a different transformation of the state, voting on the value/policy,
versus a monolithic net at matched parameter count. Nearest ML relatives: capsule
routing-by-agreement (Sabour et al., 2017), ensembles, and (structurally, not
philosophically) MoE. Numenta's open-source Monty framework implements the theory
directly.

**Verdict: diagnosis-driven evolution is the program; undirected population search is
demoted to final phase, if ever.** When automated search is used at all, constrain it
to a parameterized grammar (depth, width, block types, skip connections) rather than
free-form topology. Population-Based Training (Jaderberg et al., 2017) is the
pragmatic template — it evolves hyperparameters and warm-started weights concurrently
with training and is proven at realistic scale.

### Component 2: Pedagogical Evolution — the strongest component, promoted to first research loop

Over cultural timescales, humans changed what we teach and how: oral tradition, rote
memorization, Aristotle-as-science, and eventually a science of pedagogy. The values
instilled changed too — Spartan discipline, collectivism, individual liberty. The
machine-learning analogue: evolve the training task mixture, the time share per task,
the ordering, and potentially the update rule itself.

**This is the most tractable and best-supported component.** Curriculum learning is
established (Bengio et al., 2009). Automatic curricula are an active, successful field:
POET co-evolves environments and agents (Wang et al., 2019); PAIRED generates
environments adversarially at the frontier of agent capability (Dennis et al., 2020);
Prioritized Level Replay samples training levels by learning potential (Jiang et al.,
2021); learning-progress-based curricula have a long lineage in developmental robotics
(Oudeyer et al., 2007). Curriculum experiments are also *cheap*: same architecture,
same compute, just reorder and reweight the data. That makes this the highest
information-per-dollar loop in the framework.

**A correction from the original essay.** The earlier draft claimed "an architecture
that doesn't support intrinsic curiosity can't implement temporal difference learning."
That conflated three layers: TD learning is an algorithm property, curiosity is a
reward-design choice (Pathak et al., 2017; Burda et al., 2018), and neither is an
architecture property. The underlying instinct is correct and important — **task
difficulty must track agent capability** (the zone of proximal development, in
Vygotsky's terms; the "impedance match" in the original's terms) — but it is a
statement about curriculum-capability fit, which is measurable, not about
architecture, which was not.

**Verdict: pursue first among the research loops. Cheapest, best literature, directly
serves the transfer hypothesis.**

### Component 3: Continual Plasticity — worth pursuing, with eyes open about the real obstacle

Within a lifetime, brains adapt: connection strengths change, new connections grow,
unused ones are pruned. The proposal: make both the building blocks and the wiring of a
network dynamic, so it can keep learning and generalize to new problems without
wholesale retraining.

**The central obstacle has a name the original essay never used: catastrophic
forgetting.** The essay observed that in humans "old capabilities typically do not
wither." That is precisely what *fails* in neural networks — training on task B
overwrites task A — and it has been the field's central obstacle for over three decades
(McCloskey & Cohen, 1989; French, 1999). The mitigation literature is rich: Elastic
Weight Consolidation penalizes movement of weights important to old tasks (Kirkpatrick
et al., 2017); Progressive Networks freeze old columns and grow new ones (Rusu et al.,
2016); PackNet partitions capacity by pruning (Mallya & Lazebnik, 2018); rehearsal
methods replay old data. Any work on this component starts from this literature or
repeats its failures.

**The second, newer obstacle: loss of plasticity.** Networks trained continually don't
just forget — they progressively *lose the ability to learn anything new*, as neurons
saturate and die (Dohare et al., 2024, *Nature* — "Loss of plasticity in deep continual
learning"; Lyle et al., 2023). Their remedy, continual backprop, reinitializes dormant
neurons. Notably, a neuron-reinitialization mechanism was already prototyped in the
connect4-rl precursor — this thread continues that work.

**A reframing of the mechanics.** The original essay framed the challenge as PyTorch
engineering: make blocks and connections dynamic. The mechanics are the easy part —
masks, sparse layers, and modular routing all exist. The hard part is **credit
assignment for structure**: what signal says *which* connection to grow? Gradients
don't exist for connections that don't exist. The known approaches are instructive:
RigL grows connections where gradient magnitude is largest (Evci et al., 2020); the
Lottery Ticket Hypothesis shows sparse trainable subnetworks exist inside dense ones
(Frankle & Carbin, 2019). Growth guided by gradient signal is the practical starting
point.

**Verdict: worth pursuing, and unusually well-suited to a single-machine budget** —
forgetting and plasticity-loss experiments run on small networks. Sequence it after
the loop and measurement exist, because "did plasticity help?" is unanswerable without
Component 0.

### Component 4: Population Selection and End of Life — the harness solves this; the ecology does not need to be built

No living thing lasts forever, and death makes room for new forms. The original essay
proposed limited resources, competition, and a conservation law — a simulated ecology
in which termination is an outcome.

**This revision separates two ideas the original conflated, because they differ in cost
by orders of magnitude:**

1. **Selection pressure and turnover** — necessary, and *nearly free*. A
   champion-challenger ladder already kills losers implicitly. A league with
   promotion/relegation, where losing policies stop receiving compute, delivers the
   full evolutionary function of death. The loop harness (Phase 0 of the roadmap)
   provides this on day one.

2. **A multi-agent resource-competition ecology** — a large simulation-engineering
   project (resource dynamics, interaction protocols, conservation bookkeeping) with a
   well-documented failure mode: evolution in a hand-built ecology optimizes quirks of
   the ecology — physics bugs, degenerate equilibria — rather than intelligence. The
   artificial-life field has half a century of beautiful terrariums that evolved
   uninteresting things (see Lehman et al., 2020, "The Surprising Creativity of Digital
   Evolution," for an entertaining and cautionary catalog). **This is much, much harder
   than the harness, and it sits far down the roadmap — parked with an explicit
   revisit trigger** (see ADR-002): revisit only if population diversity collapses in
   ways that league mechanisms (exploiter agents, fitness sharing) cannot fix.

**One correction to the death mechanism itself: archive, never delete.** Strategy
spaces are non-transitive; a "dead" policy may hold the skill that beats a future
champion. AlphaStar's league keeps past agents alive as exploiters for exactly this
reason (Vinyals et al., 2019). Death means "stops consuming compute," never "erased."
The policy repository is the fossil record, and disk is cheap.

---

## 3. The Central Hypothesis: Transfer Buys Sample Efficiency

The reason this system should exist, stated falsifiably:

> **If a single network masters task A and then learns related task B, it should reach
> mastery on B in substantially fewer samples than a fresh network — and this advantage
> should compound as the task portfolio grows.**

A child who has learned that balls fly through the air learns Pong faster than a DQN
learning from pixels ex nihilo. Humans are extremely sample-efficient learners
*because* we have already learned so much; each marginal skill is cheaper than the
last. If the system demonstrates compounding transfer — and we can verify it with
sample-efficiency curves (Component 0) — we are on the path. If it does not, the
framework's premise fails and we will know early and cheaply.

This is also where the model-based question enters honestly. The original essay claimed
model-free RL "only works for simple games" — that is wrong (model-free methods reached
grandmaster-level StarCraft II and champion-level Dota 2: Vinyals et al., 2019; Berner
et al., 2019). The *correct* argument: model-free methods are sample-inefficient
because every lesson must be experienced rather than inferred. A learned world model
lets the agent simulate — imagine — and that is the plausible mechanism for deep
transfer (Ha & Schmidhuber, 2018 "World Models"; Hafner et al., 2020-2023, Dreamer;
Schrittwieser et al., 2020, MuZero). Note that for board games the dynamics model is
free — the rules are known — which is what AlphaZero exploits via MCTS (Silver et al.,
2018). World-model learning was initially deferred but has been **promoted** (ADR-003):
Phase 2 includes a learned-dynamics track in which the rules are withheld and the agent
must learn a transition model from observation and plan against it — MuZero's move, in
a domain where the true model exists for verification.

On reward and motivation, the original essay's observations survive intact: rewards are
fleeting (satisfaction has a temporal half-life), meaningful rewards are challenging
(they push capability), and rewards evolve (mastery breeds boredom; meaning must
evolve). These map onto the intrinsic-motivation literature — curiosity as prediction
error (Pathak et al., 2017), novelty as reward (Burda et al., 2018), learning progress
itself as the reward signal (Oudeyer et al., 2007). They become concrete design inputs
for the curriculum scheduler in Phase 3, rather than a separate reward-philosophy
component.

### Working Hypotheses

The crux, stated as the two halves of the **stability-plasticity dilemma**: agents
must (i) continually learn without catastrophic forgetting (stability / backward
transfer) and (ii) learn each new thing faster because of what they already know
(plasticity / forward transfer). Many methods buy one by selling the other — EWC
protects old tasks by making new learning harder. This project's non-negotiable is
refusing that trade. Enablers hypothesized, each independently testable:

- **(a) World models** are necessary for deep transfer — simulation lets lessons be
  inferred rather than experienced (promoted; ADR-003).
- **(b) Pedagogical improvements** — curricula matched to capability (Phase 3).
- **(c) Weight-update improvements** — continual backprop / dormant-neuron
  reinitialization in the Sutton line (Dohare et al., 2024) (Phase 4).
- **(d) Architectural changes** — most speculative; pursued as diagnosis-driven design
  (Component 1 tenets).
- **(e) Rehearsal / experience management** — interleaving old-task experience is
  empirically the strongest known anti-forgetting lever and the baseline every
  cleverer method must beat.
- **(f) Capacity growth** — a fixed-size network cannot absorb tasks forever; growing
  capacity as the portfolio grows (progressive-networks lineage) is distinct from
  both (c) and (d).

### The Environment Ladder

Each tier adds one demand. **Promotion rule: an agent graduates a tier only by
demonstrating transfer** — it must learn tier N+1 measurably faster because of tier N,
without losing tier N. The ladder is a sequence of transfer exams, not a list of
environments.

| Tier | Environments | New demand |
|---|---|---|
| 0 | Connect-N family, tic-tac-toe | Exact evaluation (solvers); learned-dynamics track requires a verifiable world model |
| 1 | MiniGrid / BabyAI (Chevalier-Boisvert et al., 2019) | Partial observability; grounded language instructions |
| 2 | Crafter (Hafner, 2021) / Craftax | Open-ended long-horizon competence; world models earn their keep |
| 3 | DM Control; Procgen (Cobbe et al., 2020) | Continuous physics; generalization across procedural variation |
| 4 | Melting Pot (Leibo et al., 2021); Hanabi; Overcooked | Other agents: coordination, conventions ("culture"), theory of mind |
| 5 | Minecraft via MineDojo (Fan et al., 2022) | Open world: physics + language + culture + agents; beyond single-machine compute |

---

## 4. Implementation Roadmap

The organizing principle — learned empirically in the connect4-rl precursor, where
every "clever" component change (Double DQN, low LR, wider FC) lost to the plain
baseline and only the loop-level variable (more training) mattered:

> **Close the loop with the dumbest possible components first. Then upgrade one
> component at a time, measuring each upgrade against the dumb baseline.**

- **Phase 0 — Loop harness.** Generalize the connect4-rl champion-challenger loop into
  domain-agnostic interfaces: `Task` (environment + reward + fixed benchmark), `Policy`
  (artifact + embedded metadata), `Evaluator`, `Archive` (append-only fossil record),
  `Scheduler` (heuristic). Connect 4 is seed task #1. **The harness's league +
  archive IS the End-of-Life mechanism** — no ecology required.
- **Phase 1 — Measurement foundation.** Minimax solver benchmark for Connect 4
  (absolute yardstick). Transfer-metric protocol: episodes-to-threshold, ± pretraining,
  fixed seeds, matched compute. No evolution work until this exists.
- **Phase 2 — Transfer hypothesis test.** A family of related tasks (Connect-3, board
  size variants, tic-tac-toe, gravity-off variants). One network, multiple tasks.
  Measure whether mastery of A accelerates B. First real science result; falsifiable
  either way.
- **Phase 3 — Pedagogical evolution.** Auto-curriculum over the task family — sample
  tasks by learning progress (PLR-style / Oudeyer-style). Cheapest research loop,
  strongest literature.
- **Phase 4 — Continual plasticity.** Sequential task learning; measure catastrophic
  forgetting and plasticity loss; neuron reinitialization (extending the connect4-rl
  prototype); then gradient-guided structural growth (RigL-style).
- **Phase 5 — Architectural evolution.** Constrained architecture grammar, small
  population, selection via the existing league. Last, because it is the most expensive
  per bit of information gained.

---

## 5. Deferred Ideas (parked, with revisit triggers)

Recorded so future sessions do not wander into them prematurely:

| Idea | Why deferred | Revisit trigger |
|---|---|---|
| **Resource-competition ecology** | Orders of magnitude more engineering than the league, with a known "beautiful terrarium, degenerate result" failure mode | Population diversity collapse that league mechanisms can't fix (ADR-002) |
| **Learned structural-update policy** (a network that rewrites other networks' structure) | A meta-learning research program unto itself; heuristics (gradient-magnitude growth, dormancy-based pruning) must be exhausted first | Heuristic structural rules demonstrably plateau on Phase 4 metrics |
| **Robotic embodiment, self-repair, physical self-sufficiency** | The far-future version of the vision | Not on this roadmap |

*World models, imagination, and planning were parked in Revision 2 and have since been
promoted to the roadmap (Phase 2 learned-dynamics track) — see ADR-003.*

---

## 6. Long-Term Vision

*(The original essay's closing ambitions, preserved — motivating context, not
engineering guidance. Nothing in this section directs near-term work.)*

Different versions of such an agent could live online, in a robot, or across a network
of physical agents. An entity with the capability to modify itself should be able to
learn ever more skills, transfer its intelligence to new substrates, and network its
capabilities. We as humans are primitive versions of what we will become and of the
entities we will create. The ambition — stated plainly — is to have the agency to play,
to design some of these systems, and to help the take-off happen. A worthy grand
challenge in this direction: artificial self-subsistence — a system that provides
itself with energy and repairs itself indefinitely, without human intervention.

---

## References

- Allis, V. (1988). *A Knowledge-Based Approach of Connect-Four.* MSc thesis, Vrije Universiteit Amsterdam. (Connect 4 solved: first player wins.)
- Bahdanau, D., Cho, K., & Bengio, Y. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate.* (The original attention mechanism, born from the fixed-vector bottleneck diagnosis.)
- Balduzzi, D., et al. (2019). *Open-ended learning in symmetric zero-sum games.* ICML.
- Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). *Curriculum Learning.* ICML.
- Berner, C., et al. (2019). *Dota 2 with Large Scale Deep Reinforcement Learning.* (OpenAI Five.)
- Burda, Y., et al. (2018). *Exploration by Random Network Distillation.*
- Chevalier-Boisvert, M., et al. (2019). *BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning.* ICLR.
- Cobbe, K., et al. (2020). *Leveraging Procedural Generation to Benchmark Reinforcement Learning.* ICML. (Procgen.)
- Dennis, M., et al. (2020). *Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design.* NeurIPS. (PAIRED.)
- Dohare, S., Sutton, R. S., et al. (2024). *Loss of plasticity in deep continual learning.* Nature 632.
- Elsken, T., Metzen, J. H., & Hutter, F. (2019). *Neural Architecture Search: A Survey.* JMLR.
- Evci, U., et al. (2020). *Rigging the Lottery: Making All Tickets Winners.* ICML. (RigL.)
- Fan, L., et al. (2022). *MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge.* NeurIPS.
- Fedus, W., Zoph, B., & Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR.
- Frankle, J., & Carbin, M. (2019). *The Lottery Ticket Hypothesis.* ICLR.
- French, R. M. (1999). *Catastrophic forgetting in connectionist networks.* Trends in Cognitive Sciences.
- Fukushima, K. (1980). *Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position.* Biological Cybernetics.
- Ha, D., & Schmidhuber, J. (2018). *World Models.*
- Hafner, D. (2021). *Benchmarking the Spectrum of Agent Capabilities.* (Crafter.)
- Hafner, D., et al. (2023). *Mastering Diverse Domains through World Models.* (DreamerV3.)
- Hawkins, J. (2021). *A Thousand Brains: A New Theory of Intelligence.* Basic Books.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR. (ResNets, from the degradation diagnosis.)
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation.
- Jaderberg, M., et al. (2017). *Population Based Training of Neural Networks.*
- Jiang, M., Grefenstette, E., & Rocktäschel, T. (2021). *Prioritized Level Replay.* ICML.
- Kirkpatrick, J., et al. (2017). *Overcoming catastrophic forgetting in neural networks.* PNAS. (EWC.)
- Lehman, J., et al. (2020). *The Surprising Creativity of Digital Evolution.* Artificial Life 26(2).
- Leibo, J. Z., et al. (2021). *Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot.* ICML.
- Lyle, C., et al. (2023). *Understanding Plasticity in Neural Networks.* ICML.
- Mallya, A., & Lazebnik, S. (2018). *PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning.* CVPR.
- McCloskey, M., & Cohen, N. J. (1989). *Catastrophic interference in connectionist networks.* Psychology of Learning and Motivation.
- Novikov, A., et al. (2025). *AlphaEvolve: A coding agent for scientific and algorithmic discovery.* Google DeepMind. (LLM-guided evolution producing new algorithms.)
- Oudeyer, P.-Y., Kaplan, F., & Hafner, V. (2007). *Intrinsic Motivation Systems for Autonomous Mental Development.* IEEE Trans. Evolutionary Computation.
- Pathak, D., et al. (2017). *Curiosity-driven Exploration by Self-supervised Prediction.* ICML. (ICM.)
- Real, E., et al. (2019). *Regularized Evolution for Image Classifier Architecture Search.* AAAI.
- Real, E., et al. (2020). *AutoML-Zero: Evolving Machine Learning Algorithms From Scratch.* ICML.
- Romera-Paredes, B., et al. (2024). *Mathematical discoveries from program search with large language models.* Nature. (FunSearch.)
- Rusu, A. A., et al. (2016). *Progressive Neural Networks.*
- Sabour, S., Frosst, N., & Hinton, G. E. (2017). *Dynamic Routing Between Capsules.* NeurIPS. (Reference frames + voting — nearest ML relative of cortical columns.)
- Schrittwieser, J., et al. (2020). *Mastering Atari, Go, chess and shogi by planning with a learned model.* Nature. (MuZero.)
- Shazeer, N., et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR.
- Silver, D., et al. (2018). *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.* Science. (AlphaZero.)
- Stanley, K. O., & Miikkulainen, R. (2002). *Evolving Neural Networks through Augmenting Topologies.* Evolutionary Computation. (NEAT.)
- Sutton, R. S. (2019). *The Bitter Lesson.*
- Taylor, M. E., & Stone, P. (2009). *Transfer Learning for Reinforcement Learning Domains: A Survey.* JMLR.
- Thousand Brains Project (Numenta, open-sourced 2024). *Monty — a sensorimotor learning framework implementing the Thousand Brains Theory.* Documentation: https://thousandbrainsproject.readme.io — Code: https://github.com/thousandbrainsproject/tbp.monty. (Named for Vernon Mountcastle, who proposed the cortical column as the neocortex's repeating functional unit. Directly relevant to the queued cortical-column experiment.)
- Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
- Vinyals, O., et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning.* Nature. (AlphaStar.)
- Wang, R., Lehman, J., Clune, J., & Stanley, K. O. (2019). *POET: Paired Open-Ended Trailblazer.* (Endlessly generating increasingly complex environments.)
- Zoph, B., & Le, Q. V. (2017). *Neural Architecture Search with Reinforcement Learning.* ICLR.
