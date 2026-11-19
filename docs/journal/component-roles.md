# Component roles and responsibilities

Written 2026-11-15. Labels: see [README.md](README.md); **[Verified]** means read in a primary or directly shown source, **[Unverified]** means from a secondary source or memory.

## 1. Scope and sources

This document states what each component of the HJepa-Agent is expected to do and not do, drawn from the Lucid diagram ([../diagrams/README.md](../diagrams/README.md)) and LeCun's *A Path Towards Autonomous Machine Intelligence* (APAMI, version 0.9.2, 2022). It then applies the project's intentions ([intentions.md](intentions.md)) and lists concrete ways the components could interface in our system, including a possible configurator role for attentive pooling.

| Source | Status |
| --- | --- |
| APAMI Figure 2 and its caption, as embedded in the Lucid page | **[Verified]**: seen in the exported diagram image |
| The Lucid diagram (modules and arrows) | **[Verified]**: from the Lucid data, see `docs/diagrams/README.md` |
| APAMI full text and section numbers | **[Verified]**: PDF v0.9.2 (62 pages) read via the Wayback copy of the OpenReview PDF; see the Verification note at the end |
| A 24-slide review of APAMI (Temple University, `scratch/papers/temple_apami_slides.pdf`) | Secondary; gives module roles, Mode-1 and Mode-2 loops, H-JEPA |
| A wiki summary (mishig-jepawiki) | Secondary; no section numbers |
| Section numbers such as 3.1, 4.7, 4.9 | **[Verified]**: checked against the PDF (3.1 loops, 4.7 hierarchical planning, 4.9 writable memory); earlier carried from the world journal ([../agent_world/world-spec.md](../agent_world/world-spec.md)) |

## 2. Components

### What they are (declarative)

| Component | Role | Inputs | Outputs |
| --- | --- | --- | --- |
| **World** | The environment outside the agent | Actions | Percepts (here: a 64x48 RGB view, later a body channel) |
| **Perception** | Estimates the current world state | Percept x; configurator settings | State `s[0] = Enc(x)` |
| **World Model** | Predicts future states given imagined actions | State, action, optional latent z | Next state `s[t+1] = Pred(s[t], a[t], z)` |
| **Short Term Memory** | Keeps current and predicted states with their intrinsic costs | States and costs written after each step | Stored `(state, cost)` pairs for planning and critic training |
| **Cost: intrinsic cost** | Immediate discomfort of a state ("pain", "hunger"); hard-wired | State | Scalar energy |
| **Cost: Critic** | Predicts future values of the intrinsic cost | State | Scalar predicted future cost |
| **Actor** | Proposes action sequences and picks the first action (Mode-2); or maps state to action directly (Mode-1) | State, world-model rollouts, cost | Action |
| **Configurator** | Configures the other modules for the task at hand | Inputs from all modules | Settings, parameters, connection graph, subgoals, cost configuration |

### Trained how, and what each is not responsible for

| Component | Trained how (per the paper) | Not responsible for |
| --- | --- | --- |
| Perception | Trainable (Sec. 1 lists most modules as trainable); in JEPA terms the encoder, learned self-supervised (Secs. 4.4, 4.5) **[Verified]**; the paper does not say which modules are frozen | Prediction, cost, action choice |
| World Model | Trainable by self-supervised prediction in latent space (JEPA, Sec. 4.4); stackable as H-JEPA (Sec. 4.6) **[Verified]** | Evaluating cost, proposing or optimising actions **[Verified]** (Fig. 2 caption and Sec. 3: predicts as a function of imagined actions *proposed by the actor*). The paper also gives it a second job: estimating state information perception missed (Sec. 3) |
| Short Term Memory | A store, a key-value associative memory (Secs. 3, 3.3, 4.8.1); not trained as such **[Verified]** | Learning; it only keeps states and costs. The world model reads and writes it; the critic and Mode-1 policy read it (Sec. 3). A writable world-state memory updated sparsely is in Sec. 4.9 |
| Intrinsic cost | **Not trained**: immutable, hard-wired **[Verified]** (Fig. 2 caption, Sec. 3). Its weights may be modulated by the configurator (Secs. 3, 6), though the paper warns that heavy modulation endangers safety terms | Predicting the future (that is the critic) |
| Critic | Trainable; predicts future intrinsic cost; trained by regressing a later intrinsic energy on an earlier state retrieved from memory (Sec. 3.3) **[Verified]** | Defining drives (that is the intrinsic cost) |
| Actor | Mode-2 by optimisation at run time (gradient-based, or dynamic programming, beam search or MCTS for discrete actions), no training needed; Mode-1 policy trained from Mode-2 results (Secs. 3.1.3, 5) **[Verified]** | Predicting the world, computing cost |
| Configurator | Left open: the paper describes a role and mechanism, no implementation; how it learns to decompose tasks into subgoals is explicitly unanswered (Sec. 6) **[Verified]**. Trainability is implied, not stated | Doing perception, prediction or action itself |

Per-component notes:

- **Perception.** Mode-1 equation `s[0] = Enc(x)` (Sec. 3.1.1, Fig. 3) **[Verified]**. The configurator primes it (Secs. 3, 6: modulating low-level conv weights or high-level transformer tokens) and the Lucid arrow shows it **[Verified]**. The state may be hierarchical (Sec. 3).
- **World Model.** The latent z carries what cannot be predicted from the past (Secs. 3, 4.2) **[Verified]**. Higher H-JEPA levels predict further ahead from coarser states (Secs. 4.6, 4.7) **[Verified]**. The configurator may also configure it for the task (Secs. 3, 6); see the INT-11 note in Section 4.
- **Cost.** One scalar "energy" measuring the agent's discomfort **[Verified]** (Fig. 2 caption); the intrinsic cost sets the agent's basic behavioural nature (Sec. 3) **[Verified]**. Total cost is the sum `C(s) = IC(s) + TC(s)` (Sec. 3.2, Eq. 1); the critic is the trainable part `TC`.
- **Actor.** Three roles (Sec. 5) **[Verified]**: infer action sequences minimising cost; produce latent-variable configurations for the unknown part of the world; train Mode-1 policies. It outputs the first action or a short sequence (Sec. 3).
- **Configurator.** "Main controller" (Sec. 6) **[Verified]**: takes input from all modules, modulates parameters and connection graphs; its most important function is to set subgoals and configure the cost for them. It is needed for hardware reuse and knowledge sharing. Mechanism suggested: its outputs as extra input tokens to transformer blocks of the predictor and upper perception layers. The paper says it may prime perception, world model and cost for a goal (Sec. 3), which sits in tension with INT-11 (see Section 4).

## 3. One decision step (procedural)

Mode-2, from Sec. 3.1.2 (Fig. 4) and the Lucid diagram:

1. **Perceive.** Perception encodes the percept to `s[0]`; the cost of `s[0]` is computed and stored at once (Sec. 3.1.2, step 1).
2. **Configure.** The configurator pre-configures perception, world model, cost and actor for the task (Sec. 3). It acts before and around the loop, not as a numbered step of the paper's episode; horizon is our addition.
3. **Propose.** The actor proposes an action sequence `a[0..T]`.
4. **Simulate.** The world model predicts `s[1..T]` from `s[0]` and the proposal. It receives no cost as input; it feeds the cost.
5. **Evaluate.** The cost module scores the predicted states: total cost `F = sum_t C(s[t])`, each `C` being intrinsic cost plus critic (Sec. 3.1.2, step 4; Sec. 3.2).
6. **Plan.** The actor changes the sequence to lower the cost, by gradient through cost and world model or by discrete search; full optimisation iterates steps 3 to 6 here (the paper's steps 2 to 5).
7. **Act.** The actor sends the first action to the world.
8. **Remember.** States and costs go to short term memory; the critic later trains on them (Secs. 3, 3.3). The paper stores the current state's cost at step 1 and may also store imagined states.

Mode-1 (Sec. 3.1.1, Fig. 3): perception, then the policy gives the action directly, with no use of world model or cost for choosing it; the cost of `s[0]` is still computed and the pair stored in memory. Optionally the world model predicts `s[1]` to be adjusted against the next percept. Mode-2 results later train the Mode-1 policy (Sec. 3.1.3) **[Verified]**.

## 4. Project-specific constraints

From [intentions.md](intentions.md):

| Intention | Consequence for the components |
| --- | --- |
| **INT-11** Planning never exists in the world model | The world model sees only states and actions. No goal, cost, goal-matching or task input inside it. Steps 3, 5 and 6 above belong to the actor and cost. Reusing `stable-worldmodel`'s LeWM class means using `encode`, `predict`, `rollout` only, not `criterion` or `get_cost` |
| **INT-10** Staged training | Perception is pre-trained and frozen, then the world model is trained on it, then everything is fine-tuned jointly. Perception and world model are trained with no cost or actor present |
| **INT-12** Runtime novelties | Chasers and the turning patch appear only after pretraining. Adapting to them is an online job (memory, critic, world-model adaptation), not part of the pretraining data |
| **INT-20 to INT-26** Data | Random-walk, real-life-like data; world model learns pure dynamics. The paper's data-stream modes are "active egomotion" (mode 4) and "active agency" (mode 5) (Sec. 4.10); random walking with own actions fits mode 5 **[Verified]**; the mapping of our data to a mode is our judgement |
| **INT-43a** Uniform coverage | Consistent with the paper's emphasis on diversity of situations for training the world model (Secs. 3.2, 4.10) **[Verified, loosely]** |
| **INT-06** Survival and intrinsic costs | The intrinsic cost will encode hunger, pain avoidance, curiosity and discomfort with uncertainty; the world has no body channel for them yet |
| **INT-33, INT-31** | Costly runs use the `hitl-dev` step files; independent work runs in parallel |

## 5. Precise potential integrations

All **[Proposed]**.

| # | Interface | Proposal |
| --- | --- | --- |
| I-1 | Perception to world model | One mean-pooled patch-token vector (192-d) per frame, as in LeWM. The frozen tokens stay available to other modules |
| I-2 | World model to actor | Rollout only: the actor supplies action sequences and receives predicted latents. The planner owns the cost function |
| I-3 | Body channel | Pain and own motion enter as a second input stream, encoded next to the image. The world model predicts it like any signal; the intrinsic cost is a fixed function applied outside the model |
| I-4 | Memory to critic | The memory stores `(latent, action, intrinsic cost)`; the critic trains on it with a stop-gradient into the encoder |
| I-5 | Memory to online adaptation | The memory doubles as a replay buffer for adapting the predictor to runtime novelties (JEPA-TTT style); related to the tabled alignment idea (INT-05) |
| I-6 | Actor modes | Mode-2 uses a solver from `stable-worldmodel` over the world model; Mode-1 is a policy distilled from its outputs |
| I-7 | Uncertainty | A predictor ensemble first (PLDM's penalty), a latent variable z later |
| I-8 | Hierarchy | H-JEPA levels (stride 2) added above the frozen perception; the top level is planned first, as in H-JEPA |
| I-9 | Configurator, first version | A static settings object: cost weights, horizon, levels, planner budget. Rule-based, then learned, later |

### Potential configurator role: attentive pooling of the frozen encoder's patch tokens

- **[Kumar]** Stage 2 uses LeWM's baseline per-frame latent (the mean of patch tokens, one vector). Kumar likes attentive pooling over the frozen tokens and thinks the configurator could do this more thoroughly; to be given more thought. Documented here as a *potential* responsibility, not a plan.

**What it would be (declarative)**

| Aspect | Description |
| --- | --- |
| Operation | Learned queries attend over the 64 frozen patch tokens and return one or a few vectors |
| Controls | The query vectors, which tokens or regions get attention, attention sharpness, the number of pooled vectors |
| Reads | Task or cost weights, memory state (such as recent prediction error or uncertainty), the current latent |
| Setting | Static (fixed learned queries); rule-based (pick a query set by context); learned (a configurator network emits queries) |
| Benefit | Keeps spatial detail that the mean discards, such as small entities, and could focus on what a task needs |

**How it could work (procedural)**

1. Stage 1 freezes the encoder; its tokens are fixed.
2. A pooling module turns tokens into the latent that downstream modules use.
3. The configurator chooses or generates the pooling parameters for the current context.
4. The pooled latent goes to the module that needs it.

**The tension with INT-11**

- **[Open]** If the pooling is conditioned on the task, goal or cost weights **and** feeds the world model, the world model's input depends on the task. That puts a notion of the objective inside the world model, which INT-11 forbids.
- If gradients from a goal-driven objective flow into the pooling queries, the separation also leaks through training, even if the world model's weights never see a goal.
- The world model must also stay stationary: changing pooling at run time shifts its input distribution.

| Placement | Respects INT-11? | Note |
| --- | --- | --- |
| A. Task-conditioned pooling feeds the world model | **No** | World model input depends on the task |
| B. Task-conditioned pooling feeds the cost and critic (a readout of predicted tokens) | Yes | Needs the world model to predict tokens, not one vector; heavier |
| C. Task-conditioned pooling feeds the Mode-1 policy only | Yes | Keeps the world model clean; affects only the actor |
| D. Task-agnostic multi-query pooling learned with the world model; the configurator only **selects** among fixed query sets downstream | Yes, if selection is outside the model | World model predicts all pooled vectors; the choice happens at the actor or cost |
| E. Context-only pooling (uncertainty or memory state, no goal) feeding the world model | Debatable | No goal, but not static; needs a stationarity check |

**Risks**

- Breaks the frozen-encoder or goal-free separation if placement A is used, or if gradients flow back.
- Attention may collapse to trivial tokens; needs a rank or probe check.
- Adds trainable parts and makes evaluation harder than one fixed mean.
- The paper leaves the configurator largely open (Sec. 6: subgoal decomposition unanswered), so there is no reference design **[Verified]**. It does suggest extra input tokens into transformer blocks, close to the pooling-query idea.

**Which stage it could enter**

| Stage | Role |
| --- | --- |
| Stage 2 baseline | Mean of patch tokens, as in LeWM **[Kumar]** |
| Stage 2 ablation | Placement D with a fixed learned pooler, task-agnostic, to compare with the mean on probes of held-out entities |
| Actor and cost stage | Placements B or C, with the configurator choosing among fixed query sets, only if the world model stays task-agnostic |
| Later | A learned configurator emitting queries, once the actor and cost exist |

**[Open]** Questions for Kumar: which placement suits the intent; whether the world model may ever predict tokens instead of one vector; whether context-only pooling (placement E) counts as planning-free.

## Verification

Checked 2026-11-15 against APAMI v0.9.2 (62 pages), the OpenReview PDF as archived by the Wayback Machine; text saved at `scratch/papers/lecun_apami.txt`. Read: Fig. 2 caption, 3, 3.1 to 3.3, 4.9, 4.10, 5, 6; skimmed 4.6 and 4.7.

**Verified (upgraded or newly cited):** roles of perception, world model, intrinsic cost, critic, short-term memory, actor and configurator; intrinsic cost immutable; critic trained from memory; Mode-1 and Mode-2 steps; Mode-2 to Mode-1 policy training; actor's three roles; configurator as main controller; data-stream modes 4 and 5; section numbers 3.1, 4.7, 4.9.

**Corrections:**
- Step 5 said "intrinsic cost per state, critic at the end"; the paper sums `C = IC + TC` over all predicted states.
- Step 6 said repeat 4 to 6; the paper iterates its steps 2 to 5 (here steps 3 to 6).
- Configurator is not a numbered step of the paper's Mode-2 episode; marked as ours.
- Short-term memory is also read and written by the world model and read by the Mode-1 policy.
- Added: the configurator may modulate the intrinsic cost's weights and may prime the world model for a goal. INT-11 is therefore a deliberate deviation from the paper, not a misreading.
- All slide-only and wiki-only citations were replaced with section numbers.

**Still not from the paper:** integrations I-1 to I-9 and the stop-gradient critic training are our proposals; frozen-encoder staging (INT-10) is not discussed in the paper.
