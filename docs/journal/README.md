# Journal guide (hjepa_agent)

Records the thinking behind `hjepa_agent`: an extendable agent following LeCun (2022), built to act in `agent_world`.
Rules and labels are inherited from `../agent_world/` (snapshot of the world repo's journal). Read `../agent_world/README.md` first.

| Document | Answers | Changes when |
| --- | --- | --- |
| [journey.md](journey.md) | How did we get here? | Appended after every effort or decision |
| [codebase-study.md](codebase-study.md) | What do the 4 reference papers and repos do, and where do they fall short for us? | A repo or paper is studied |
| [agent-proposal.md](agent-proposal.md) | How could the diagram's agent be built? | A proposal is made, agreed or dropped |
| [world-exploration.md](world-exploration.md) | What does the world's data look like, and is the variance enough? | The world or the probes change |
| [training-strategies.md](training-strategies.md) | How do we train: joint or staged, and on what data? | A strategy is agreed, dropped or compared |
| [intentions.md](intentions.md) | What has Kumar explicitly intended, and what have I noticed? | An intention is stated or noticed |
| [predictability-study.md](predictability-study.md) | Can masked regions of our frames be predicted from the visible context? | The study is repeated or extended |
| [simulation-privileges.md](simulation-privileges.md) | Which capabilities exist only because we work in simulation, and how are they used? | A privilege is used or considered |
| [sigreg-weight.md](sigreg-weight.md) | How is the SIGReg weight chosen, and how sensitive are results to it? | New evidence or our own sweep |
| [ssl-landscape.md](ssl-landscape.md) | What self-supervised latent-prediction schemes exist, what is proven, and what is installed? | A scheme is added or evidence changes |
| [stage1-spec.md](stage1-spec.md) | What exactly is stage 1: data, preprocessing, model, training, evaluation, compute, and which items are agreed or open? | A decision is agreed, changed or opened |
| [component-roles.md](component-roles.md) | What is each component of the agent responsible for, per LeCun's paper and the diagram, and where could they integrate? | A role or integration is agreed or changed |
| [stage1-plan.md](stage1-plan.md) | What does perception pre-training need, what do we reuse, and what is left to build? | The plan or the reuse choices change |
| [compute-options.md](compute-options.md) | What does training need, what do the papers used, and what do rentals cost? | Compute is measured or a price changes |

## Status labels

Same as the world journal: **[Kumar]**, **[Agreed]**, **[Proposed]**, **[Open]**, **[Built]**, **[Dropped]**. Add **[Verified]** (read in paper or code, or measured) and **[Unverified]** (inferred, not checked).
Docs are append-only; a pivot is appended only after it has been agreed.

## Added 2026-11-17
- `dev-workflow.md`: the development workflow (branches, PRs, pushing, enforcement, journals) and how it compares with the standards.
- `cloud-iterations.md`: one report per auto-research iteration on the cloud GPU.
- `evaluation-methods.md`: how the world model is evaluated (monitors, probes, baselines, identifiability, layer probes, convergence) and how each hypothesis was tested.
