# hjepa_agent

An extendable agent following LeCun's *A Path Towards Autonomous Machine Intelligence*, built to act in the sibling `agent_world`.
Read `docs/agent_world/README.md` (guide), then `principles.md` and `world-spec.md`. Those are a reference snapshot; see `docs/agent_world/SOURCE.md`.

Standing rules (same as agent_world; full list in `docs/agent_world/principles.md`):

- After every effort or decision, append a dated section to this project's journey doc.
- One branch per effort off `main`, small atomic commits, every commit passes the tests, merge locally into `main`.
- Use `uv` only. Investigate with throwaway probes in `scratch/<task>.py` that write output to a text file.
- Raise questions, concerns and blockers as indexed items (Q, C, B), each with a default.
- Docs are append-only. Record a pivot only after it has been agreed.
- Prefer GIFs/visual outputs of behaviour over asking Kumar to run things.
- Code stays machine-independent (containerizable); no personal paths or usernames.
- Keep from-scratch code to a minimum: reuse stable packages (stable-pretraining, stable-worldmodel, timm, scikit-learn) where they give a pipeline with little effort, and be critical before writing new code.
- Run independent work in parallel and delegate to subagents wherever subtask and data independence allow.
- Surface questions through the AskUserQuestion tool: default first (Recommended), other sensible options, room for notes.
- Costly, risky or critical-environment commands (cloud, large training) follow the `hitl-dev` skill: Claude does the groundwork and writes a step file (in `scratch/hitl/`); the user runs the steps.
- Pushing: safe, verified, secret-free code may be pushed; journals and docs need Kumar's confirmation before they are included in git. Large efforts get their own branch, merged into `main` after testing; small non-additive commits may go straight to `main`. HITL step files, helper scripts and outputs live in the gitignored `scratch/hitl/`.
