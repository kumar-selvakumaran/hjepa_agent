# Development workflow

How code, journals and experiments move from an idea to GitHub in this project, and how that compares with the common standards.
Written 2026-11-17. Sources of each rule: the intentions register (`INT-nn`), `CLAUDE.md`, and the agent_world principles snapshot.
This document describes; the intentions register decides. If they disagree, the register wins and this document is corrected.

---

## 1. The standards this workflow draws on

| Standard | Integration branch | Work branches | How work merges | Releases |
| --- | --- | --- | --- | --- |
| **GitHub flow** | `main` (always deployable) | One short-lived branch per change | Pull request, review, CI, merge into `main` | Deploy from `main` |
| **git-flow** (Driessen, 2010) | `develop` | `feature/*` off `develop`; `release/*`, `hotfix/*` | Feature branches merge into `develop` | `release/*` merges into `main` and is tagged; `main` holds only releases |
| **Stacked pull requests** | Either of the above | Each branch built on the previous open PR's branch | Each PR targets the branch below it; when the lower PR merges, the next is retargeted to the integration branch | Unchanged |
| **Trunk-based development** | `main` | Very short-lived or none | Small changes straight to `main`, guarded by CI and feature flags | Continuous |

Common conventions shared by all of them:
- **Branch prefixes:** `feature/<name>` for new behaviour, `fix/<name>` for bug fixes; also seen: `chore/`, `docs/`, `refactor/`.
- **Pull request:** the unit of review; it carries the description, the discussion and the CI result.
- **CI:** automated tests run on every PR; a PR merges only when they pass.
- **Protection rules:** the integration branch accepts changes only through PRs; force pushes and deletion are blocked.

## 2. This project's workflow, in one line

**git-flow naming with GitHub-flow simplicity:** effort branches off `develop` (or stacked on an open PR), a pull request reviewed and merged by Kumar, no release branches yet, and no CI yet (tests run locally and on the target machine).

## 3. What it is (declarative)

### 3.1 Repositories and where they live

| Location | What it is | Holds | Pushed? |
| --- | --- | --- | --- |
| `projects/hjepa_agent` | The working repository | Code **and** journals (`docs/`); local `main` | Never as a whole until the journals are reviewed (INT-92) |
| `projects/hjepa_agent_worktrees/code-only` | A separate clone (sibling folder) | The code-only history (no `docs/`); the PR branches | Yes: `develop` and the effort branches (INT-97, INT-98) |
| GitHub `kumar-selvakumaran/hjepa_agent` | The remote | `main` (initial commit only, for now), `develop`, open effort branches | (it is the remote) |

Why two local places: local `main` mixes code and journal commits; journals may not be published before Kumar reviews them, so the code line was split into its own history (`docs/` filtered out) in the sibling clone (INT-97).

### 3.2 Branches

| Branch | Role | Who changes it | How |
| --- | --- | --- | --- |
| GitHub `main` | Default branch; will receive the code line and the journals after review | Kumar | Through a PR, later |
| GitHub `develop` | Integration branch for code (INT-98) | Kumar | Merging reviewed PRs |
| `feature/<name>`, `fix/<name>` | One per effort | Claude (or Kumar) | Commits, then a PR into `develop` or into the branch it stacks on |
| Local `main` (working repo) | Code plus journals; what is deployed to the cloud | Claude | Cherry-picks of merged PRs, plus journal commits after Kumar confirms |

Branch rules:
- **One branch per effort**: a feature, a fix, a study (INT-91). Small, non-additive commits may go straight to local `main` for now (INT-91).
- **Never delete branches**: they record each effort (Kumar, 2026-11-17).
- **Never force-push** (INT-89, INT-91).

### 3.3 Commits

- Small and atomic; each commit passes the tests (`CLAUDE.md`).
- Message: a short imperative title, a body that says why, and the attribution line `Co-Authored-By: Claude <model> <noreply@anthropic.com>` when Claude made the commit.
- Journals and docs are committed only after Kumar confirms them (INT-91, INT-92).

### 3.4 Pull requests

- Base: `develop`, or the branch of the open PR it stacks on.
- Description: what changed, why, how it was verified (tests, measurements), what is not covered; ends with the Claude Code attribution line.
- No `@`-mentions outside backticks (they notify people).
- Review and merge: **Kumar only**. Claude never merges, approves or closes PRs.

### 3.5 Pushing

| What | May Claude push it? | Rule |
| --- | --- | --- |
| Safe, verified, secret-free code (effort branches) | Yes | INT-91 |
| Journals and docs | Only after Kumar confirms | INT-91, INT-92 |
| Local `main` as a whole | No, until the journals are reviewed | INT-92 |
| Force pushes, branch deletion | Never | INT-89, Kumar 2026-11-17 |

### 3.6 Enforcement

| Layer | Where | Enforces | Status |
| --- | --- | --- | --- |
| 1. Fine-grained token | GitHub token used by `gh` and git | Only this repository; only Contents and Pull requests | In place (2026-11-17); its exact settings are Kumar's |
| 2. Rulesets | GitHub repository settings | PR required on `develop` and `main`; no force push or deletion on any branch | Not yet set (Kumar's action) |
| 3. Claude Code guard | `.claude/settings.json`, `.claude/hooks/guard_git.py` (PR #1) | Blocks merges, approvals, closes, settings, API, issues, releases, secrets, workflows, force or delete pushes, branch deletion, `@`-mentions | In place, tested |
| (missing) Separate bot account | GitHub | Server-side distinction between Claude and Kumar (required approvals) | Not set; optional |

### 3.7 Tests (until CI exists)

- `uv run pytest` on the laptop before every commit and before every push.
- The same suite on the cloud machine after each redeploy (step C04).
- Long-running behaviour is verified by the runs themselves (monitoring, probes), reported in `docs/journal/cloud-iterations.md`.

### 3.8 Journals and records

| Record | Purpose | Location | In git? |
| --- | --- | --- | --- |
| Journey | Dated account of each effort and decision | `docs/journal/journey.md` | After Kumar confirms |
| Intentions | The register of decisions (`INT-nn`) | `docs/journal/intentions.md` | After Kumar confirms |
| Cloud iterations | One report per auto-research iteration | `docs/journal/cloud-iterations.md` | After Kumar confirms |
| This document | The workflow | `docs/journal/dev-workflow.md` | After Kumar confirms |
| Campaign artefacts | Checkpoints, logs, plots, evidence, reasoning log | `scratch/auto-research/<campaign>/` | Never (gitignored) |
| HITL step files | Commands Kumar runs on paid machines | `scratch/hitl/` | Never (gitignored) |

## 4. How it works (procedural)

### 4.1 A code change, end to end

1. **Branch** in the sibling clone: `git fetch origin`, then `git checkout -b fix/<name> origin/develop` (or off the open PR's branch when stacking).
2. **Change** the least code that does the job; reuse stable packages; `uv` only.
3. **Test** locally: `uv run pytest`; for training code, a short smoke run.
4. **Commit** with a message that says why, plus the attribution line.
5. **Push** the branch: `git push -u origin fix/<name>`.
6. **Open the PR**: `gh pr create --base develop` (or `--base <branch below>` when stacked), with the description in 3.4.
7. **Deploy without waiting** when an experiment needs it: pack the branch tip, upload, reinstall and test on the instance (steps L01, C03, C04), then relaunch; the iteration report cites the PR.
8. **Kumar reviews and merges.**
9. **Bring it home**: in the working repo, `git fetch origin develop` and `git cherry-pick <commit>` onto local `main`, so local `main` (which also holds the journals) has the change.
10. **Restack**: if another PR was stacked on this one, retarget it to `develop` (`gh pr edit <n> --base develop`).

### 4.2 Parallel efforts (stacked PRs)

1. PR A (`fix/a`) is open against `develop`.
2. Effort B needs A's change: `git checkout -b feature/b fix/a`, and its PR targets `fix/a`, so the PR shows only B's diff.
3. When A merges, B is retargeted to `develop`; if A changed during review, B is rebased onto the new `develop` locally (never force-pushed; a new branch name is used if history must change).

### 4.3 A journal change

1. Edit the journal in the working repo; leave it uncommitted.
2. Kumar reviews; on confirmation, commit to local `main` (or a `docs/<name>` branch).
3. Publishing journals to GitHub happens later, through their own PR into `main`, after the review agreed in INT-92.

### 4.4 Cloud and costly commands

1. Claude prepares a step file (`hitl-dev` skill) in `scratch/hitl/`; each step is one command with absolute paths that writes `log.md`, `summary.json`, `env.txt`.
2. Kumar runs billing, launch and termination steps; under the auto-research authority (INT-96) Claude runs job steps (start, stop, monitor, redeploy) itself.
3. Never `shutdown` a cloud instance; termination is Kumar's console action.

## 5. Where this workflow departs from the standards, and why

| Departure | Standard practice | Here | Reason |
| --- | --- | --- | --- |
| No CI | Tests run automatically on each PR | Tests run by hand, locally and on the instance | No CI set up yet; principles say PRs with CI once it exists |
| Two histories (working repo and code-only clone) | One repository, one history | Code line in a sibling clone with `docs/` filtered out | Journals must not be published before review (INT-92) |
| Cherry-picks back into local `main` | Pull `develop` and merge | Cherry-pick each merged PR | The two histories are unrelated after filtering, so merges do not apply |
| Deploying before review | Deploy after merge | Experiments deploy from the PR branch tip | Cloud time costs money; waiting on review would idle the GPU (INT-98) |
| No release branches or tags | `release/*` and tags in git-flow | None yet | No releases yet; a stage-1 tag is a candidate once the world model converges |
| Same GitHub account for Claude and Kumar | Bots have their own accounts | One account, separated by the local guard and the token | Simplicity; a bot account is optional (3.6) |

## 6. Limits and open items

- GitHub rulesets (layer 2) are not set; until then, the server accepts any push the token allows.
- The journal review (INT-92) is pending; after it, the two histories can be reunited through a journal PR into `main`.
- `scratch/hitl/` and `scratch/auto-research/` are never pushed; their key results reach git only through the journals.

## 7. Update 2026-11-18: rulesets set; journals pushable after a safety check

- **[Kumar, done]** GitHub rulesets (layer 2) are active on hjepa_agent, the same as on agent_world:

  | Ruleset | Targets | Rules | Bypass |
  | --- | --- | --- | --- |
  | `protect-develop-main` | `develop`, `main` | Restrict deletions; block force pushes; require a pull request (0 approvals, merge commits only) | Repository admin, always |
  | `no-force-no-delete` | All branches | Restrict deletions; block force pushes | Repository admin, always |

- **Verification:**
  - **[Verified]** `protect-develop-main`: all its settings were checked on a screenshot of the settings page.
  - **[Stated by Kumar]** `no-force-no-delete`: its name, target, bypass and Active status were seen; the ticked rules were not.
- **Caveat:** Claude pushes with Kumar's token, so the admin bypass covers Claude too. The local guard (layer 3) remains what stops Claude from force-pushing or deleting branches.
- **Supersedes, in section 6:** the line saying the rulesets are not set.
- **[Kumar] INT-101:** journals no longer wait for review. They may be committed and pushed once Claude has checked them for secrets and identifying information. They reach GitHub as `docs/` pull requests into `develop`. This supersedes the pending journal review in section 6.
