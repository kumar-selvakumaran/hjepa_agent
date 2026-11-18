# Choosing the SIGReg weight

Investigated 2026-11-16 (INT-74). Labels: see [README.md](README.md). **[Checked]** means I read it in the paper text; **[Reported]** means a research agent read an arXiv HTML page or a repo file through a summarising fetcher (not the PDF); **[Unverified]** means a snippet or a claim not checked.

## 1. What the weight is

- The training loss is the prediction error plus lambda times SIGReg; lambda sets how hard the latents are pushed toward an isotropic Gaussian relative to how well the next latent must be predicted.
- Other knobs are said to matter little: the number of random projections (64 to 1024) and the integration knots (4 to 32) in LeWM's ablations **[Checked]**; 512 vs 2048 slices changed ImageNet-100 accuracy by under one point in LeJEPA **[Reported]**.
- With the (time, batch, dimension) convention, the weight transfers across batch sizes but not across sample definitions (H-JEPA appendix K.3) **[Checked]**.

## 2. What each user chose and found

| Source | Weight | How chosen | Sensitivity found | Evidence |
| --- | --- | --- | --- | --- |
| LeWM, Push-T | 0.09 in the config (0.1 in the paper text) | "Simple bisection search"; no protocol given | Above 80% success for 0.01 to 0.2; peak near 0.09; sharp drop at 0.5, where the regulariser dominates | **[Checked]** |
| LeWM, simple environments | not swept | none | Weaker results on Two-Room, which the paper attributes to low diversity and low intrinsic dimension | **[Checked]** |
| H-JEPA | 0.09 on AntMaze, Push-T, Cube; **0.72 (LeWM) and 0.18 (H-JEPA) on FourRoom**; 0.08 on DROID level 1 | Cross-validated jointly with an inverse-dynamics weight | Without SIGReg the rank collapses to 1 to 14 of 384; the two weights must scale together; level 2 is insensitive | **[Checked]** |
| LeJEPA | 0.05, with the loss written as (1 - lambda) x prediction + lambda x SIGReg | Called a robust default | Stable across lambda on ImageNet-100; the optimum shifts slightly with the number of views | **[Reported]** |
| Temporally centred SIGReg | 0.09 default | A sweep on LIBERO-Long | Non-monotonic: 0.01 best (50.3), 0.09 gives 44.7, 0.005 gives 43.4 | **[Reported]** |
| SkyJEPA | 0.02 | Not stated | No sweep | **[Reported]** |
| stable-pretraining LeJEPA code | 0.02 | none | none | **[Checked]** (code) |

## 3. Reading

- **Difficulty:** moderate. The weight is the only knob that matters, and within a factor of three to five of the best value results change little. Across users the chosen values span from 0.005 to 0.72, so no value is universal.
- **It depends on the environment.** Simple environments took larger values (FourRoom 0.72 for LeWM); one long-horizon task preferred 0.01.
- **It depends on the other terms.** With an inverse-dynamics term the two weights must be set together.
- **No schedule found.** No source warms up or anneals the weight **[Reported]**.
- **Hypothesis for our setting [Unverified]:** with frameskip 1 the prediction error is small, so a fixed weight regularises relatively harder than at LeWM's frameskip 5; the best weight may therefore be lower than 0.09. The FourRoom result points the other way for simple scenes, so the range worth testing is wide.

## 4. Practical procedure (from the sources, adapted to our rule of no planning)

1. Fix 1,024 projections, 17 knots and a batch of at least 128.
2. Sweep the weight on a logarithmic grid.
3. Judge each run without a planning score (INT-11), using: validation prediction error at 1, 3 and 5 steps against a copy-the-last-latent baseline; the SIGReg statistic against its floor of about 1.0 to 1.4 on healthy data; effective rank; probes for position, heading, room and food visibility; and the predictor's sensitivity to the action.
4. Choose the centre of the plateau of acceptable runs, not the single best value.

## 5. Unknown

- How LeWM's bisection is run (objective, bounds).
- Whether any weight schedule helps.
- The link between the best weight and data diversity beyond the qualitative claim.
- Full-text weights for several 2026 papers (VISReg, Weak-SIGReg, Gaussian-Constrained LeJEPA, Mine-JEPA).
