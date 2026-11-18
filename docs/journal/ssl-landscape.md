# Landscape: self-supervised latent-prediction schemes for the perception encoder

Researched 2026-11-15 by four parallel research streams. Labels: see [README.md](README.md). Added **[Checked]** for a claim read in a primary source or in installed code this session, and **[Memory]** for a claim from a research agent's memory of a paper that was not rechecked; treat **[Memory]** and **[Unverified]** alike. Several 2026 arXiv ids came only from search results and some are implausible; verify before citing.

## One line

Latent-prediction self-supervision trains an encoder to predict some embedding from other information, and a mechanism keeps the embedding from collapsing to a constant.

## 1. What it is (declarative)

### Four families

| Family | What is predicted, from what | Needs augmented views? | Examples |
| --- | --- | --- | --- |
| A. Masked latent prediction | The embeddings of hidden regions, from visible regions of the same input | No | I-JEPA, V-JEPA 1 to 2.1, SALT, data2vec 1 and 2, CAE, C-JEPA, Latent MIM, Var-JEPA |
| B. Multi-view invariance | The embedding of one view, from another view of the same input | **Yes** (or another source of pairs) | SimCLR, MoCo, BYOL, SimSiam, DINO family, SwAV, Barlow Twins, VICReg, TiCo, W-MSE, LeJEPA |
| C. Temporal and action-aware prediction | The next (or later) embedding, from the current one, optionally with the action | No | CPC, SPR, PLDM, LeWM, DINO-WM, V-JEPA 2-AC, H-JEPA, HWM, TD-MPC2, Delta-JEPA |
| D. Reconstruction (for contrast) | Pixels or tokens | No | MAE, SimMIM, Dreamer, IRIS, DIAMOND, Genie |

### Anti-collapse mechanisms

| Mechanism | Introduced | What it prevents | Evidence | Assumptions or failure modes |
| --- | --- | --- | --- | --- |
| EMA teacher | BYOL | Constant output | Empirical | Tian et al. find EMA is not necessary; the predictor and stop-gradient matter. C-JEPA reports that EMA alone does not reliably prevent collapse |
| Stop-gradient plus predictor | SimSiam | Constant output | Linear-model theory (Tian 2021) | Not proven for deep nonlinear networks |
| Centering and sharpening | DINO | One dimension dominating; uniform output | Heuristic | Temperature and momentum must balance |
| Sinkhorn balancing | SwAV | Cluster collapse | Empirical | Assumes uniform use of prototypes |
| Variance and covariance terms | VICReg, VCReg | Dimensional collapse | Second moments only | PLDM uses six loss weights |
| Cross-correlation to identity | Barlow Twins | Redundancy | Empirical | Needs a wide projector and a large batch |
| Whitening | W-MSE | Collapse | Empirical | Batch must exceed the dimension |
| **SIGReg** | LeJEPA | Collapse of all kinds, as a distribution constraint | **Theory**: isotropic Gaussian minimises bias for k-NN and kernel probes (Theorem 1); bounded gradients (Theorem 4); O(1/N) minibatch bias (Theorem 6). **No theorem covers masked or temporal prediction or deep nonlinear training** | Marginal Gaussianisation can hurt structured data (temporally centred SIGReg paper); guarantee is distributional, not semantic |
| Frozen teacher | SALT | Collapse, by construction | Authors' claim | Needs a first-stage teacher |
| Fixed hand-made target | MaskFeat (HOG) | Collapse, by construction | Empirical | HOG carries little signal on flat regions |
| Auxiliary dynamics terms (IDM, temporal alignment) | PLDM, H-JEPA, Delta-JEPA | Slow-feature collapse, action-insensitive latents | Empirical | Not themselves anti-collapse |

### Monitors

- Effective rank (RankMe) and the LDA-rank predictor LiDAR are in `stable-pretraining` and run at validation time. alpha-ReQ and KoLeo are not installed. **[Checked]**

## 2. Candidates (A and C, plus B with temporal pairs)

| # | Scheme | Year, id | Target and anti-collapse | In installed packages | Fit for our setting |
| --- | --- | --- | --- | --- | --- |
| 1 | I-JEPA | 2023, 2301.08243 | Teacher latents of 4 target blocks; EMA, stop-gradient | `stable-pretraining` `IJEPA` **[Checked]** | Established. Large blocks over smooth floor and wall are cheap to predict, so little signal; small objects weak **[Memory]** |
| 2 | SALT | 2025, 2509.24317 | A frozen teacher trained first with a pixel MAE; student predicts its latents; no EMA | `SALT` **[Checked]** | No schedule to tune; teacher stage uses pixels |
| 3 | data2vec 1 and 2 | 2022, 2202.03555 and 2212.07525 | Average of the teacher's top-K layer outputs at masked positions; EMA | `Data2Vec` **[Checked]** | Averaged targets avoid trivial targets **[Memory]** |
| 4 | C-JEPA | 2024, 2410.19560 | I-JEPA plus VICReg | No | Addresses EMA's weakness; custom code |
| 5 | Latent MIM | 2024, 2407.15837 | Latents with a cross-attention decoder and a patch-discrimination loss | No | Reports redundancy between visible and target patches must be controlled |
| 6 | SIGReg with stop-gradient masked targets | untested for vision | The network's own latents at masked positions, detached; SIGReg | No (needs custom code) | Your choice; only protein-sequence (ProteinJEPA, 2605.07554) and 4D point-cloud (2608.24093) abstracts found |
| 7 | V-JEPA 2.1 | 2026, 2603.14482 | Loss on all tokens plus deep supervision | No | Targets the dense-feature gap, relevant to small objects |
| 8 | V-JEPA / V-JEPA 2 | 2024, 2404.08471; 2025, 2506.09985 | Masked space-time regions; EMA | No (only `ijepa.py`) | Adds temporal context to masking; frozen after pretraining, then an action-conditioned stage |
| 9 | LeWM | 2026, 2603.19312 | Next embedding from the action; SIGReg | `stable-worldmodel` `wm/lewm` **[Checked]** | Strategy A; validated on 2D and 3D control |
| 10 | PLDM | 2025, 2502.14819 | Next latent; VICReg-style terms plus inverse dynamics | `wm/pldm` **[Checked]** | Strategy A with six loss weights |
| 11 | VICReg, TiCo, Barlow Twins with temporally adjacent frames as views | 2021, 2022 | Variance and covariance terms | `VICReg`, `TiCO`, `BarlowTwins` **[Checked]** | No augmentation needed, but adjacent frames are nearly identical (risk 1 below) |
| 12 | LeJEPA with temporal pairs | 2025, 2511.08544 | Prediction to the other view plus SIGReg | `LeJEPA` **[Checked]** | Same risk as row 11 |
| 13 | Delta-JEPA, physically grounded JEPA, MotionJEPA | 2026 | Latent prediction plus action decoding, inverse dynamics, or slow-feature fix | No | Address action-insensitive and slow-feature collapse; abstract-level only |
| 14 | DINO-WM | 2024, 2411.04983 | Frozen DINOv2 features | `wm/prejepa` **[Checked]** | Large pretrained encoder; our 48x64 frames are far from its domain |

### Methods that need augmented views

SimCLR, MoCo v1 to v3, NNCLR, PIRL, BYOL, SimSiam, DINO, DINOv2, DINOv3, iBOT, MSN, PMSN, SwAV and MC-JEPA depend on crops, flips or colour changes. Our constraint forbids them: flips would reverse the spin direction later, and crops break the fixed camera geometry. LeJEPA also needs views.

## 3. How a masked-latent scheme works (procedural)

1. Split the image into patches and hide some. 
2. Encode the visible patches to context embeddings. 
3. A small predictor guesses the embeddings of the hidden patches from the context and the hidden patches' positions. 
4. A target network (EMA copy, frozen teacher, or the same network with the gradient stopped) embeds the full image to give the true targets. 
5. The loss is the distance between guesses and targets; a regulariser or the target's design prevents collapse.
6. After training, the encoder is kept and the predictor is discarded.

## 4. Evidence gaps and risks

- **[Unverified]** No paper compares, on the same data, an encoder pretrained with masking and then frozen against one trained jointly with next-latent prediction. All comparisons found are whole-system comparisons with large pretrained encoders.
- **[Unverified]** No test of any of these methods on tiny (48x64), egocentric, deterministically rendered frames.
- No SIGReg theorem covers masked prediction. One abstract-level masked-latent-plus-SIGReg result exists outside vision (4D point clouds).
- **Predictability risk (Latent MIM):** smooth regions are nearly perfectly predictable from their neighbours, so the loss is dominated by trivial patches; small objects may be ignored. The V-JEPA 2.1 paper reports that masked-only loss does not give reliable dense features **[Memory]**.
- **Slow-feature bias:** temporal and action-conditioned prediction pulls toward slowly changing features (MotionJEPA, Sobal et al.), which matters for jittery random walks **[Unverified]**.
- **Risks with temporal pairs as views:** adjacent frames are near-identical, so invariance is trivial and only the spread terms shape features; a deterministic renderer has no nuisance variation, so invariance can erase small fast-changing features; random walks revisit places, so distant frames may duplicate. A larger or random frame gap and an auxiliary dynamics term are suggested.
- SIGReg's guarantee is distributional: non-collapse does not make the features predict the right factors (two 2026 abstract-level papers claim this, **[Unverified]**).

## 5. What is available out of the box

- **`stable-pretraining` methods** **[Checked]**: BarlowTwins, BEiT, BYOL, CMAE, CrossMAE, Data2Vec, DINO, DINOv2, DINOv3, iBOT, iGPT, IJEPA, MAE, LeJEPA, MaskFeat, MIMRefiner, MoCoV2, MoCoV3, MSN, NEPA, NNCLR, PIRL, PMSN, SALT, SimCLR, SimMIM, SimSiam, SwAV, TiCo, VICReg, VICRegL, WMSE. Not present: V-JEPA, CAE, BootMAE, MC-JEPA, LeWM, C-JEPA, Latent MIM, VCReg as a method.
- **`stable-worldmodel` `wm/`** **[Checked]**: lewm, pldm, prejepa (DINO-WM), gcrl, tdmpc2, genie; losses `SIGReg`, `VCReg`, `PLDMLoss`, `TemporalStraighteningLoss`.

## 6. Reading for our decision

- **[Proposed]** Augmentation-free candidates are families A and C and, with caveats, family B using temporal pairs.
- **[Proposed]** Cheapest pilots in the installed stack: I-JEPA (EMA), SALT, data2vec, VICReg and LeJEPA with a frame gap, LeWM (strategy A).
- **[Proposed]** The SIGReg-with-stop-gradient-masked scheme needs our own code and has no vision precedent; keep it as a challenger.
- **[Open]** The choice, pending the predictability study (INT-65) and Kumar's decision.
