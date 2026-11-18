# Predictability study: can masked regions be predicted from visible context?

Run 2026-11-15 (script `scratch/predictability_study.py`, output `scratch/predictability_study.txt`; figures in `../media/`). Labels: see [README.md](README.md). This answers Kumar's concern (INT-65) that visible tokens may carry no information about masked ones.

## Method

- **[Checked]** A small inpainting CNN (0.29M parameters, 3,000 training steps) predicts the pixels of hidden regions from the visible ones. Pixel prediction is a proxy for latent prediction, not the same thing.
- **[Checked]** Data: `transfer_patch` (512 episodes, doorway patch present), 20,000 training frames from 409 episodes, 4,000 validation frames from 103 other episodes.
- **[Checked]** Two masks on the 8x8 grid of 6x8-pixel patches: single random patches (15.6% of pixels hidden) and I-JEPA-like blocks (49.6% hidden). A variant also receives the previous frame.
- **[Checked]** Metric: R squared = 1 minus hidden-pixel error over hidden-pixel variance. **Per-class numbers use the overall hidden-pixel variance as the denominator**, so they show each class's error relative to the whole, not a within-class R squared (the within-class figures in the raw output are meaningless for near-constant colours).

## Results (overall R squared)

| Predictor | Single patches | Blocks |
| --- | --- | --- |
| Mean colour | 0.00 | 0.00 |
| Mean of the visible pixels in the same row | 0.31 | 0.03 |
| Nearest visible pixel | 0.65 | 0.14 |
| CNN, spatial context only | 0.88 | 0.72 |
| CNN plus the previous frame | 0.92 | 0.90 |

## Per class (CNN, then CNN plus previous frame; shared denominator)

| Class (share of hidden pixels) | Single | Blocks |
| --- | --- | --- |
| Floor (42 to 44%) | 0.94, 0.96 | 0.86, 0.95 |
| Wall (31 to 34%) | 0.97, 0.98 | 0.92, 0.97 |
| Objects: obstacles and blue blocks (6%) | 0.71, 0.82 | 0.24, 0.78 |
| Doorway patch (11%) | 0.75, 0.82 | 0.48, 0.77 |
| Own body (3 to 4%) | 0.99, 0.99 | 0.98, 0.99 |
| **Food discs (0.3 to 0.4%)** | **-0.55, -0.15** | **-1.03, -0.18** |

## Reading

- **Structure is predictable from spatial context.** Floor, wall and own body reach 0.86 to 0.99 even with half the image hidden. The fear that visible tokens say nothing about hidden ones does not hold for the large structured parts of the view.
- **Small isolated objects are not.** Food discs are predicted worse than a constant colour (negative values), with or without the previous frame. They are about 0.3% of hidden pixels. Objects and the patch are only partly predictable from spatial context in block masks (0.24 and 0.48).
- **The previous frame adds a lot** (blocks: overall 0.72 to 0.90, objects 0.24 to 0.78): consecutive views overlap heavily, so temporal context makes hidden content largely recoverable.
- **Consequence:** an objective that learns only what is predictable will encode geometry well and may drop food discs, the likely hunger cue of later stages (predictability versus relevance, concern C3 of the strategies document).

## Caveats

- **[Unverified]** A pixel proxy: latents are smoother than pixels, so latent predictability may differ.
- **[Unverified]** The CNN is small and trained briefly, so these are lower bounds on predictability, not limits.
- The patch is present in these frames (not the pretraining world), which raises its share; the validation frames come from episodes unseen in training.
