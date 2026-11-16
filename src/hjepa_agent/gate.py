"""Stage-1 exit gate (INT-83): turns a run's summary and probe results into pass or fail checks."""

from hjepa_agent.recipe import Recipe

R2_MIN = 0.90      # position and heading probes
AUC_MIN = 0.95     # room probe
R2_KEYS = ["pos_x", "pos_y", "heading_cos", "heading_sin"]
AUC_KEYS = ["room2", "food_visible"]  # food visibility has no absolute threshold: it must beat both baselines


def check(name, value, rule, ok):
    return {"check": name, "value": None if value is None else round(float(value), 4), "rule": rule, "pass": bool(ok)}


def evaluate(summary, probes, pixels, random, recipe=Recipe()):
    """summary: summary.json of the run; probes, pixels, random: eval_probes results for the checkpoint latents,
    raw pixels and a random-init model. Any of the three may be None (the check then fails as missing)."""
    out, m = [], (summary or {}).get("metrics", {})
    out.append(check("run finished without collapse or failure", None, "status is complete", (summary or {}).get("status") == "complete"))
    for k in R2_KEYS:
        v = (probes or {}).get("r2", {}).get(k)
        out.append(check(f"probe R2 {k}", v, f">= {R2_MIN}", v is not None and v >= R2_MIN))
    v = (probes or {}).get("auc", {}).get("room2")
    out.append(check("probe AUC room2", v, f">= {AUC_MIN}", v is not None and v >= AUC_MIN))
    for kind, key in [("r2", k) for k in R2_KEYS] + [("auc", k) for k in AUC_KEYS]:
        v = (probes or {}).get(kind, {}).get(key)
        base = [b[kind][key] for b in (pixels, random) if b and key in b.get(kind, {})]
        out.append(check(f"beats both baselines: {kind} {key}", v, "above pixels and random-init", v is not None and len(base) == 2 and v > max(base)))
    cr, sens = m.get("val/copy_ratio"), m.get("val/action_sensitivity")
    out.append(check("prediction error below copy-the-last-latent", cr, "val copy ratio < 1", cr is not None and cr < 1))
    out.append(check("predictor uses the action", sens, "shuffled-action error / true-action error > 1", sens is not None and sens > 1))
    rank, sig = m.get("val/eff_rank"), m.get("val/sigreg")
    out.append(check("effective rank healthy", rank, f">= {recipe.collapse_rank_min_frac * recipe.embed_dim:.0f}", rank is not None and rank >= recipe.collapse_rank_min_frac * recipe.embed_dim))
    out.append(check("SIGReg statistic healthy", sig, f"<= {recipe.collapse_sigreg_max}", sig is not None and sig <= recipe.collapse_sigreg_max))
    return out
