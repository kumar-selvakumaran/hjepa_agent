from hjepa_agent.gate import evaluate

GOOD = {"status": "complete", "metrics": {"val/copy_ratio": 0.6, "val/action_sensitivity": 1.3, "val/eff_rank": 90.0, "val/sigreg": 1.5}}
PROBES = {"r2": {"pos_x": 0.95, "pos_y": 0.93, "heading_cos": 0.92, "heading_sin": 0.91}, "auc": {"room2": 0.99, "food_visible": 0.9}}
BASE = {"r2": {"pos_x": 0.4, "pos_y": 0.3, "heading_cos": 0.2, "heading_sin": 0.1}, "auc": {"room2": 0.8, "food_visible": 0.6}}


def test_gate_passes_when_everything_is_good():
    assert all(c["pass"] for c in evaluate(GOOD, PROBES, BASE, BASE))


def test_gate_fails_on_collapse_missing_baselines_and_weak_probes():
    bad = dict(GOOD, status="collapse: sigreg 9", metrics=dict(GOOD["metrics"], **{"val/sigreg": 9.0, "val/eff_rank": 10.0}))
    names = {c["check"] for c in evaluate(bad, PROBES, BASE, BASE) if not c["pass"]}
    assert {"run finished without collapse or failure", "effective rank healthy", "SIGReg statistic healthy"} <= names
    weak = {"r2": dict(PROBES["r2"], pos_x=0.5), "auc": PROBES["auc"]}
    assert any(not c["pass"] and c["check"] == "probe R2 pos_x" for c in evaluate(GOOD, weak, BASE, BASE))
    assert any(not c["pass"] and c["check"].startswith("beats both baselines") for c in evaluate(GOOD, PROBES, None, BASE))
    assert not any(c["pass"] for c in evaluate(None, None, None, None) if c["value"] is not None)
