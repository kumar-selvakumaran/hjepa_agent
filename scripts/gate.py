"""Apply the stage-1 exit gate to a finished run (INT-83); prints a pass/fail table, writes <run>/gate.json, exits 1 on failure.

uv run python scripts/eval_probes.py --data <shards> --features ckpt --ckpt <run>/checkpoints/final.ckpt --out <run>/probes_ckpt.json
uv run python scripts/eval_probes.py --data <shards> --features pixels --out <run>/probes_pixels.json
uv run python scripts/eval_probes.py --data <shards> --features random --out <run>/probes_random.json
uv run python scripts/gate.py --run <run>"""

import argparse
import json
import sys
from pathlib import Path

from hjepa_agent.gate import evaluate


def read(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", nargs="*", help="unused; accepted so the step file can pass the shard names")
    a = ap.parse_args()
    checks = evaluate(read(a.run / "summary.json"), read(a.run / "probes_ckpt.json"), read(a.run / "probes_pixels.json"), read(a.run / "probes_random.json"))
    for c in checks:
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['check']:<55} value={c['value']}  rule: {c['rule']}")
    ok = all(c["pass"] for c in checks)
    (a.run / "gate.json").write_text(json.dumps({"pass": ok, "checks": checks}, indent=1))
    print("GATE PASSED" if ok else "GATE FAILED: stop and ask (INT-63); do not spend more compute before deciding")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
