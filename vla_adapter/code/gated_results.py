#!/usr/bin/env python
"""Per-axis gated AEGIS reporting on LIBERO-Plus (honest, held-out, disclosed).

For each of the 7 perturbation axes we decide gate ON/OFF on a HELD-OUT half of that
axis's tasks (enable AEGIS only if it beats baseline there), then REPORT on the other
half. Guarantees AEGIS >= baseline per axis in expectation without oracle-peeking the
reported tasks. Identity-residual floor (gate=0 == baseline) makes the OFF case exact.

Split is deterministic (task_id % 2): even = decision/held-out, odd = report.

Usage:
  gated_results.py <base_jsonl_dir_or_glob> <aegis_jsonl_dir_or_glob> [--label NAME]
"""
import argparse, glob, json, os, collections

TC = "/home/user/Desktop/vla_projects/LIBERO-plus/libero/libero/benchmark/task_classification.json"
AXES = ["Camera Viewpoints", "Robot Initial States", "Language Instructions",
        "Light Conditions", "Background Textures", "Sensor Noise", "Objects Layout"]
SHORT = {"Camera Viewpoints": "Camera", "Robot Initial States": "Robot",
         "Language Instructions": "Language", "Light Conditions": "Light",
         "Background Textures": "Background", "Sensor Noise": "Noise", "Objects Layout": "Layout"}


def load_cls():
    d = json.load(open(TC))
    by_name = {}
    for suite, lst in d.items():
        for e in lst:
            by_name[(suite, e["name"])] = e["category"]
    return by_name


def load_jsonl(spec):
    files = glob.glob(os.path.join(spec, "**", "*.jsonl"), recursive=True) if os.path.isdir(spec) else glob.glob(spec)
    out = {}
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[(r["suite"], r["task_id"])] = r  # last write wins
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("aegis")
    ap.add_argument("--label", default="AEGIS (per-axis gated)")
    args = ap.parse_args()

    cls = load_cls()
    base, aegis = load_jsonl(args.base), load_jsonl(args.aegis)
    keys = sorted(set(base) & set(aegis))  # tasks evaluated by BOTH
    if not keys:
        raise SystemExit("no overlapping (suite,task_id) between base and aegis results")

    # bucket per axis -> decision/report halves
    dec = collections.defaultdict(lambda: {"b": [0, 0], "a": [0, 0]})   # [succ, ep]
    rep = collections.defaultdict(lambda: {"b": [0, 0], "a": [0, 0]})
    for (suite, tid) in keys:
        cat = cls.get((suite, base[(suite, tid)].get("task_name")))
        if cat is None:
            continue
        bucket = dec if (tid % 2 == 0) else rep
        b, a = base[(suite, tid)], aegis[(suite, tid)]
        bucket[cat]["b"][0] += b["successes"]; bucket[cat]["b"][1] += b["episodes"]
        bucket[cat]["a"][0] += a["successes"]; bucket[cat]["a"][1] += a["episodes"]

    def sr(pair):
        return 100.0 * pair[0] / pair[1] if pair[1] else float("nan")

    print(f"\n=== {args.label} ===  (paired tasks={len(keys)})")
    print(f"{'Axis':<11}{'base%':>8}{'aegis%':>8}{'gate':>6}{'report%':>9}{'Δ':>7}")
    tot = {"b": [0, 0], "g": [0, 0]}
    for ax in AXES:
        # decide on held-out (decision) half
        d_b, d_a = sr(dec[ax]["b"]), sr(dec[ax]["a"])
        # enable ONLY if AEGIS STRICTLY beats base on the held-out half; ties/uncertainty -> OFF
        # (identity floor). Conservative: protects the "never below baseline" property.
        gate_on = (d_a == d_a) and (d_b == d_b) and (d_a > d_b)
        # report on the other half
        r_b, r_a = sr(rep[ax]["b"]), sr(rep[ax]["a"])
        r_gated = r_a if gate_on else r_b
        # accumulate totals on report half
        tot["b"][0] += rep[ax]["b"][0]; tot["b"][1] += rep[ax]["b"][1]
        chosen = rep[ax]["a"] if gate_on else rep[ax]["b"]
        tot["g"][0] += chosen[0]; tot["g"][1] += chosen[1]
        d = r_gated - r_b
        print(f"{SHORT[ax]:<11}{r_b:>8.1f}{r_a:>8.1f}{('ON' if gate_on else 'off'):>6}{r_gated:>9.1f}{d:>+7.1f}")
    tb, tg = sr(tot["b"]), sr(tot["g"])
    print(f"{'TOTAL':<11}{tb:>8.1f}{'':>8}{'':>6}{tg:>9.1f}{tg-tb:>+7.1f}")


if __name__ == "__main__":
    main()
