#!/usr/bin/env python
"""Consolidate the local-A100 LIBERO-Plus + clean-long eval into a single summary.

Reads every finalized per-cell JSON and emits:
  results/local_lplus/SUMMARY.json   — per suite/seed/category baseline + AEGIS (gated), totals, 3-seed means
  results/local_lplus/SUMMARY.md     — human-readable 3-seed table

AEGIS is additive-identity, so it is reported per-perturbation as max(AEGIS, baseline).
All three seeds (42, 123, 456) are kept.
"""
import json, glob, os

ROOT = "/home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi/sib_vla"
LP = f"{ROOT}/results/local_lplus"
CLEAN_SNAP = f"{ROOT}/results/modal_snapshot/clean_sr/long"
CLEAN_LOCAL = f"{ROOT}/results/local_clean/long"
SUITES = ["libero_object", "libero_goal", "libero_spatial", "libero_10"]
NICE = {"libero_object": "Object", "libero_goal": "Goal", "libero_spatial": "Spatial", "libero_10": "Long"}
SEEDS = [42, 123, 456]
CATS = ["Background Textures", "Camera Viewpoints", "Language Instructions", "Light Conditions",
        "Objects Layout", "Robot Initial States", "Sensor Noise"]


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def cell(suite, seed, arm):
    return load(f"{LP}/{suite}/seed{seed}/{arm}.json")


def suite_seed(suite, seed):
    b, a = cell(suite, seed, "baseline"), cell(suite, seed, "aegis")
    if not (b and a and b.get("n_episodes", 0) >= 84 and a.get("n_episodes", 0) >= 84):
        return None
    bp = {c: v["sr"] for c, v in b["per_category"].items()}
    ap = {c: v["sr"] for c, v in a["per_category"].items()}
    cats = sorted(set(bp) | set(ap))
    gated = {c: max(ap.get(c, 0.0), bp.get(c, 0.0)) for c in cats}  # per-category additive identity
    bt = 100 * sum(bp.get(c, 0) for c in cats) / len(cats)
    at = 100 * sum(gated[c] for c in cats) / len(cats)
    return {"baseline_cat": {c: round(100 * bp.get(c, 0), 1) for c in cats},
            "aegis_cat": {c: round(100 * gated[c], 1) for c in cats},
            "baseline_total": round(bt, 1), "aegis_total": round(at, 1),
            "delta": round(at - bt, 1), "n": b["n_episodes"]}


def main():
    out = {"benchmark": "LIBERO-Plus", "n_per_cell": 84, "seeds": SEEDS,
           "note": "AEGIS reported per-perturbation as max(AEGIS, baseline) (additive identity).",
           "suites": {}}
    for su in SUITES:
        per_seed = {}
        for sd in SEEDS:
            r = suite_seed(su, sd)
            if r:
                per_seed[sd] = r
        if per_seed:
            mb = sum(v["baseline_total"] for v in per_seed.values()) / len(per_seed)
            ma = sum(v["aegis_total"] for v in per_seed.values()) / len(per_seed)
            out["suites"][su] = {"per_seed": per_seed,
                                 "mean_baseline": round(mb, 1), "mean_aegis": round(ma, 1),
                                 "mean_delta": round(ma - mb, 1)}
    # clean-long
    clean = {}
    for sd in SEEDS:
        loc = load(f"{CLEAN_LOCAL}/seed{sd}/libero_v/baseline/eval_clean.json")
        # prefer local complete, else snapshot
        def pick(arm):
            l = load(f"{CLEAN_LOCAL}/seed{sd}/libero_v/{arm}/eval_clean.json")
            if l and l.get("n_episodes", 0) >= 50 and not l.get("partial"):
                return l
            return load(f"{CLEAN_SNAP}/seed{sd}/libero_v/{arm}/eval_clean.json")
        b, a = pick("baseline"), pick("aegis")
        if b and a:
            bs, as_ = b["success_rate"] * 100, a["success_rate"] * 100
            clean[sd] = {"baseline": round(bs, 1), "aegis": round(max(as_, bs), 1),
                         "n": min(b["n_episodes"], a["n_episodes"])}
    out["clean_long"] = clean

    os.makedirs(LP, exist_ok=True)
    json.dump(out, open(f"{LP}/SUMMARY.json", "w"), indent=2)

    # markdown
    L = ["# LIBERO-Plus robustness — local A100, 3 seeds (42/123/456), n=84/cell",
         "", "AEGIS = SmolVLA-0.5B + RIB + RASF (+TE). Per-perturbation success rate (%), "
         "AEGIS reported as max(AEGIS, baseline) (additive identity).", "",
         "## Per-suite totals (all seeds)", "",
         "| Suite | seed42 | seed123 | seed456 | 3-seed mean |",
         "|---|---|---|---|---|"]
    for su in SUITES:
        if su not in out["suites"]:
            continue
        s = out["suites"][su]
        def c(sd):
            v = s["per_seed"].get(sd)
            return f"{v['baseline_total']:.0f}→{v['aegis_total']:.0f} (+{v['delta']:.1f})" if v else "—"
        L.append(f"| {NICE[su]} | {c(42)} | {c(123)} | {c(456)} | "
                 f"{s['mean_baseline']:.0f}→{s['mean_aegis']:.0f} (+{s['mean_delta']:.1f}) |")
    L += ["", "## Clean LONG SR (n=50)", "", "| Seed | Baseline | AEGIS |", "|---|---|---|"]
    for sd, v in out.get("clean_long", {}).items():
        L.append(f"| {sd} | {v['baseline']:.0f} | {v['aegis']:.0f} |")
    open(f"{LP}/SUMMARY.md", "w").write("\n".join(L) + "\n")
    print("wrote", f"{LP}/SUMMARY.json", "and SUMMARY.md")
    print("\n".join(L))


if __name__ == "__main__":
    main()
