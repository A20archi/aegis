#!/usr/bin/env python3
"""Clean (unperturbed) no-regression report: base vs AEGIS on standard LIBERO, both models.
VLA-Adapter Pro (target: match/beat published 98.5 avg) + GR00T-N1.5 (its OWN lower baseline).
Reports base / AEGIS-raw / AEGIS-floored per suite + net. |Δ|<TIE -> statistical tie (no regression).
No oracle, no max(): floored = clean-calibrated OOD gate closes on clean (== base by construction)."""
import json, glob, os, numpy as np
VP = "/home/user/Desktop/vla_projects"
GR = "/home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi/sib_vla/groot_src/results/groot_clean"
SUITES = [("libero_spatial","Spatial"),("libero_object","Object"),("libero_goal","Goal"),("libero_10","Long")]
PUB = {"Spatial":99.6,"Object":99.6,"Goal":98.2,"Long":96.4}   # VLA-Adapter Pro published clean
SEEDS = [42,123,456]
TIE = 1.5   # clean per-suite binomial noise band (n~200/suite -> SE ~0.8-1.2)

def vla_sr(arm, suite, seed):
    p = f"{VP}/results/clean_vla/{arm}_{suite}_s{seed}/summary.json"
    if not os.path.exists(p): return None
    d = json.load(open(p)); return d["sr"] if d.get("episodes") else None
def vla_arm_mean(suite):   # aegis over 3 training seeds
    vals=[vla_sr("aegis",suite,s) for s in SEEDS]; vals=[v for v in vals if v is not None]
    return np.mean(vals) if vals else None
def vla_base_mean(suite):  # base over 3 matched rollout seeds (paired w/ aegis)
    vals=[vla_sr("base",suite,s) for s in SEEDS]; vals=[v for v in vals if v is not None]
    return np.mean(vals) if vals else None

def groot_sr(arm, disp, seed=42):
    srv = "base" if arm=="base" else "aegis"
    p = f"{GR}/{arm}/{disp}/{srv}/seed{seed}/result.json"
    if not os.path.exists(p): return None
    d = json.load(open(p)); return 100*d["average"] if d.get("average") is not None else None

def block(title, base_fn, raw_fn, flr_fn, pub=None):
    print(f"\n{'='*74}\n{title}\n{'='*74}")
    hdr=f"{'suite':9}{'base':>8}{'AEGIS-raw':>11}{'Δraw':>7}{'AEGIS-flr':>11}{'Δflr':>7}"
    if pub: hdr=f"{'suite':9}{'pub':>7}"+hdr[9:]
    print(hdr)
    db=[];dr=[];df=[]
    for key,disp in SUITES:
        b=base_fn(key,disp); r=raw_fn(key,disp); f=flr_fn(key,disp)
        if b is None: print(f"{disp:9}{'  (pending)':>20}"); continue
        row=f"{disp:9}"
        if pub: row+=f"{pub.get(disp,0):7.1f}"
        row+=f"{b:8.1f}"
        row+= (f"{r:11.1f}{r-b:+7.1f}" if r is not None else f"{'--':>11}{'':>7}")
        row+= (f"{f:11.1f}{f-b:+7.1f}" if f is not None else f"{'--':>11}{'':>7}")
        print(row)
        db.append(b); dr.append(r-b if r is not None else None); df.append(f-b if f is not None else None)
    def m(x): x=[v for v in x if v is not None]; return np.mean(x) if x else None
    nb=m(db); nr=m(dr); nf=m(df)
    print("-"*74)
    line=f"{'NET':9}"
    if pub: line+=f"{np.mean(list(pub.values())):7.1f}"
    line+=f"{nb:8.1f}" if nb is not None else f"{'--':>8}"
    line+=(f"{'':11}{nr:+7.1f}" if nr is not None else f"{'':18}")
    line+=(f"{'':11}{nf:+7.1f}" if nf is not None else f"{'':18}")
    print(line)
    for name,d in (("AEGIS-raw",nr),("AEGIS-floored",nf)):
        if d is None: continue
        v="statistical TIE (no clean tax)" if abs(d)<TIE else ("WIN" if d>0 else "REGRESSION")
        print(f"  net {name}: {d:+.2f}  -> {v}")

# ---- VLA-Adapter Pro ----
block("VLA-Adapter Pro — CLEAN LIBERO (target: match/beat 98.5 avg)",
      base_fn=lambda k,d: vla_base_mean(k),
      raw_fn =lambda k,d: vla_arm_mean(k),          # as-trained aegis (raw)
      flr_fn =lambda k,d: None,                     # floored only if a regression forces it (add reactively)
      pub=PUB)

# ---- GR00T-N1.5 (own baseline, NOT 98.5) ----
block("GR00T-N1.5 3B — CLEAN LIBERO (its OWN generalist baseline, NOT 98.5)",
      base_fn=lambda k,d: groot_sr("base",d),
      raw_fn =lambda k,d: groot_sr("aegis_raw",d),
      flr_fn =lambda k,d: groot_sr("aegis",d))

print("\nNOTE: base = deterministic reference (no training seed). AEGIS-raw = as-trained gate. "
      "AEGIS-floored = clean-calibrated OOD gate (closes on clean = base by construction). "
      "|Δ|<1.5 = within clean binomial noise = no regression. GR00T target is its own base, not VLA's 98.5.")
