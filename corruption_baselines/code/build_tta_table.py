#!/usr/bin/env python3
"""Final ImageNet-C corruption table: base / +TTA / +AEGIS, 3 seeds, paired task-bootstrap 95% CI.
Pairing unit = (seed, task_id) success_rate (EP=5 -> rate in {0,.2,.4,.6,.8,1}); base/tta/aegis share
tasks per seed, so the comparison is paired. Emits the LaTeX table body + a plain summary."""
import json, glob, collections
import numpy as np
ROOT = "/home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi/sib_vla/results/act_tta_compare"
rng = np.random.default_rng(0); NB = 10000
def norm(a): return 'tta' if a == 'base_tta' else a

# (suite,corr,sev,arm) -> {(seed,task): rate}
D = collections.defaultdict(dict)
for rj in glob.glob(ROOT + "/*/*/*/*/result.json"):
    p = rj.split("/act_tta_compare/")[1].split("/"); suite=p[0]; arm=norm(p[1]); corr,sev=p[2].rsplit("_s",1); seed=p[3].replace("seed","")
    d = json.load(open(rj))
    if len(d.get("per_task", {})) < 10: continue
    for tid, t in d["per_task"].items():
        D[(suite,corr,sev,arm)][(seed,tid)] = t["success_rate"]

def paired_ci(a_key, b_key):
    A = D[a_key]; B = D[b_key]
    keys = sorted(set(A) & set(B))
    da = np.array([A[k]-B[k] for k in keys])
    obs = da.mean()*100
    idx = rng.integers(0, len(da), size=(NB, len(da)))
    boot = da[idx].mean(axis=1)*100
    return obs, np.percentile(boot,2.5), np.percentile(boot,97.5), np.mean([A[k] for k in keys])*100, np.mean([B[k] for k in keys])*100

CORR = [("gaussian_noise","Gaussian noise"), ("motion_blur","Motion blur"), ("fog","Fog")]
SUITES = ["Spatial","Object"]; SEV = ["3","5"]
rows = []
print(f"{'corr':16}{'sev':4}{'suite':8}{'base':>7}{'tta':>7}{'aegis':>7}{'A-base[CI]':>22}{'A-tta':>8}")
allpairs_ab = []; allpairs_at = []
for cid, cname in CORR:
    for sv in SEV:
        for su in SUITES:
            ka=(su,cid,sv,'aegis'); kb=(su,cid,sv,'base'); kt=(su,cid,sv,'tta')
            if not (D.get(ka) and D.get(kb) and D.get(kt)):
                print(f"  MISSING {su}/{cid}/{sv}"); continue
            dab,lo,hi,ae,ba = paired_ci(ka,kb)
            dat,_,_,_,tt = paired_ci(ka,kt)
            rows.append((cname,sv,su,ba,tt,ae,dab,lo,hi,dat))
            print(f"{cname:16}{sv:>4}{su:>8}{ba:7.0f}{tt:7.0f}{ae:7.0f}   {dab:+5.1f}[{lo:+.1f},{hi:+.1f}]{dat:+8.1f}")
            # pool for net
            kk=sorted(set(D[ka])&set(D[kb])); allpairs_ab += [D[ka][k]-D[kb][k] for k in kk]
            kk2=sorted(set(D[ka])&set(D[kt])); allpairs_at += [D[ka][k]-D[kt][k] for k in kk2]

def net(dl):
    dl=np.array(dl); obs=dl.mean()*100; idx=rng.integers(0,len(dl),size=(NB,len(dl))); bt=dl[idx].mean(axis=1)*100
    return obs, np.percentile(bt,2.5), np.percentile(bt,97.5)
nab=net(allpairs_ab); nat=net(allpairs_at)
print(f"\nNET AEGIS-base: {nab[0]:+.1f} [{nab[1]:+.1f},{nab[2]:+.1f}]  | NET AEGIS-TTA: {nat[0]:+.1f} [{nat[1]:+.1f},{nat[2]:+.1f}]")

# ---- LaTeX body ----
print("\n===LATEX===")
def esc(x): return f"{x:.0f}"
last_corr=None
for (cname,sv,su,ba,tt,ae,dab,lo,hi,dat) in rows:
    star = "\\,*" if (lo>0 or hi<0) else ""
    cshow = cname if cname!=last_corr or sv!=rows[[r[0] for r in rows].index((cname))][1] else cname
    print(f"{cname} & S{sv} & {su} & {ba:.0f} & {tt:.0f} & {ae:.0f} & ${dab:+.1f}$ $[{lo:+.1f},{hi:+.1f}]${star} & ${dat:+.1f}$ \\\\")
print("\\midrule")
print(f"\\multicolumn{{6}}{{l}}{{\\textbf{{Net}} (24 conditions, 3 seeds)}} & $\\mathbf{{{nab[0]:+.1f}}}$ $[{nab[1]:+.1f},{nab[2]:+.1f}]$ & $\\mathbf{{{nat[0]:+.1f}}}$ \\\\")
