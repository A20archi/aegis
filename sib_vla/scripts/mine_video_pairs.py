"""Find VERIFIED baseline-fail / AEGIS-succeed video pairs per condition.
Scans npz `success` flags; only keeps episodes that actually have an .mp4.
Extracts a poster frame (mid-clip) for each chosen mp4. Writes a manifest.
"""
import os, glob, json, subprocess
import numpy as np

BASE = "results/videos/libv_baseline_libero_spatial"
AEG  = "results/videos/libv_aegis_libero_spatial"
POSTER = "presentation/posters"; os.makedirs(POSTER, exist_ok=True)

# robustness conditions to cover (clean handled separately)
CONDS = ["motion_blur_1","gaussian_noise_1","lighting_1","texture_1",
         "viewpoint_medium","viewpoint_large","object_offset_5","clean"]

def ep_success(npz):
    try:
        return bool(np.load(npz, allow_pickle=True)["success"])
    except Exception:
        return None

def scan(run, cond):
    """return {task: [(ep_idx, mp4, success)]} for episodes that HAVE an mp4."""
    out={}
    cdir=os.path.join(run, cond)
    if not os.path.isdir(cdir): return out
    for tdir in sorted(glob.glob(os.path.join(cdir, f"{cond}_t*"))):
        task=os.path.basename(tdir)
        lst=[]
        for mp4 in sorted(glob.glob(os.path.join(tdir,"ep*.mp4"))):
            npz=mp4[:-4]+".npz"
            if not os.path.exists(npz): continue
            s=ep_success(npz)
            if s is None: continue
            lst.append((os.path.basename(mp4), mp4, s))
        if lst: out[task]=lst
    return out

def poster(mp4, tag):
    out=os.path.join(POSTER, tag+".jpg")
    # grab a frame ~40% through the clip
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",mp4,
                    "-vf","thumbnail,scale=480:-1","-frames:v","1",out],
                   check=False)
    return out if os.path.exists(out) else None

manifest=[]
for cond in CONDS:
    b=scan(BASE,cond); a=scan(AEG,cond)
    pair=None
    # want a task where baseline has a FAIL mp4 and aegis has a SUCCESS mp4
    for task in sorted(set(b)&set(a)):
        bfail=[e for e in b[task] if e[2] is False]
        asucc=[e for e in a[task] if e[2] is True]
        if bfail and asucc:
            pair=(task, bfail[0], asucc[0]); break
    # fallback: any baseline fail + any aegis (even if not same task) — only if no clean pair
    if pair is None and cond!="clean":
        bf=[(t,e) for t in b for e in b[t] if e[2] is False]
        as_=[(t,e) for t in a for e in a[t] if e[2] is True]
        if bf and as_:
            pair=(bf[0][0]+" / "+as_[0][0], bf[0][1], as_[0][1])
    if pair:
        task,be,ae=pair
        bp=poster(be[1], f"{cond}_base"); ap=poster(ae[1], f"{cond}_aegis")
        manifest.append({"condition":cond,"task":task,
                         "base_mp4":be[1],"base_success":be[2],"base_poster":bp,
                         "aegis_mp4":ae[1],"aegis_success":ae[2],"aegis_poster":ap})
        print(f"[{cond:18s}] base {be[0]}(succ={be[2]}) | aegis {ae[0]}(succ={ae[2]}) | {task}")
    else:
        print(f"[{cond:18s}] NO PAIR (b_tasks={len(b)} a_tasks={len(a)})")

json.dump(manifest, open("presentation/video_pairs.json","w"), indent=2)
print(f"\n{len(manifest)} pairs -> presentation/video_pairs.json")
