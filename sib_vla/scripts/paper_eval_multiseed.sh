#!/bin/bash
# Paper-protocol multi-seed eval (arXiv 2506.01844 §4).
# n=10 trials/task, episode_length=280, n_action_steps=1, init_states 0-9.
# Runs 3 seeds; aggregates at the end.
# Run from: sib_vla/
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 123 456)

for SEED in "${SEEDS[@]}"; do
    TAG="paper_eval_seed${SEED}"
    RESULT="results/eval_${TAG}.json"
    if [ -f "$RESULT" ]; then
        echo "[skip] $TAG already done -> $RESULT"
        continue
    fi
    echo "=== Paper-protocol eval | seed=${SEED} ==="
    # Write a minimal per-seed override into configs/ so load_config can resolve
    # the parent path (configs/paper_eval.yaml) correctly.
    SEEDCFG="configs/_paper_eval_s${SEED}.yaml"
    cat > "$SEEDCFG" << YAML
inherit: paper_eval.yaml
seeds: [${SEED}]
YAML
    python scripts/eval.py --config "$SEEDCFG" --tag "$TAG"
    rm -f "$SEEDCFG"
done

echo ""
echo "=== Aggregated results ==="
python3 - << 'PYEOF'
import json, os, math

seeds = [42, 123, 456]
srs, per_task_all = [], {}

for s in seeds:
    path = f"results/eval_paper_eval_seed{s}.json"
    if not os.path.exists(path):
        print(f"  seed={s}: MISSING")
        continue
    with open(path) as f:
        d = json.load(f)
    sr = d["success_rate"]
    srs.append(sr)
    lo, hi = d["success_wilson95"]
    print(f"  seed={s}: SR={sr:.3f}  CI=[{lo:.3f}, {hi:.3f}]  n={d['n_episodes']}")
    if "per_task" in d:
        for t in d["per_task"]:
            tid = t["task_id"]
            per_task_all.setdefault(tid, []).append(t["success_rate"])

if len(srs) > 1:
    mean = sum(srs) / len(srs)
    std  = math.sqrt(sum((x - mean)**2 for x in srs) / len(srs))
    print(f"\nPaper-protocol (3-seed): {mean:.3f} +/- {std:.3f}")
    if per_task_all:
        print("\nPer-task mean across seeds:")
        for tid in sorted(per_task_all):
            vals = per_task_all[tid]
            print(f"  task_{tid}: {sum(vals)/len(vals):.2f}")
elif srs:
    print(f"\nSingle-seed result: {srs[0]:.3f}")
PYEOF
