#!/bin/bash
# ============================================================================
# migrate.sh — push the project from THIS A100 box to the A6000 box.
# Transfer route: Tailscale (this box = 100.64.26.49). rsync = resumable.
#
# USAGE (run FROM this A100 box, AFTER training is done & paused):
#   1) edit DEST_HOST below to the A6000's tailscale name/IP + user
#   2) dry-run first:   bash migrate.sh --dry
#   3) real:            bash migrate.sh
#
# KEEP PATHS IDENTICAL on the A6000 (/home/user/...) or configs + Claude break.
# This copies DATA only. Recreate conda envs from the .yml files (see checklist) —
# do NOT rsync env directories (CUDA/flash-attn are machine-specific).
# ============================================================================
set -uo pipefail
DEST_HOST="user@A6000-HOSTNAME"     # <-- EDIT: tailscale name or 100.64.x.x
DRY=""; [ "${1:-}" = "--dry" ] && DRY="--dry-run" && echo ">>> DRY RUN (nothing copied)"

R="rsync -aP --human-readable $DRY"   # -a archive, -P resume+progress

echo "=== 1/5  Repo (code + outputs/ trained ckpts + results) ~21G  [IRREPLACEABLE] ==="
$R /home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi/ \
   "$DEST_HOST":/home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi/

echo "=== 2/5  GR00T repo ~469M ==="
$R /home/user/vla/Isaac-GR00T/ "$DEST_HOST":/home/user/vla/Isaac-GR00T/

echo "=== 3/5  Claude context (transcript + memory) ~160M ==="
$R /home/user/.claude/projects/-home-user-Desktop-SAPTARSHI-ALT-steer-information-Saptarshi/ \
   "$DEST_HOST":/home/user/.claude/projects/-home-user-Desktop-SAPTARSHI-ALT-steer-information-Saptarshi/
# also settings/skills (optional but handy):
$R /home/user/.claude/settings.json /home/user/.claude/CLAUDE.md \
   "$DEST_HOST":/home/user/.claude/ 2>/dev/null || true

echo "=== 4/5  CRITICAL HF-cache subset only (NOT the full 408G) ~ a few tens of GB ==="
HF=/home/user/.cache/huggingface/hub
for D in \
  models--nvidia--GR00T-N1.5-3B \
  models--lerobot--smolvla_base \
  models--HuggingFaceVLA--smolvla_libero \
  models--HuggingFaceTB--SmolVLM2-500M-Instruct \
  models--distilbert-base-uncased \
  datasets--HuggingFaceVLA--libero \
  datasets--lerobot--libero_10 \
  datasets--lerobot--libero_spatial_image \
  datasets--lerobot--libero-assets ; do
  [ -e "$HF/$D" ] && $R "$HF/$D" "$DEST_HOST":"$HF/" || echo "  (skip, not cached: $D)"
done
# NOTE: bert-base-uncased is NOT cached here; NanoVLA will auto-download it (~440M) on the A6000.

echo "=== 5/5  DONE. On the A6000, follow migration/NEW_MACHINE_CHECKLIST.md ==="
echo ">>> Did NOT copy: conda envs (recreate from .yml), the other 380G of HF cache (re-downloads on demand)."
