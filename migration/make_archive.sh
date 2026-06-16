#!/bin/bash
# ============================================================================
# make_archive.sh — single-file transfer archive for USB/cloud (when rsync isn't an option).
# Uses tar (NOT zip): checkpoints are incompressible, so we don't waste CPU compressing them.
#
#   bash make_archive.sh            # CORE  ~22G: repo (code+ckpts) + GR00T repo + Claude context
#   FULL=1 bash make_archive.sh     # FULL  ~50G: CORE + critical HF-cache subset (offline-ready)
#   SPLIT=1 bash make_archive.sh    # also split into 5G parts for FAT32/upload limits
#
# Output goes to /home/user/Desktop/SAPTARSHI_ALT/transfer/ (OUTSIDE the repo -> no recursion).
# ============================================================================
set -uo pipefail
DEST=/home/user/Desktop/SAPTARSHI_ALT/transfer
mkdir -p "$DEST"
TAR="$DEST/saptarshi_transfer.tar"
ZST=""; command -v zstd >/dev/null && ZST="--zstd" && TAR="$TAR.zst"   # zstd: fast, helps the code/text parts only

EXCL="--exclude=__pycache__ --exclude=*.pyc --exclude=.git --exclude=transfer"
ITEMS=(
  /home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi
  /home/user/vla/Isaac-GR00T
  /home/user/.claude/projects/-home-user-Desktop-SAPTARSHI-ALT-steer-information-Saptarshi
)
if [ "${FULL:-0}" = "1" ]; then
  HF=/home/user/.cache/huggingface/hub
  for D in models--nvidia--GR00T-N1.5-3B models--lerobot--smolvla_base \
           models--HuggingFaceVLA--smolvla_libero models--HuggingFaceTB--SmolVLM2-500M-Instruct \
           models--distilbert-base-uncased datasets--HuggingFaceVLA--libero \
           datasets--lerobot--libero_10 datasets--lerobot--libero_spatial_image \
           datasets--lerobot--libero-assets; do
    [ -e "$HF/$D" ] && ITEMS+=("$HF/$D")
  done
fi

echo "[archive] building $TAR  (FULL=${FULL:-0})  $(date '+%F %T')"
echo "[archive] contents:"; printf '   %s\n' "${ITEMS[@]}"
# -P keeps absolute paths so extraction restores the SAME locations (paths must match!)
tar $ZST $EXCL -cvPf "$TAR" "${ITEMS[@]}" >"$DEST/manifest.txt" 2>&1
echo "[archive] size: $(du -h "$TAR" | cut -f1)   (file list -> $DEST/manifest.txt)"

if [ "${SPLIT:-0}" = "1" ]; then
  echo "[archive] splitting into 5G parts ..."
  split -b 5G -d "$TAR" "$TAR.part_"
  echo "[archive] parts: $(ls "$TAR".part_* | wc -l)  (rejoin: cat $TAR.part_* > $TAR)"
fi

cat <<EOF

[archive] DONE. Transfer "$TAR" to the A6000 (USB / scp / cloud), then on the A6000:
   cd / && tar $ZST -xvPf /path/to/$(basename "$TAR")     # -P restores absolute paths
Then follow migration/NEW_MACHINE_CHECKLIST.md (rebuild envs) and run a6000_queue.sh.
Note: extraction needs another ~$(du -sh --apparent-size "$TAR" 2>/dev/null | cut -f1) free on the A6000.
EOF
