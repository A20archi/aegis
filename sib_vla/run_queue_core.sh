#!/bin/bash
# OOM-SAFE dynamic scheduler core. Sourced by stage runners.
# Launches jobs from a queue with parallelism sized to BOTH live free VRAM AND
# live free host RAM:
#   gpu_cap = floor((gpu_free  - GPU_BUFFER) / PER_JOB_MIB)
#   ram_cap = floor((ram_avail - RAM_BUFFER) / RAM_PER_JOB_MIB)
#   max_parallel = min(HARDCAP, gpu_cap, ram_cap)
# Re-evaluated before every launch, so it never oversubscribes GPU *or* host RAM
# and yields to volatile external jobs. A job is (logfile :: command...).
set -uo pipefail
PER_JOB_MIB=${PER_JOB_MIB:-15000}       # GPU footprint per eval ~14GB + headroom
BUFFER_MIB=${BUFFER_MIB:-9000}          # keep this much VRAM free for external jobs
RAM_PER_JOB_MIB=${RAM_PER_JOB_MIB:-32000}  # host-RAM per eval (~20GB RSS) + headroom
RAM_BUFFER_MIB=${RAM_BUFFER_MIB:-60000}    # keep 60GB host RAM free for system + external
HARDCAP=${HARDCAP:-4}                    # never exceed this many of MY jobs at once
POLL=${POLL:-20}

free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }
ram_avail_mib(){ free -m 2>/dev/null | awk '/^Mem:/{print $7}'; }   # "available" column
running_jobs(){ jobs -rp | wc -l | tr -d ' '; }
allowed_now(){
  local f r gpu_cap ram_cap
  f=$(free_mib);      f=${f:-0}
  r=$(ram_avail_mib); r=${r:-0}
  gpu_cap=$(( (f - BUFFER_MIB) / PER_JOB_MIB ));        [ "$gpu_cap" -lt 0 ] && gpu_cap=0
  ram_cap=$(( (r - RAM_BUFFER_MIB) / RAM_PER_JOB_MIB )); [ "$ram_cap" -lt 0 ] && ram_cap=0
  local cap=$gpu_cap
  [ "$ram_cap" -lt "$cap" ] && cap=$ram_cap     # min(gpu, ram)
  [ "$cap" -gt "$HARDCAP" ] && cap=$HARDCAP
  echo "$cap"
}

# run_stage <name> <joblist-file>   (joblist: one job per line: "LOG\tCMD")
run_stage(){
  local name="$1" jl="$2" total done=0
  total=$(grep -cve '^[[:space:]]*$' "$jl")
  echo "[$(date +%T)] === STAGE: $name ($total jobs) ==="
  local -a CMDS LOGS
  while IFS=$'\t' read -r lg cmd; do
    [ -z "${cmd:-}" ] && continue
    LOGS+=("$lg"); CMDS+=("$cmd")
  done < "$jl"
  local i=0 n=${#CMDS[@]}
  while [ "$i" -lt "$n" ] || [ "$(running_jobs)" -gt 0 ]; do
    while [ "$i" -lt "$n" ]; do
      local cap; cap=$(allowed_now)
      local cur; cur=$(running_jobs)
      if [ "$cap" -ge 1 ] && [ "$cur" -lt "$cap" ]; then
        mkdir -p "$(dirname "${LOGS[$i]}")"
        echo "[$(date +%T)] launch [$((i+1))/$n] gpu_free=$(free_mib)MiB ram_avail=$(ram_avail_mib)MiB cap=$cap cur=$cur -> ${LOGS[$i]}"
        bash -c "${CMDS[$i]}" >"${LOGS[$i]}" 2>&1 &
        i=$((i+1)); sleep 8   # stagger so two launches don't both grab memory at once
      else
        break
      fi
    done
    sleep "$POLL"
  done
  echo "[$(date +%T)] === STAGE DONE: $name ==="
}
