#!/bin/bash
# Pings when finetune produces an evaluable artifact (merged ckpt OR step-2000 adapter), dies, or heartbeat.
set -uo pipefail; cd "$(dirname "$0")"
t0=$(date +%s); HB=${HB:-1800}; PID=${PID:-3095834}
MERGED=results/lora_boost/smolvla_lora_boost/model.safetensors
ADAPTER=results/lora_boost/adapter_s2000/adapter_model.safetensors
while true; do
  [ -f "$MERGED" ] && { echo "BOOST DONE — merged ckpt saved $(date +%T)"; break; }
  [ -f "$ADAPTER" ] && { echo "BOOST adapter_s2000 saved (mergeable) $(date +%T)"; break; }
  if ! ps -p "$PID" >/dev/null 2>&1; then echo "BOOST proc gone $(date +%T)"; ls results/lora_boost/adapter_s* 2>/dev/null; tail -3 results/lora_boost/finetune2.log; break; fi
  if [ $(( $(date +%s)-t0 )) -ge $HB ]; then echo "BOOST hb: $(grep -oE 'step= *[0-9]+/2000' results/lora_boost/finetune2.log|tail -1) $(date +%T)"; break; fi
  sleep 60
done
