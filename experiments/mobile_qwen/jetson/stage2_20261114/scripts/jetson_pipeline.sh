#!/usr/bin/env bash
set -euo pipefail
S=/home/yd/Documents/VLA/experiments/mobile_qwen/jetson/stage2_20261114
R=/home/csip/ali-jetson/project/experiments/mobile_qwen/jetson/stage2_20261114
host=${JETSON_HOST:-csip@192.168.31.168}
mkdir -p "$S/evaluations"
# One model at a time on Jetson; PC and Jetson own distinct GPUs.
for size in 2b 4b; do
 while [[ ! -f "$S/nf4_$size/EXPORT.json" ]]; do
  if ! systemctl --user is-active --quiet ali-pc-nf4-driver-20261114; then echo "PC export absent: $size" >&2;exit 1;fi
  sleep 5
 done
 rsync -a -e 'sshpass -e ssh' "$S/scripts/" "$host:$R/scripts/"
 bash "$S/scripts/sync_model.sh" "$size"
 set +e
 sshpass -e ssh "$host" "bash '$R/scripts/jetson_evaluate.sh' '$size'" > "$S/jetson_${size}_driver.log" 2>&1
 status=$?
 set -e
 rsync -a -e 'sshpass -e ssh' "$host:$R/evaluations/" "$S/evaluations/"
 printf '%s\n' "$status" > "$S/jetson_${size}_exit_code.txt"
 if (( status != 0 )); then
  echo "Jetson $size did not complete, preserving logs and stopping this model" >&2
  [[ "$size" != 2b ]] || exit "$status"
 fi
done
printf '%s\n' JETSON_PIPELINE_COMPLETE
