#!/usr/bin/env bash
# Keep swap as a load reserve, then disable it before the formal single-pass test.
# SSHPASS is provided by the caller, never stored in source files.
set -euo pipefail
host=${JETSON_HOST:-csip@192.168.55.1}
S=/home/yd/Documents/VLA/experiments/mobile_qwen/jetson/stage2_20261114
R=/home/csip/ali-jetson/project/experiments/mobile_qwen/jetson/stage2_20261114
size=${1:?2b or 4b}
tag="jetson_${size}_lora_nf4_512_noswap"
root_command() { printf '%s\n' "$SSHPASS" | sshpass -e ssh "$host" "sudo -S bash -c '$1'"; }
restore_swap() {
 root_command 'swapon -a; for d in /dev/zram[0-5]; do swapon --priority 5 "$d" 2>/dev/null || true; done' || true
}
trap restore_swap EXIT
sshpass -e ssh "$host" "bash '$R/scripts/jetson_evaluate.sh' '$size'" > "$S/jetson_${size}_noswap_driver.log" 2>&1 &
job=$!
for ((i=0;i<180;i++)); do
 if ! kill -0 "$job" 2>/dev/null; then wait "$job"; exit 1;fi
 if sshpass -e ssh "$host" test -f "$R/evaluations/$tag.load.json"; then
  root_command 'swapoff -a'
  sshpass -e ssh "$host" "test \$(wc -l < /proc/swaps) -eq 1 && touch '$R/evaluations/$tag.swap_ready'"
  wait "$job"
  rsync -a -e 'sshpass -e ssh' "$host:$R/evaluations/" "$S/evaluations/"
  exit 0
 fi
 sleep 2
done
echo 'Load gate timed out' >&2
exit 1
