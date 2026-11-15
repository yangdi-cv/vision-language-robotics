#!/usr/bin/env bash
# Optional single validation query after the mandatory 2B test has finished.
set -euo pipefail
S=/home/yd/Documents/VLA/experiments/mobile_qwen/jetson/stage2_20261114
R=/home/csip/ali-jetson/project/experiments/mobile_qwen/jetson/stage2_20261114
host=${JETSON_HOST:-csip@192.168.55.1}
root_command() { printf '%s\n' "$SSHPASS" | sshpass -e ssh "$host" "sudo -S bash -c '$1'"; }
restore_swap() { root_command 'swapon -a; for d in /dev/zram[0-5]; do swapon --priority 5 "$d" 2>/dev/null || true; done' || true; }
trap restore_swap EXIT
bash "$S/scripts/sync_model.sh" 4b
rsync -a -e 'sshpass -e ssh' "$S/scripts/" "$host:$R/scripts/"
sshpass -e scp "$S/probe_val13.png" "$S/probe_4b_request.json" "$host:$R/"
sshpass -e ssh "$host" "systemd-run --user --wait --pipe --collect --unit=ali-4b-qualification-20261114 --property=MemoryHigh=5G --property=MemoryMax=6G --property=OOMPolicy=stop /home/csip/ali-jetson/deployment_20261114/jetson_python.sh '$R/scripts/probe_4b.py'" > "$S/probe_4b.log" 2>&1 &
job=$!
for ((i=0;i<180;i++)); do
 if ! kill -0 "$job" 2>/dev/null; then wait "$job";exit 1;fi
 if sshpass -e ssh "$host" test -f "$R/probe_4b.load.json"; then
  root_command 'swapoff -a'
  sshpass -e ssh "$host" "test \$(wc -l < /proc/swaps) -eq 1 && touch '$R/probe_4b.swap_ready'"
  wait "$job"
  sshpass -e scp "$host:$R/probe_4b_result.json" "$S/"
  exit 0
 fi
 sleep 2
done
echo '4B qualification timed out' >&2; exit 1
