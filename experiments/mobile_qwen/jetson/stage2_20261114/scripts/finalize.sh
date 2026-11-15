#!/usr/bin/env bash
# Wait for the mandatory single-pass comparison, then qualify 4B and write report.
set -euo pipefail
S=/home/yd/Documents/VLA/experiments/mobile_qwen/jetson/stage2_20261114
P=/home/yd/Documents/VLA/.venvs/qwen-scene/bin/python
host=${JETSON_HOST:-csip@192.168.55.1}
R=/home/csip/ali-jetson/project/experiments/mobile_qwen/jetson/stage2_20261114
while systemctl --user is-active --quiet ali-jetson-2b-noswap-driver-20261114; do sleep 5;done
rsync -a -e 'sshpass -e ssh' "$host:$R/evaluations/" "$S/evaluations/"
rsync -a -e 'sshpass -e ssh' "$host:$R/diagnostics/" "$S/diagnostics/"
[[ -f "$S/evaluations/jetson_2b_lora_nf4_512_noswap.resources.json" ]]
while systemctl --user is-active --quiet ali-pc-nf4-driver-20261114;do sleep 5;done
set +e
bash "$S/scripts/probe_4b_driver.sh"
probe_status=$?
set -e
printf '%s\n' "$probe_status" > "$S/probe_4b_exit_code.txt"
"$P" "$S/scripts/summarize.py" --out "$S/SUMMARY.json"
"$P" "$S/scripts/write_report.py"
sshpass -e ssh "$host" 'swapon --show; /usr/sbin/nvpmodel -q; systemctl get-default' > "$S/jetson_final_state.txt"
printf '%s\n' STAGE2_CORE_COMPLETE
