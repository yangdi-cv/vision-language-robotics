#!/usr/bin/env bash
set -euo pipefail
Q=/home/csip/ali-jetson/project/experiments/mobile_qwen
S=$Q/jetson/stage2_20261114
P=/home/csip/ali-jetson/deployment_20261114/jetson_python.sh
size=${1:?2b or 4b}
mkdir -p "$S/evaluations"
for resolution in 512; do
 tag="jetson_${size}_lora_nf4_${resolution}_noswap"
 systemd-run --user --wait --pipe --collect --unit="ali-${tag//_/-}-20261114" \
   --setenv=JETSON_SWAP_GATE=1 --property=MemoryHigh=5G --property=MemoryMax=6G --property=OOMPolicy=stop \
   "$P" "$S/scripts/evaluate_once.py" --data "$S/test350" --out "$S/evaluations/$tag.json" \
   --model "$S/nf4_$size" --precision nf4 --adapter "$S/adapter_$size" --resolution "$resolution" \
   > "$S/evaluations/$tag.log" 2>&1
done
printf '%s\n' "JETSON_${size}_COMPLETE"
