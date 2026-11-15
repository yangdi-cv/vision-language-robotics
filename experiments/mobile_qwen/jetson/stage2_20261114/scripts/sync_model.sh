#!/usr/bin/env bash
# SSHPASS is supplied by the caller; credentials are never written into this file.
set -euo pipefail
size=${1:?2b or 4b}
S=/home/yd/Documents/VLA/experiments/mobile_qwen/jetson/stage2_20261114
R=/home/csip/ali-jetson/project/experiments/mobile_qwen/jetson/stage2_20261114
host=${JETSON_HOST:-csip@192.168.31.168}
adapter="$S/lora_2b/best_adapter/"
[[ "$size" != 4b ]] || adapter=/home/yd/Documents/VLA/experiments/mobile_qwen/affordance/lora_v1/best_adapter/
sshpass -e ssh "$host" mkdir -p "$R/nf4_$size" "$R/adapter_$size"
rsync -a -e 'sshpass -e ssh' "$S/nf4_$size/" "$host:$R/nf4_$size/"
rsync -a -e 'sshpass -e ssh' "$adapter" "$host:$R/adapter_$size/"
sshpass -e ssh "$host" /home/csip/miniforge3/envs/ali-jetson/bin/python "$R/scripts/verify_manifest.py" "$R/nf4_$size" --export
