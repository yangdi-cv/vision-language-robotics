#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/yd/Documents/VLA
S=$ROOT/experiments/mobile_qwen/jetson/stage2_20261114
P=$ROOT/.venvs/qwen-scene/bin/python
cd "$ROOT"
while [[ ! -e "$S/MODEL_2B.json" ]]; do
  if ! systemctl --user is-active --quiet ali-2b-download-20261114; then echo 'Download did not finish' >&2;exit 1;fi
  sleep 3
done
"$P" "$S/scripts/train_2b.py"
mkdir -p "$S/evaluations"
model2b=$("$P" -c "import json;print(json.load(open('$S/MODEL_2B.json'))['snapshot'])")
"$P" "$S/scripts/evaluate_once.py" --data "$S/test350" --out "$S/evaluations/pc_2b_base_bf16_512.json" --model "$model2b" --precision bf16
"$P" "$S/scripts/evaluate_once.py" --data "$S/test350" --out "$S/evaluations/pc_2b_lora_bf16_512.json" --model "$model2b" --precision bf16 --adapter "$S/lora_2b/best_adapter"
"$P" "$S/scripts/quantize.py" --model "$model2b" --out "$S/nf4_2b"
for resolution in 512 384 256; do
 "$P" "$S/scripts/evaluate_once.py" --data "$S/test350" --out "$S/evaluations/pc_2b_lora_nf4_${resolution}.json" --model "$S/nf4_2b" --precision nf4 --adapter "$S/lora_2b/best_adapter" --resolution "$resolution"
done
model4b=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
"$P" "$S/scripts/quantize.py" --model "$model4b" --out "$S/nf4_4b"
for resolution in 512 384 256; do
 "$P" "$S/scripts/evaluate_once.py" --data "$S/test350" --out "$S/evaluations/pc_4b_lora_nf4_${resolution}.json" --model "$S/nf4_4b" --precision nf4 --adapter "$ROOT/experiments/mobile_qwen/affordance/lora_v1/best_adapter" --resolution "$resolution"
done
printf '%s\n' 'PC_PIPELINE_COMPLETE'
