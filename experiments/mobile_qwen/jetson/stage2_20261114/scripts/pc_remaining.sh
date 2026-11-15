#!/usr/bin/env bash
set -euo pipefail
S=/home/yd/Documents/VLA/experiments/mobile_qwen/jetson/stage2_20261114
P=/home/yd/Documents/VLA/.venvs/qwen-scene/bin/python
ROOT=/home/yd/Documents/VLA
for size in 2b 4b; do
 if [[ "$size" == 4b ]]; then
  model4b=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
  systemd-run --user --wait --pipe --collect --unit=ali-4b-nf4-export-20261114 \
   --property=MemoryHigh=16G --property=MemoryMax=18G --property=OOMPolicy=stop \
   --setenv=TORCHINDUCTOR_COMPILE_THREADS=1 --property=WorkingDirectory="$ROOT" \
   "$P" "$S/scripts/quantize.py" --model "$model4b" --out "$S/nf4_4b"
 fi
 adapter="$S/lora_2b/best_adapter"
 [[ "$size" != 4b ]] || adapter="$ROOT/experiments/mobile_qwen/affordance/lora_v1/best_adapter"
 for resolution in 512; do
  tag="pc_${size}_lora_nf4_${resolution}"
  systemd-run --user --wait --pipe --collect --unit="ali-${tag//_/-}-20261114" \
   --property=MemoryHigh=16G --property=MemoryMax=18G --property=OOMPolicy=stop \
   --setenv=TORCHINDUCTOR_COMPILE_THREADS=1 --property=WorkingDirectory="$ROOT" \
   "$P" "$S/scripts/evaluate_once.py" --data "$S/test350" --out "$S/evaluations/$tag.json" \
   --model "$S/nf4_$size" --precision nf4 --adapter "$adapter" --resolution "$resolution"
 done
done
printf '%s\n' PC_REMAINING_COMPLETE
