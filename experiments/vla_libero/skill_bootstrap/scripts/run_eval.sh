#!/usr/bin/env bash
set -euo pipefail
V=/home/yd/Documents/VLA/experiments/vla_libero
for i in {1..120}; do
  if python3 -c 'import json; assert json.load(open("/home/yd/Documents/VLA/experiments/vla_libero/skill_bootstrap/lora_pilot/TRAIN_STATUS.json"))["state"]=="complete"' 2>/dev/null; then break; fi
  sleep 2
done
python3 -c 'import json; assert json.load(open("/home/yd/Documents/VLA/experiments/vla_libero/skill_bootstrap/lora_pilot/TRAIN_STATUS.json"))["state"]=="complete"'
for variant in native_reference zero base lora; do
  bash "$V/skill_bootstrap/scripts/python_model.sh" "$V/skill_bootstrap/scripts/evaluate_policy.py" --variant "$variant" > "$V/skill_bootstrap/logs/eval_${variant}.log" 2>&1
done
