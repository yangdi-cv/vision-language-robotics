#!/usr/bin/env bash
set -euo pipefail
P=/home/csip/miniforge3/envs/ali-jetson/bin/python
OUT=/home/csip/ali-jetson/deployment_20261114
"$P" -m pip install --report "$OUT/torch_installed.json" --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch==2.8.0 torchvision==0.23.0 bitsandbytes==0.48.0.dev0+ff389db
cat > "$OUT/constraints.txt" <<'CONSTRAINTS'
torch==2.8.0
torchvision==0.23.0
bitsandbytes==0.48.0.dev0+ff389db
numpy==1.26.4
CONSTRAINTS
"$P" -m pip install -c "$OUT/constraints.txt" --report "$OUT/hf_installed.json" transformers==4.57.6 peft==0.18.1 accelerate==1.12.0
"$P" -m pip check
"$P" -m pip freeze > "$OUT/pip_after.txt"
