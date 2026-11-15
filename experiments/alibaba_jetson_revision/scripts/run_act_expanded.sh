#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
for seed in 20261024 20261025; do
 bash experiments/desktop_pickplace/scripts/act-python experiments/alibaba_jetson_revision/scripts/train_act_expanded.py --out experiments/alibaba_jetson_revision/runs/act_seed_${seed} --seed "$seed" --steps 10000 --interval 2000 --validation-seeds 1500 1501 1502 1503 1504 1505 --test-seeds 1600 1601 1602 1603 1604 1605 1606 1607
done
