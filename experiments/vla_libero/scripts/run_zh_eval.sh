#!/usr/bin/env bash
# Paired flat vs Qwen-hierarchical evaluation on Chinese Long-suite instructions.
set -uo pipefail
cd "$(dirname "$0")/.."
INITS="${INITS:-0 1 2}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
for t in $TASKS; do
  zh=$(python3 -c "import json;print(json.load(open('configs/libero_long_zh.json'))['$t'])")
  scripts/python_shared scripts/run_hierarchical.py --task-id "$t" --init $INITS \
    --planner qwen --monitor qwen --tag "${HIER_TAG:-zh_eval_hier}" --instruction "$zh" 2>&1 | grep '^{"status' | sed "s/^/hier t$t /"
  if [ -n "${TRANSLATE_TAG:-}" ]; then
    scripts/python_shared scripts/run_hierarchical.py --task-id "$t" --init $INITS \
      --planner translate --tag "$TRANSLATE_TAG" --instruction "$zh" 2>&1 | grep '^{"status' | sed "s/^/translate t$t /"
  fi
  [ -n "${SKIP_FLAT:-}" ] && continue
  scripts/python_shared scripts/run_hierarchical.py --task-id "$t" --init $INITS \
    --planner none --tag zh_eval_flat --instruction "$zh" 2>&1 | grep '^{"status' | sed "s/^/flat t$t /"
done
echo ALL_DONE
