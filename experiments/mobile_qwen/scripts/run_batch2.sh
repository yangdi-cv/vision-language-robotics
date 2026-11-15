#!/usr/bin/env bash
# Online milestone monitor on fresh initial states 3-7 (not used for prompt design).
set -euo pipefail
S="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
req="把桌上的碗收进柜子最下面那层抽屉，然后关好抽屉"
for i in 3 4 5 6 7; do
  "$S/launch.sh" "ms_zh_i0$i" "$i" "10$i" "$req" --monitor milestone --check-every 10
done
echo BATCH_DONE
