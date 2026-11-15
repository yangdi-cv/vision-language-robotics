#!/usr/bin/env bash
# Revised milestones (own-words bowl description, drawer confirm 3) on fresh initial states 8-12.
set -euo pipefail
S="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
req="把桌上的碗收进柜子最下面那层抽屉，然后关好抽屉"
for i in 8 9 10 11 12; do
  "$S/launch.sh" "ms2_zh_i$(printf %02d "$i")" "$i" "$((100 + i))" "$req" --monitor milestone --check-every 10
done
echo BATCH_DONE
