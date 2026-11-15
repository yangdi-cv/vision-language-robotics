#!/usr/bin/env bash
# Sequential mission set; each case runs in its own memory-limited unit via launch.sh.
set -euo pipefail
Q="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
run() { "$Q/scripts/launch.sh" "$@" | tail -1; }
run zh_bowl_i01 1 93 "把桌上的碗收进柜子最下面那层抽屉，然后关好抽屉"
run zh_bowl_i02 2 94 "把桌上的碗收进柜子最下面那层抽屉，然后关好抽屉"
run en_para_i00 0 95 "Could you tidy the bowl away into the lowest drawer and shut it?"
run zh_black_i00 0 96 "把黑色的碗放进最下面的抽屉并关上"
run refuse_microwave_i00 0 97 "把杯子放进微波炉，然后关上微波炉门"
echo BATCH_DONE
