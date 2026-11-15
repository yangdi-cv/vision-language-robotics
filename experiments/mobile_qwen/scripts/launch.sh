#!/usr/bin/env bash
# Run one case inside a memory-limited systemd user unit, outside the editor's cgroup.
# Usage: launch.sh <tag> <initial> <domain> "<request>" [extra run_case_qwen.py args]
set -euo pipefail
Q="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tag="$1"; initial="$2"; domain="$3"; request="$4"; shift 4
mkdir -p "$Q/logs"
if [[ -e "$Q/results/$tag" || -e "$Q/logs/${tag}_journal" ]]; then
  echo "Refusing existing tag: $tag" >&2
  exit 2
fi
unit="ali-mobile-qwen-${tag//_/-}"
status=0
systemd-run --user --wait --collect --unit="$unit" \
  --property=MemoryHigh=16G --property=MemoryMax=18G --property=OOMPolicy=stop \
  --property=WorkingDirectory="$Q" \
  /usr/bin/python3 "$Q/scripts/run_case_qwen.py" --tag "$tag" --initial "$initial" --domain "$domain" --request "$request" "$@" \
  > "$Q/logs/${tag}_launch.log" 2>&1 || status=$?
tail -3 "$Q/logs/${tag}_launch.log"
cat "$Q/results/$tag/summary.json" 2>/dev/null || echo "no summary"

exit "$status"
