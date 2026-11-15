#!/usr/bin/env bash
set -euo pipefail
INSTANCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLA_EXPERIMENT_ROOT="$(cd -- "$INSTANCE_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$VLA_EXPERIMENT_ROOT/../.." && pwd)"
DEMO_UNIT="ali-vla-demo-$(date +%s)"
systemd-run --user --unit="$DEMO_UNIT" --property=MemoryHigh=14G --property=MemoryMax=16G --property=OOMPolicy=stop --property=WorkingDirectory="$WORKSPACE_ROOT" /bin/bash "$INSTANCE_ROOT/python_model.sh" "$INSTANCE_ROOT/scripts/run.py" "$@"
echo "日志：journalctl --user -fu $DEMO_UNIT"
