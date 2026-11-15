#!/usr/bin/env bash
# Scoped C++ runtime fix for Jetson torch + conda-forge scipy; no system library edits.
set -euo pipefail
ALI_PREFIX=/home/csip/miniforge3/envs/ali-jetson
export LD_PRELOAD="$ALI_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
exec "$ALI_PREFIX/bin/python" "$@"
