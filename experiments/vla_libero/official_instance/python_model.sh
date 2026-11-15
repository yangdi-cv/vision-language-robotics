#!/usr/bin/env bash
set -euo pipefail
INSTANCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLA_EXPERIMENT_ROOT="$(cd -- "$INSTANCE_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$VLA_EXPERIMENT_ROOT/../.." && pwd)"
unset PYTHONHOME PYTHONPATH
export PYTHONPATH="$VLA_EXPERIMENT_ROOT/third_party/libero"
export OPENPI_DATA_HOME="$VLA_EXPERIMENT_ROOT/cache" LIBERO_CONFIG_PATH="$VLA_EXPERIMENT_ROOT/configs/libero"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
exec "$WORKSPACE_ROOT/.venvs/openpi-libero/bin/python" "$@"
