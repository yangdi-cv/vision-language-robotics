#!/usr/bin/env bash
set -euo pipefail
V=/home/yd/Documents/VLA/experiments/vla_libero
unset PYTHONHOME PYTHONPATH
export PYTHONPATH="$V/third_party/libero"
export OPENPI_DATA_HOME="$V/cache" LIBERO_CONFIG_PATH="$V/configs/libero"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled
exec /home/yd/Documents/VLA/.venvs/openpi-libero/bin/python "$V/skill_bootstrap/scripts/train_lora.py"
