#!/usr/bin/env bash
set -euo pipefail
GPUS="${GPUS:-8}"
torchrun --standalone --nproc_per_node="$GPUS" -m vid_vpr.train_teacher --config configs/teacher.yaml "$@"
