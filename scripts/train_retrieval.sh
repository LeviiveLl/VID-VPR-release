#!/usr/bin/env bash
set -euo pipefail
GPUS="${GPUS:-4}"
torchrun --standalone --nproc_per_node="$GPUS" -m vid_vpr.train_retrieval --config configs/retrieval.yaml "$@"
