# VID-VPR

Official implementation of **VLM Injection Distillation for Image-Only Visual Place Recognition**.
A VLM-conditioned teacher exposes layer-wise injection residuals. The student learns a depth-conditioned synthetic-prior memory and router, then performs image-only nearest-neighbor retrieval at test time.

The release contains no datasets, VLM weights, VLM cache, or DINOv2 pretraining weights.

## Installation

```bash
conda env create -f environment.yml
conda activate vid-vpr
pip install -e .
```

The experiments used Python 3.10, PyTorch 2.10 and CUDA 12.x. xFormers is optional.

## Data layout

Create links or directories with the following layout:

```text
data/
  gsv_cities/          # Dataframes/ and Images/ from GSV-Cities
  benchmarks/          # evaluation datasets
  vlm/
    gsv/                # cached training VLM states
    pitts30k_test/      # cached states needed only for teacher evaluation
pretrained/
  dinov2_vitl14.pth
```

Cached files are aligned with image paths and contain `hidden_states` and `attention_mask` tensors. The fixed VPR instruction from the paper is the default prompt in `scripts/extract_vlm_states.py`.

## Checkpoints

| File | Role |
|---|---|---:|---|
| `checkpoints/teacher.pth` | VLM-conditioned teacher | 
| `checkpoints/student.pth` | final image-only VID-VPR |

Both files contain only `model_config` and a complete `model_state_dict`; no intermediate checkpoint is required. Note that the released final student has an 8448-D descriptor (`64 x 128 + 256`).

## Evaluation

```bash
python -m vid_vpr.evaluate \
  --config configs/eval.yaml \
  --model student
```

Teacher evaluation additionally reads cached VLM states:

```bash
python -m vid_vpr.evaluate \
  --config configs/eval.yaml \
  --model teacher
```

Configuration values can be overridden from the command line:

```bash
python -m vid_vpr.evaluate --config configs/eval.yaml --model student \
  evaluation.dataset_name=Tokyo247 data.benchmark_root=/path/to/benchmarks
```

## Training

```bash
# 1. VLM-conditioned teacher
bash scripts/train_teacher.sh

# 2. Image-only student: router, then intervention path
bash scripts/train_distill_stage1.sh
bash scripts/train_distill_stage2.sh

# 3. Final retrieval head and teacher-utility adaptation
bash scripts/train_retrieval.sh
```

The launchers use `GPUS` to control process count, for example `GPUS=4 bash scripts/train_distill_stage2.sh`. Training requires GSV-Cities and cached VLM states. Deployment and student evaluation do not use a VLM or cache.

## Minimal API

```python
import torch
from vid_vpr import load_vid_vpr

model = load_vid_vpr("checkpoints/student.pth").cuda().eval()
images = torch.randn(2, 3, 322, 322, device="cuda")
with torch.inference_mode():
    descriptors = model(images)  # [2, 8448], L2 normalized
```

## Licenses

Project code is released under MIT. The copied DINOv2 backbone and benchmark utilities retain their upstream notices in `licenses/`.
