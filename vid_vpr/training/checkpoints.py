import re
import shutil
from collections import OrderedDict
from pathlib import Path

import torch


def extract_state_dict(checkpoint, prefer_student=False):
    if not isinstance(checkpoint, dict):
        return checkpoint
    if prefer_student and checkpoint.get("student_state_dict") is not None:
        return checkpoint["student_state_dict"]
    if checkpoint.get("model_state_dict") is not None:
        return checkpoint["model_state_dict"]
    if checkpoint.get("state_dict") is not None:
        return checkpoint["state_dict"]
    return checkpoint


def strip_distributed_prefix(state_dict):
    return OrderedDict(
        (key.removeprefix("module."), value)
        for key, value in state_dict.items()
    )


def _cross_attention_indices(state_dict):
    pattern = re.compile(r"backbone\.vlm_crossattns\.(\d+)\.")
    return sorted(
        {
            int(match.group(1))
            for key in state_dict
            if (match := pattern.search(key))
        }
    )


def remap_legacy_cross_attention(state_dict, target_state_dict):
    source_indices = _cross_attention_indices(state_dict)
    target_indices = _cross_attention_indices(target_state_dict)
    if source_indices != list(range(0, 16, 2)) or target_indices != list(range(8, 24, 2)):
        return state_dict

    pattern = re.compile(r"(backbone\.vlm_crossattns\.)(\d+)(\..+)")
    remapped = OrderedDict()
    for key, value in state_dict.items():
        match = pattern.match(key)
        if match:
            candidate = f"{match.group(1)}{int(match.group(2)) + 8}{match.group(3)}"
            if candidate in target_state_dict and target_state_dict[candidate].shape == value.shape:
                key = candidate
        remapped[key] = value
    return remapped


def load_model_checkpoint(model, checkpoint_path, prefer_student=False, strict=True):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint, prefer_student=prefer_student)
    state_dict = strip_distributed_prefix(state_dict)
    state_dict = remap_legacy_cross_attention(state_dict, model.state_dict())
    prepare_state_dict = getattr(model, "prepare_checkpoint_state_dict", None)
    if prepare_state_dict is not None:
        state_dict = prepare_state_dict(state_dict)
    result = model.load_state_dict(state_dict, strict=strict)
    return checkpoint, result


def save_training_checkpoint(
    run_dir,
    payload,
    is_best,
    is_best_r1,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last_model.pth"
    torch.save(payload, last_path)
    if is_best:
        shutil.copyfile(last_path, run_dir / "best_model.pth")
    if is_best_r1:
        shutil.copyfile(last_path, run_dir / "best_r1_model.pth")
