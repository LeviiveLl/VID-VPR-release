import argparse
import json
import logging

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from vid_vpr.config import load_config, project_path, require_sections
from vid_vpr.data.factory import build_train_loader
from vid_vpr.evaluation.retrieval import evaluate_model
from vid_vpr.models.factory import build_vid_vpr, load_teacher
from vid_vpr.training.checkpoints import load_model_checkpoint
from vid_vpr.training.losses import build_metric_objective, metric_loss
from vid_vpr.training.runtime import close_runtime, setup_runtime, unwrap_model


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the final VID-VPR descriptor")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def normalize_patch_map(values):
    values = values.float()
    centered = values - values.mean(dim=1, keepdim=True)
    return centered * torch.rsqrt(centered.square().mean(dim=1, keepdim=True) + 1e-6)


def teacher_patch_utility(correct, shuffled, layers, patch_count):
    utilities = []
    for index in layers:
        left = correct[index][:, -patch_count:].detach().float()
        right = shuffled[index][:, -patch_count:].detach().float()
        difference = (left - right).square().mean(dim=-1).add(1e-8).sqrt()
        reference = 0.5 * (
            left.square().mean(dim=-1).add(1e-8).sqrt()
            + right.square().mean(dim=-1).add(1e-8).sqrt()
        )
        utilities.append((difference / reference.clamp_min(1e-6)).log1p())
    return normalize_patch_map(torch.stack(utilities, dim=1).mean(dim=1)).clamp(-3, 3)


def gather_batch(descriptors, labels, runtime):
    if not runtime.distributed:
        return descriptors, labels
    descriptors = torch.cat(dist_nn.all_gather(descriptors), dim=0)
    gathered_labels = [torch.empty_like(labels) for _ in range(runtime.world_size)]
    dist.all_gather(gathered_labels, labels)
    return descriptors, torch.cat(gathered_labels)


def configure_trainable(model, config):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.aggregator.parameters():
        parameter.requires_grad = True
    for parameter in model.student.router.parameters():
        parameter.requires_grad = True
    for parameter in model.student.backbone.vlm_projector.parameters():
        parameter.requires_grad = True
    for index in config.get("train_cross_attention_layers", (20, 22)):
        for parameter in model.student.backbone.vlm_crossattns[index].parameters():
            parameter.requires_grad = True
    last_n = int(config.get("unfreeze_last_n_blocks", 0))
    if last_n:
        for block in model.student.backbone.blocks[-last_n:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in model.student.backbone.norm.parameters():
            parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def main():
    args = parse_args()
    config = load_config(args.config, args.overrides)
    require_sections(config, "experiment", "model", "data", "training", "evaluation")
    runtime = setup_runtime(config)
    try:
        if runtime.is_main:
            (runtime.run_dir / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False)
            )
        model = build_vid_vpr(config["model"])
        initialization = config["model"].get("initialization_path")
        if initialization:
            payload = torch.load(project_path(initialization), map_location="cpu", weights_only=False)
            if "model_state_dict" in payload and any(
                key.startswith("student.") for key in payload["model_state_dict"]
            ):
                model.load_state_dict(payload["model_state_dict"], strict=True)
            else:
                load_model_checkpoint(
                    model.student, project_path(initialization), prefer_student=True, strict=True
                )
        parameters = configure_trainable(model, config["training"])
        model.to(runtime.device)

        utility_weight = float(config["training"].get("teacher_utility_weight", 0.0))
        teacher = None
        if utility_weight > 0:
            teacher = load_teacher(config["model"], config["model"]["teacher_path"])
            teacher.to(runtime.device).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad = False

        loader = build_train_loader(config, runtime)
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config["training"].get("learning_rate", 1e-5)),
            weight_decay=float(config["training"].get("weight_decay", 1e-8)),
        )
        loss_fn, miner = build_metric_objective()
        use_amp = runtime.device.type == "cuda"
        if runtime.distributed:
            model = DistributedDataParallel(
                model, device_ids=[runtime.local_rank], find_unused_parameters=False
            )

        for epoch in range(int(config["training"].get("epochs", 4))):
            if runtime.distributed and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            model.train()
            progress = tqdm(loader, disable=not runtime.is_main, desc=f"Epoch {epoch + 1}")
            running = 0.0
            for images, labels, hidden, masks in progress:
                images = images.to(runtime.device, non_blocking=True)
                labels = labels.to(runtime.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(runtime.device.type, torch.float16, enabled=use_amp):
                    output = model(images, return_diagnostics=utility_weight > 0)
                    descriptors = output["descriptor"] if isinstance(output, dict) else output
                    metric_descriptors, metric_labels = gather_batch(
                        descriptors, labels, runtime
                    )
                    loss = metric_loss(metric_descriptors, metric_labels, loss_fn, miner)
                    if utility_weight > 0:
                        hidden = hidden.to(runtime.device, non_blocking=True)
                        masks = masks.to(runtime.device, non_blocking=True)
                        permutation = torch.roll(torch.arange(images.shape[0], device=images.device), 1)
                        with torch.no_grad():
                            correct = teacher(
                                images, hidden, masks, return_xattn_deltas=True
                            )["xattn_deltas"]
                            shuffled = teacher(
                                images, hidden[permutation], masks[permutation], return_xattn_deltas=True
                            )["xattn_deltas"]
                            target = teacher_patch_utility(
                                correct, shuffled, unwrap_model(model).response_layers,
                                output["response_logits"].shape[1],
                            )
                        prediction = normalize_patch_map(output["response_logits"]).clamp(-3, 3)
                        loss = loss + utility_weight * 0.5 * (prediction - target).square().mean()
                loss.backward()
                optimizer.step()
                running += float(loss.detach())
                progress.set_postfix(loss=f"{running / max(progress.n, 1):.4f}")

            metrics = None
            if runtime.is_main and config["evaluation"].get("enabled", True):
                metrics, _ = evaluate_model(
                    unwrap_model(model), config, runtime.device, use_vlm=False
                )
            if runtime.distributed:
                dist.barrier()
            if runtime.is_main:
                payload = {
                    "format_version": 1,
                    "architecture": "VIDVPR",
                    "epoch": epoch,
                    "model_config": config["model"],
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "metrics": metrics,
                }
                torch.save(payload, runtime.run_dir / "last_model.pth")
                (runtime.run_dir / "metrics.json").write_text(
                    json.dumps(metrics or {}, indent=2) + "\n"
                )
                logging.info("Epoch %d: %s", epoch + 1, metrics)
    finally:
        close_runtime(runtime)


if __name__ == "__main__":
    main()
