import argparse
import logging

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from vid_vpr.config import load_config, project_path, require_sections
from vid_vpr.data.factory import build_train_loader
from vid_vpr.evaluation.retrieval import evaluate_model
from vid_vpr.models.factory import build_teacher
from vid_vpr.training.checkpoints import load_model_checkpoint, save_training_checkpoint
from vid_vpr.training.losses import build_metric_objective, metric_loss
from vid_vpr.training.optim import build_optimizer, build_scheduler
from vid_vpr.training.runtime import close_runtime, setup_runtime, unwrap_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the VLM-conditioned VPR teacher")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="Configuration overrides in key=value form")
    return parser.parse_args()


def _evaluate(runtime, model, config):
    if not config["evaluation"].get("enabled", True):
        return {1: 0.0, 5: 0.0}
    result = None
    if runtime.is_main:
        result, _ = evaluate_model(
            unwrap_model(model),
            config,
            runtime.device,
            use_vlm=True,
        )
    if runtime.distributed:
        holder = [result]
        dist.broadcast_object_list(holder, src=0)
        result = holder[0]
    return result


def main():
    cli = parse_args()
    config = load_config(cli.config, cli.overrides)
    require_sections(config, "experiment", "model", "data", "training", "evaluation")
    runtime = setup_runtime(config)
    try:
        if runtime.is_main:
            with (runtime.run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
        model = build_teacher(config["model"], load_foundation=True)
        model.set_trainable_layers(config["training"].get("unfreeze_last_n_layers", 0))
        model.to(runtime.device)
        loader = build_train_loader(config, runtime)
        optimizer = build_optimizer(model.parameters(), config["training"])
        scheduler = build_scheduler(optimizer, config["training"], len(loader))
        loss_fn, miner = build_metric_objective()

        start_epoch = 0
        best_score = 0.0
        best_r1 = 0.0
        not_improved = 0
        resume_path = config["training"].get("resume_path")
        if resume_path:
            checkpoint, _ = load_model_checkpoint(
                model,
                project_path(resume_path),
                strict=True,
            )
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch_num"] + 1
            best_score = float(checkpoint.get("best_r5", 0.0))
            best_r1 = float(checkpoint.get("best_r1", 0.0))
            not_improved = int(checkpoint.get("not_improved_num", 0))

        if runtime.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[runtime.local_rank],
                output_device=runtime.local_rank,
            )

        use_amp = runtime.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        training_config = config["training"]
        max_steps = training_config.get("max_steps")
        for epoch in range(start_epoch, training_config["epochs"]):
            if runtime.distributed:
                loader.sampler.set_epoch(epoch)
            model.train()
            losses = []
            progress = tqdm(loader, disable=not runtime.is_main, ncols=110)
            for step, batch in enumerate(progress):
                images, place_ids, hidden, masks = batch
                batch_size, views, channels, height, width = images.shape
                images = images.view(
                    batch_size * views,
                    channels,
                    height,
                    width,
                ).to(runtime.device, non_blocking=True)
                labels = place_ids.view(-1).to(runtime.device, non_blocking=True)
                hidden = hidden.to(runtime.device, non_blocking=True)
                masks = masks.to(runtime.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=runtime.device.type,
                    enabled=use_amp,
                ):
                    descriptors = model(images, hidden, masks)
                    loss, _ = metric_loss(loss_fn, miner, descriptors, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                losses.append(loss.detach().float().item())
                if runtime.is_main:
                    progress.set_postfix(
                        loss=f"{losses[-1]:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )
                if max_steps is not None and step + 1 >= max_steps:
                    break

            if runtime.is_main:
                logging.info(
                    "Epoch %02d average loss %.6f",
                    epoch,
                    sum(losses) / max(len(losses), 1),
                )
            recalls = _evaluate(runtime, model, config)
            current_r1 = recalls.get(1, 0.0)
            current_score = current_r1 + recalls.get(5, 0.0)
            is_best = current_score > best_score
            is_best_r1 = current_r1 > best_r1
            if is_best:
                best_score = current_score
                not_improved = 0
            else:
                not_improved += 1
            best_r1 = max(best_r1, current_r1)

            if runtime.is_main:
                payload = {
                    "epoch_num": epoch,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "recalls": recalls,
                    "best_r5": best_score,
                    "best_r1": best_r1,
                    "not_improved_num": not_improved,
                    "config": config,
                }
                save_training_checkpoint(
                    runtime.run_dir,
                    payload,
                    is_best=is_best,
                    is_best_r1=is_best_r1,
                )
            stop = not_improved >= training_config["patience"]
            if runtime.distributed:
                holder = [stop]
                dist.broadcast_object_list(holder, src=0)
                stop = holder[0]
            if stop:
                break
    finally:
        close_runtime(runtime)


if __name__ == "__main__":
    main()
