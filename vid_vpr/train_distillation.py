import argparse
import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from vid_vpr.config import load_config, project_path, require_sections
from vid_vpr.data.factory import build_train_loader
from vid_vpr.evaluation.retrieval import evaluate_model, extract_descriptors
from vid_vpr.models.factory import build_student, load_teacher
from vid_vpr.training.checkpoints import load_model_checkpoint, save_training_checkpoint
from vid_vpr.training.losses import (
    ALIGNMENT_TARGETS,
    build_metric_objective,
    descriptor_agreement,
    distillation_alignment_loss,
    metric_loss,
    pair_statistics,
)
from vid_vpr.training.optim import build_optimizer, build_scheduler
from vid_vpr.training.runtime import close_runtime, setup_runtime, unwrap_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the image-only VID-VPR student")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="Configuration overrides in key=value form")
    return parser.parse_args()


@torch.no_grad()
def initialize_memory_bank(config, student, loader, runtime):
    model_config = config["model"]
    if model_config.get("memory_init") != "vlm_sample":
        return
    target_tokens = max(
        model_config.get("memory_init_tokens", 4096),
        model_config["bank_size"],
    )
    if runtime.is_main:
        sampled = []
        count = 0
        for batch in loader:
            hidden, masks = batch[2], batch[3]
            valid = hidden[masks.to(dtype=torch.bool)]
            if valid.numel():
                sampled.append(valid)
                count += valid.shape[0]
            if count >= target_tokens:
                break
        if not sampled:
            raise RuntimeError("no valid VLM tokens were found for memory initialization")
        sampled = torch.cat(sampled, dim=0)[:target_tokens]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config["experiment"].get("seed", 42))
        indices = torch.randperm(sampled.shape[0], generator=generator)[
            : model_config["bank_size"]
        ]
        unwrap_model(student).router.initialize_memory_bank(sampled[indices])
        logging.info("Initialized synthetic-prior memory from %d VLM tokens", len(indices))
    if runtime.distributed:
        dist.broadcast(unwrap_model(student).router.memory_bank.data, src=0)


def _broadcast_result(runtime, result):
    if not runtime.distributed:
        return result
    holder = [result if runtime.is_main else None]
    dist.broadcast_object_list(holder, src=0)
    return holder[0]


def _expand_teacher_descriptor(descriptor, target_dim):
    source_dim = descriptor.shape[-1]
    if source_dim == target_dim:
        return descriptor
    if target_dim % source_dim:
        raise ValueError(
            f"student descriptor dim {target_dim} must be a multiple of "
            f"teacher descriptor dim {source_dim}"
        )
    return F.normalize(
        descriptor.repeat(1, target_dim // source_dim),
        p=2,
        dim=-1,
    )


def _benchmark_config(config, benchmark):
    benchmark_config = copy.deepcopy(config)
    benchmark_root = benchmark.get("benchmark_root")
    if benchmark_root is not None:
        benchmark_config["data"]["benchmark_root"] = benchmark_root
    for key in (
        "dataset_name",
        "split",
        "batch_size",
        "positive_distance",
        "recall_values",
        "test_method",
    ):
        if key in benchmark:
            benchmark_config["evaluation"][key] = benchmark[key]
    return benchmark_config


def _parse_msls_challenge_metrics(path):
    recalls = {}
    mean_average_precision = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, separator, raw_value = line.partition(":")
        if not separator:
            continue
        if key.startswith("all_recall@"):
            recalls[key.removeprefix("all_recall@")] = float(raw_value) * 100.0
        elif key.startswith("all_map@"):
            mean_average_precision[key.removeprefix("all_map@")] = (
                float(raw_value) * 100.0
            )
    if not recalls:
        raise RuntimeError(f"MSLS Challenge evaluator produced no recalls: {path}")
    return {
        "recall": recalls,
        "map": mean_average_precision,
    }


def _evaluate_msls_challenge(runtime, model, config, epoch):
    challenge_config = config["evaluation"].get("msls_challenge", {})
    if not challenge_config.get("enabled", False):
        return None

    from tools.generate_msls_challenge_predictions import (
        ImagePathsDataset,
        load_msls_dataset,
        run_official_evaluation,
        search_topk,
        write_predictions,
    )

    local_args = SimpleNamespace(
        msls_root=challenge_config["msls_root"],
        mapillary_sls_root=challenge_config["mapillary_sls_root"],
        cities=challenge_config.get("cities", "test"),
        task="im2im",
        subtask=challenge_config.get("subtask", "all"),
    )
    (
        database_paths,
        query_paths,
        database_keys,
        query_keys,
    ) = load_msls_dataset(local_args)
    combined_dataset = ImagePathsDataset(
        database_paths + query_paths,
        tuple(config["data"]["eval_resize"]),
    )
    descriptors = extract_descriptors(
        model,
        combined_dataset,
        runtime.device,
        batch_size=int(challenge_config.get("batch_size", 64)),
        num_workers=int(
            challenge_config.get(
                "num_workers",
                config["data"].get("num_workers", 8),
            )
        ),
        use_vlm=False,
        runtime=runtime,
    )

    payload = None
    if runtime.is_main:
        try:
            database_count = len(database_paths)
            database = descriptors[:database_count]
            queries = descriptors[database_count:]
            top_k = int(challenge_config.get("top_k", 100))
            predictions = search_topk(
                database,
                queries,
                top_k,
                runtime.device,
                int(challenge_config.get("search_batch_size", 256)),
            )
            output_dir = runtime.run_dir / f"epoch_{epoch + 1:02d}_msls_challenge"
            output_dir.mkdir(parents=True, exist_ok=True)
            label = f"epoch_{epoch + 1:02d}"
            prediction_path = (
                output_dir
                / f"{label}_msls_challenge_im2im_{local_args.subtask}_predictions.csv"
            )
            metrics_path = (
                output_dir
                / f"{label}_msls_challenge_im2im_{local_args.subtask}_metrics.txt"
            )
            write_predictions(
                prediction_path,
                query_keys,
                database_keys,
                predictions,
            )
            exit_code = run_official_evaluation(
                local_args,
                prediction_path,
                metrics_path,
            )
            if exit_code:
                raise RuntimeError(
                    "MSLS Challenge official evaluator failed with "
                    f"exit code {exit_code}; see {metrics_path.with_suffix('.log')}"
                )
            result = _parse_msls_challenge_metrics(metrics_path)
            result.update(
                {
                    "database_count": len(database_paths),
                    "query_count": len(query_paths),
                    "prediction_path": str(prediction_path),
                    "metrics_path": str(metrics_path),
                }
            )
            payload = {"result": result, "error": None}
            logging.info(
                "MSLS Challenge epoch %02d: %s",
                epoch + 1,
                ", ".join(
                    f"R@{key}: {value:.1f}"
                    for key, value in result["recall"].items()
                ),
            )
        except Exception as error:
            payload = {"result": None, "error": repr(error)}
    payload = _broadcast_result(runtime, payload)
    if payload["error"] is not None:
        raise RuntimeError(payload["error"])
    return payload["result"]


def _evaluate(runtime, model, config, epoch):
    if not config["evaluation"].get("enabled", True):
        return {1: 0.0, 5: 0.0}, {}
    benchmarks = config["evaluation"].get("benchmarks")
    if not benchmarks:
        benchmarks = [
            {
                "name": config["evaluation"]["dataset_name"],
                "dataset_name": config["evaluation"]["dataset_name"],
            }
        ]

    results = {}
    for benchmark in benchmarks:
        benchmark_name = benchmark.get("name") or benchmark["dataset_name"]
        benchmark_config = _benchmark_config(config, benchmark)
        result, _ = evaluate_model(
            model,
            benchmark_config,
            runtime.device,
            use_vlm=False,
            runtime=runtime,
        )
        result = _broadcast_result(runtime, result)
        results[benchmark_name] = {"recall": result}

    challenge_result = _evaluate_msls_challenge(
        runtime,
        model,
        config,
        epoch,
    )
    if challenge_result is not None:
        results["msls_challenge"] = challenge_result

    if runtime.is_main:
        metrics_path = runtime.run_dir / f"epoch_{epoch + 1:02d}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        logging.info("Saved epoch evaluation metrics to %s", metrics_path)

    selection_name = config["evaluation"].get(
        "selection_benchmark",
        benchmarks[0].get("name") or benchmarks[0]["dataset_name"],
    )
    if selection_name not in results or "recall" not in results[selection_name]:
        raise KeyError(f"selection benchmark has no recall result: {selection_name}")
    selection_recalls = {
        int(key): value
        for key, value in results[selection_name]["recall"].items()
    }
    return selection_recalls, results


def _build_student_optimizer(student, training_config):
    backbone_learning_rate = training_config.get("backbone_learning_rate")
    if backbone_learning_rate is None:
        return build_optimizer(student.parameters(), training_config)

    main_parameters = []
    backbone_parameters = []
    aggregation_parameters = []
    aggregation_prefixes = (
        "linear1.",
        "aggregation_adapter.",
        "aggregation.",
        "linear2.",
        "aggregation_branches.",
    )
    aggregation_learning_rate = training_config.get(
        "aggregation_learning_rate"
    )
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone.blocks.") or name.startswith("backbone.norm."):
            backbone_parameters.append(parameter)
        elif aggregation_learning_rate is not None and name.startswith(
            aggregation_prefixes
        ):
            aggregation_parameters.append(parameter)
        else:
            main_parameters.append(parameter)
    parameter_groups = [
        {
            "params": main_parameters,
            "lr": float(training_config["learning_rate"]),
            "group_name": "ivp",
        },
        {
            "params": backbone_parameters,
            "lr": float(backbone_learning_rate),
            "group_name": "dino_tail",
        },
    ]
    if aggregation_parameters:
        parameter_groups.append(
            {
                "params": aggregation_parameters,
                "lr": float(aggregation_learning_rate),
                "group_name": "aggregation_head",
            }
        )
    optimizer_name = training_config.get("optimizer", "adam")
    if optimizer_name == "adam":
        return torch.optim.Adam(parameter_groups)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameter_groups,
            weight_decay=float(training_config.get("weight_decay", 9.5e-9)),
        )
    raise ValueError(
        "differential backbone learning rate currently supports adam/adamw, "
        f"got {optimizer_name}"
    )


def main():
    cli = parse_args()
    config = load_config(cli.config, cli.overrides)
    require_sections(config, "experiment", "model", "data", "training", "evaluation")
    runtime = setup_runtime(config)
    try:
        if runtime.is_main:
            with (runtime.run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
        model_config = config["model"]
        stage = int(model_config["stage"])

        teacher = load_teacher(
            model_config,
            checkpoint_path=model_config["teacher_path"],
        ).to(runtime.device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False

        student = build_student(model_config, load_foundation=True)
        student.load_teacher_weights(project_path(model_config["teacher_path"]))
        initialization_path = model_config.get("initialization_path")
        if initialization_path:
            aggregation_adapter_dim = int(
                model_config.get("aggregation_adapter_dim", 0)
            )
            aggregation_num_branches = int(
                model_config.get("aggregation_num_branches", 1)
            )
            compatible_aggregation = (
                aggregation_adapter_dim > 0
                or aggregation_num_branches > 1
                or int(model_config.get("aggregation_wide_dim", 0)) > 0
            )
            _, initialization_result = load_model_checkpoint(
                student,
                project_path(initialization_path),
                prefer_student=True,
                strict=not compatible_aggregation,
            )
            if compatible_aggregation:
                allowed_prefixes = (
                    "aggregation_adapter.",
                    "aggregation_branches.",
                )
                unexpected_missing = [
                    key
                    for key in initialization_result.missing_keys
                    if not key.startswith(allowed_prefixes)
                ]
                if unexpected_missing or initialization_result.unexpected_keys:
                    raise RuntimeError(
                        "Could not compatibly initialize the expanded aggregation "
                        f"head: missing={unexpected_missing[:10]}, "
                        f"unexpected={initialization_result.unexpected_keys[:10]}"
                    )
                if runtime.is_main:
                    logging.info(
                        "Initialized aggregation adapter with %d new tensors",
                        len(initialization_result.missing_keys),
                    )
                if any(
                    key.startswith("aggregation_branches.")
                    for key in initialization_result.missing_keys
                ):
                    student.initialize_aggregation_branches_from_base()
        unfreeze_last_n_layers = int(
            model_config.get("unfreeze_last_n_layers", 0)
        )
        student.set_train_stage(
            stage,
            unfreeze_last_n_layers=unfreeze_last_n_layers,
            train_aggregation_head=bool(
                model_config.get("train_aggregation_head", False)
            ),
        )
        student.to(runtime.device)

        loader = build_train_loader(config, runtime)
        initialize_memory_bank(config, student, loader, runtime)
        optimizer = _build_student_optimizer(student, config["training"])
        scheduler = build_scheduler(optimizer, config["training"], len(loader))
        loss_fn, miner = build_metric_objective()
        if runtime.is_main:
            trainable_blocks = [
                index
                for index, block in enumerate(student.backbone.blocks)
                if any(parameter.requires_grad for parameter in block.parameters())
            ]
            logging.info("Trainable DINO block indices: %s", trainable_blocks)
            logging.info(
                "Trainable student parameters: %d",
                sum(
                    parameter.numel()
                    for parameter in student.parameters()
                    if parameter.requires_grad
                ),
            )
            logging.info(
                "Optimizer groups: %s",
                {
                    group.get("group_name", f"group_{index}"): {
                        "parameters": sum(
                            parameter.numel() for parameter in group["params"]
                        ),
                        "learning_rate": group["lr"],
                    }
                    for index, group in enumerate(optimizer.param_groups)
                },
            )

        start_epoch = 0
        best_score = 0.0
        best_r1 = 0.0
        not_improved = 0
        resume_path = config["training"].get("resume_path")
        if resume_path:
            checkpoint, _ = load_model_checkpoint(
                student,
                project_path(resume_path),
                prefer_student=True,
                strict=True,
            )
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch_num"] + 1
            best_score = float(checkpoint.get("best_r5", 0.0))
            best_r1 = float(checkpoint.get("best_r1", 0.0))
            not_improved = int(checkpoint.get("not_improved_num", 0))

        if runtime.distributed:
            student = DistributedDataParallel(
                student,
                device_ids=[runtime.local_rank],
                output_device=runtime.local_rank,
                find_unused_parameters=True,
            )

        training_config = config["training"]
        align_weight = float(training_config.get("align_weight", 1.0))
        alignment_target = training_config.get(
            "alignment_target",
            "intervention_residual",
        )
        if alignment_target not in ALIGNMENT_TARGETS:
            raise ValueError(
                f"unknown training.alignment_target={alignment_target!r}; "
                f"expected one of {sorted(ALIGNMENT_TARGETS)}"
            )
        if stage == 1 and alignment_target == "none":
            raise ValueError("Stage 1 requires a non-empty alignment target")
        need_deltas = alignment_target == "intervention_residual"
        need_feature_tokens = alignment_target == "final_local_features"
        use_metric = stage == 2
        use_amp = runtime.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        max_steps = training_config.get("max_steps")

        for epoch in range(start_epoch, training_config["epochs"]):
            if runtime.distributed:
                loader.sampler.set_epoch(epoch)
            student.train()
            epoch_losses = []
            epoch_alignments = []
            progress = tqdm(loader, disable=not runtime.is_main, ncols=130)
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
                with torch.no_grad(), torch.autocast(
                    device_type=runtime.device.type,
                    enabled=use_amp,
                ):
                    teacher_output = teacher(
                        images,
                        hidden,
                        masks,
                        return_xattn_deltas=need_deltas,
                        return_feature_tokens=need_feature_tokens,
                        return_dict=True,
                    )
                with torch.autocast(
                    device_type=runtime.device.type,
                    enabled=use_amp,
                ):
                    student_output = student(
                        images,
                        return_diagnostics=True,
                        return_xattn_deltas=need_deltas,
                        return_feature_tokens=need_feature_tokens,
                    )
                    teacher_metric_descriptor = _expand_teacher_descriptor(
                        teacher_output["descriptor"],
                        student_output["descriptor"].shape[-1],
                    )
                    alignment, aligned_units = distillation_alignment_loss(
                        alignment_target,
                        teacher_output,
                        student_output,
                    )
                    loss = align_weight * alignment
                    mined_pairs = None
                    if use_metric:
                        descriptors = torch.cat(
                            [
                                teacher_metric_descriptor,
                                student_output["descriptor"],
                            ],
                            dim=0,
                        )
                        joint_labels = torch.cat([labels, labels], dim=0)
                        metric, mined_pairs = metric_loss(
                            loss_fn,
                            miner,
                            descriptors,
                            joint_labels,
                        )
                        loss = loss + metric
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_losses.append(loss.detach().float().item())
                epoch_alignments.append(alignment.detach().float().item())
                if runtime.is_main:
                    diagnostics = student_output["diagnostics"]
                    learning_rates = {
                        group.get("group_name", f"group_{index}"): group["lr"]
                        for index, group in enumerate(optimizer.param_groups)
                    }
                    progress.set_postfix(
                        loss=f"{epoch_losses[-1]:.4f}",
                        align=f"{epoch_alignments[-1]:.4f}",
                        units=aligned_units,
                        entropy=f"{diagnostics['router_entropy'].item():.3f}",
                        ivp_lr=f"{learning_rates.get('ivp', 0.0):.2e}",
                        dino_lr=f"{learning_rates.get('dino_tail', 0.0):.2e}",
                        head_lr=(
                            f"{learning_rates.get('aggregation_head', 0.0):.2e}"
                        ),
                    )
                if (
                    runtime.is_main
                    and training_config.get("log_pair_stats")
                    and mined_pairs is not None
                ):
                    stats = pair_statistics(mined_pairs, teacher_output["descriptor"].shape[0])
                    stats["agreement"] = descriptor_agreement(
                        teacher_metric_descriptor,
                        student_output["descriptor"],
                    )
                    logging.debug("Pair statistics: %s", stats)
                if max_steps is not None and step + 1 >= max_steps:
                    break

            if runtime.is_main:
                logging.info(
                    "Epoch %02d loss %.6f alignment %.6f",
                    epoch,
                    sum(epoch_losses) / max(len(epoch_losses), 1),
                    sum(epoch_alignments) / max(len(epoch_alignments), 1),
                )
            evaluation_config = config["evaluation"]
            evaluation_interval = int(evaluation_config.get("every_n_epochs", 1))
            if evaluation_interval <= 0:
                raise ValueError("evaluation.every_n_epochs must be positive")
            should_evaluate = bool(evaluation_config.get("enabled", True)) and (
                (epoch + 1) % evaluation_interval == 0
                or epoch + 1 == training_config["epochs"]
            )
            if should_evaluate:
                recalls, epoch_evaluations = _evaluate(
                    runtime,
                    student,
                    config,
                    epoch,
                )
            else:
                recalls, epoch_evaluations = {}, {}
            current_r1 = recalls.get(1, 0.0)
            current_score = current_r1 + recalls.get(5, 0.0)
            is_best = should_evaluate and current_score > best_score
            is_best_r1 = should_evaluate and current_r1 > best_r1
            if is_best:
                best_score = current_score
                not_improved = 0
            elif should_evaluate:
                not_improved += 1
            if should_evaluate:
                best_r1 = max(best_r1, current_r1)

            if runtime.is_main:
                state = unwrap_model(student).state_dict()
                payload = {
                    "epoch_num": epoch,
                    "model_state_dict": state,
                    "student_state_dict": state,
                    "teacher_checkpoint": model_config["teacher_path"],
                    "teacher_trainable_state_dict": None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "recalls": recalls,
                    "best_r5": best_score,
                    "best_r1": best_r1,
                    "not_improved_num": not_improved,
                    "epoch_evaluations": epoch_evaluations,
                    "ivp_meta": {
                        "stage": stage,
                        "unfreeze_last_n_layers": unfreeze_last_n_layers,
                        "aggregation_adapter_dim": int(
                            model_config.get("aggregation_adapter_dim", 0)
                        ),
                        "aggregation_num_branches": int(
                            model_config.get("aggregation_num_branches", 1)
                        ),
                        "aggregation_wide_dim": int(
                            model_config.get("aggregation_wide_dim", 0)
                        ),
                        "aggregation_wide_hidden_dim": int(
                            model_config.get(
                                "aggregation_wide_hidden_dim",
                                0,
                            )
                        ),
                        "aggregation_branch_dim": int(
                            model_config.get(
                                "aggregation_branch_dim",
                                model_config["output_dim"],
                            )
                        ),
                        "train_aggregation_head": bool(
                            model_config.get("train_aggregation_head", False)
                        ),
                        "bank_size": model_config["bank_size"],
                        "router_dim": model_config["router_dim"],
                        "top_k": model_config["top_k"],
                        "router_type": "single_summary",
                        "router_layers": model_config["router_layers"],
                        "memory_init": model_config["memory_init"],
                        "align_weight": align_weight,
                        "alignment_target": alignment_target,
                        "crossattn_layer_mode": model_config["crossattn_layer_mode"],
                        "vlm_dim": model_config["vlm_dim"],
                    },
                    "student_dim": model_config["output_dim"],
                    "teacher_dim": model_config.get(
                        "teacher_output_dim",
                        model_config["output_dim"],
                    ),
                    "config": config,
                }
                save_training_checkpoint(
                    runtime.run_dir,
                    payload,
                    is_best=is_best,
                    is_best_r1=is_best_r1,
                )
            stop = should_evaluate and not_improved >= training_config["patience"]
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
