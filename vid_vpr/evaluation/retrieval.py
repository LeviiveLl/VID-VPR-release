import logging
import os
import time
from types import SimpleNamespace

import faiss
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from vid_vpr.config import project_path
from vid_vpr.data.benchmarks import BaseDataset
from vid_vpr.data.collate import evaluation_collate


def build_benchmark_dataset(config, use_vlm=False):
    data_config = config["data"]
    eval_config = config["evaluation"]
    args = SimpleNamespace(
        resize=list(data_config["eval_resize"]),
        test_method=eval_config.get("test_method", "hard_resize"),
        val_positive_dist_threshold=eval_config.get("positive_distance", 25),
        use_vlm_crossattn=use_vlm,
        eval_vlm_cache_dir=str(project_path(data_config.get("eval_vlm_cache")))
        if use_vlm
        else None,
        vlm_cache_dir=None,
    )
    return BaseDataset(
        args,
        str(project_path(data_config["benchmark_root"])),
        eval_config["dataset_name"],
        eval_config.get("split", "test"),
    )


@torch.inference_mode()
def extract_descriptors(
    model,
    dataset,
    device,
    batch_size=32,
    num_workers=8,
    use_vlm=False,
    runtime=None,
    query_batch_size=None,
):
    distributed = runtime is not None and runtime.distributed
    dataset_parts = [("descriptors", dataset, batch_size)]
    if query_batch_size is not None and query_batch_size != batch_size:
        dataset_parts = [
            ("database descriptors", Subset(dataset, range(dataset.database_num)), batch_size),
            (
                "query descriptors",
                Subset(dataset, range(dataset.database_num, len(dataset))),
                query_batch_size,
            ),
        ]
    local_indices = []
    local_descriptors = []
    model.eval()
    for description, dataset_part, part_batch_size in dataset_parts:
        sampler = None
        if distributed:
            sampler = DistributedSampler(
                dataset_part,
                num_replicas=runtime.world_size,
                rank=runtime.rank,
                shuffle=False,
                drop_last=False,
            )
        loader = DataLoader(
            dataset_part,
            batch_size=part_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=evaluation_collate,
            sampler=sampler,
        )
        for batch in tqdm(
            loader,
            desc=f"Extracting {description}",
            ncols=100,
            disable=distributed and not runtime.is_main,
        ):
            if use_vlm:
                images, indices, hidden, masks = batch
                output = model(
                    images.to(device, non_blocking=True),
                    hidden.to(device, non_blocking=True),
                    masks.to(device, non_blocking=True),
                )
            else:
                images, indices = batch
                output = model(images.to(device, non_blocking=True))
            output = output.detach().float().cpu().numpy()
            local_indices.append(indices.numpy())
            local_descriptors.append(output)
    if not local_descriptors:
        raise RuntimeError("the evaluation dataset is empty")

    local_indices = np.concatenate(local_indices, axis=0)
    local_descriptors = np.concatenate(local_descriptors, axis=0).astype(np.float32, copy=False)

    if not distributed:
        descriptors = np.empty((len(dataset), local_descriptors.shape[1]), dtype=np.float32)
        descriptors[local_indices] = local_descriptors
        return descriptors

    shard_dir = runtime.run_dir / "distributed_descriptor_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"rank_{runtime.rank}.npz"
    tmp_shard_path = shard_dir / f"rank_{runtime.rank}.tmp.npz"
    np.savez(tmp_shard_path, indices=local_indices, descriptors=local_descriptors)
    os.replace(tmp_shard_path, shard_path)
    dist.barrier()

    if not runtime.is_main:
        return None

    shard_paths = [shard_dir / f"rank_{rank}.npz" for rank in range(runtime.world_size)]
    wait_start = time.time()
    while not all(path.exists() for path in shard_paths):
        if time.time() - wait_start > 12 * 60 * 60:
            missing = [str(path) for path in shard_paths if not path.exists()]
            raise TimeoutError(f"Timed out waiting for descriptor shards: {missing}")
        time.sleep(1)

    descriptors = np.empty((len(dataset), local_descriptors.shape[1]), dtype=np.float32)
    for path in shard_paths:
        with np.load(path) as payload:
            indices = payload["indices"]
            descriptors[indices] = payload["descriptors"]
    return descriptors


def compute_recalls(descriptors, dataset, recall_values=(1, 5, 10, 100)):
    database = np.ascontiguousarray(descriptors[: dataset.database_num], dtype=np.float32)
    queries = np.ascontiguousarray(descriptors[dataset.database_num :], dtype=np.float32)
    max_k = min(max(recall_values), len(database))
    index = faiss.IndexFlatL2(database.shape[1])
    index.add(database)
    _, predictions = index.search(queries, max_k)

    positives = dataset.get_positives()
    recalls = np.zeros(len(recall_values), dtype=np.float64)
    for query_index, prediction in enumerate(predictions):
        for recall_index, k in enumerate(recall_values):
            if np.any(np.isin(prediction[:k], positives[query_index])):
                recalls[recall_index:] += 1
                break
    recalls = recalls / len(queries) * 100.0
    return recalls, predictions


def evaluate_model(model, config, device, use_vlm=False, runtime=None):
    dataset = build_benchmark_dataset(config, use_vlm=use_vlm)
    descriptors = extract_descriptors(
        model,
        dataset,
        device,
        batch_size=config["evaluation"].get("batch_size", 32),
        num_workers=config["data"].get("num_workers", 8),
        use_vlm=use_vlm,
        runtime=runtime,
        query_batch_size=config["evaluation"].get("query_batch_size") if use_vlm else None,
    )
    if descriptors is None:
        return None, None
    recall_values = tuple(config["evaluation"].get("recall_values", [1, 5, 10, 100]))
    recalls, predictions = compute_recalls(descriptors, dataset, recall_values)
    result = dict(zip(recall_values, recalls.tolist()))
    logging.info(
        "Recalls on %s: %s",
        dataset,
        ", ".join(f"R@{k}: {result[k]:.1f}" for k in recall_values),
    )
    return result, predictions
