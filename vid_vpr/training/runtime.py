import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from vid_vpr.config import project_path


@dataclass
class Runtime:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    distributed: bool
    is_main: bool
    run_dir: Path


def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_runtime(config):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if distributed:
        holder = [timestamp if rank == 0 else None]
        dist.broadcast_object_list(holder, src=0)
        timestamp = holder[0]
    experiment = config["experiment"]
    run_dir = project_path(experiment.get("output_root", "runs")) / experiment["name"] / timestamp
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(run_dir / "info.log"),
                logging.StreamHandler(),
            ],
            force=True,
        )
    if distributed:
        dist.barrier()
    make_deterministic(experiment.get("seed", 42))
    return Runtime(
        device=device,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        distributed=distributed,
        is_main=rank == 0,
        run_dir=run_dir,
    )


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def close_runtime(runtime):
    if runtime.distributed:
        dist.barrier()
        dist.destroy_process_group()
