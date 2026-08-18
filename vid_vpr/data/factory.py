import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from vid_vpr.config import project_path
from vid_vpr.data.collate import train_collate
from vid_vpr.data.gsv_cities import GSVCitiesDataset


TRAIN_CITIES = [
    "Bangkok",
    "BuenosAires",
    "LosAngeles",
    "MexicoCity",
    "OSL",
    "Rome",
    "Barcelona",
    "Chicago",
    "Madrid",
    "Miami",
    "Phoenix",
    "TRT",
    "Boston",
    "Lisbon",
    "Medellin",
    "Minneapolis",
    "PRG",
    "WashingtonDC",
    "Brussels",
    "London",
    "Melbourne",
    "Osaka",
    "PRS",
]


def build_train_loader(config, runtime):
    data_config = config["data"]
    global_batch_size = config["training"]["global_batch_size"]
    shuffle_train = bool(data_config.get("shuffle_train", False))
    if global_batch_size % runtime.world_size:
        raise ValueError(
            f"global_batch_size={global_batch_size} must be divisible by world_size={runtime.world_size}"
        )
    transform = transforms.Compose(
        [
            transforms.Resize(
                tuple(data_config["train_resize"]),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandAugment(
                num_ops=3,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset = GSVCitiesDataset(
        cities=TRAIN_CITIES,
        img_per_place=data_config.get("images_per_place", 4),
        min_img_per_place=data_config.get("images_per_place", 4),
        random_sample_from_each_place=True,
        transform=transform,
        base_path=project_path(data_config["gsv_root"]),
        vlm_cache_dir=project_path(data_config["train_vlm_cache"]),
    )
    sampler = None
    if runtime.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=shuffle_train,
            drop_last=False,
        )
    num_workers = data_config.get("num_workers", 8)
    loader_options = {
        "dataset": dataset,
        "batch_size": global_batch_size // runtime.world_size,
        "sampler": sampler,
        "shuffle": shuffle_train if sampler is None else False,
        "num_workers": num_workers,
        "drop_last": False,
        "pin_memory": runtime.device.type == "cuda",
        "collate_fn": train_collate,
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = True
        loader_options["prefetch_factor"] = 4
    return DataLoader(**loader_options)
