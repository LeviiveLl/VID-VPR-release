import math

import torch


def build_optimizer(parameters, config):
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    name = config.get("optimizer", "adam")
    learning_rate = config["learning_rate"]
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=9.5e-9)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=0.001,
        )
    raise ValueError(f"unknown optimizer: {name}")


def build_scheduler(optimizer, config, steps_per_epoch):
    name = config.get("scheduler", "cosine")
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if name != "cosine":
        raise ValueError(f"unknown scheduler: {name}")

    scheduler_steps = max(
        steps_per_epoch * config.get("scheduler_epochs", config["epochs"]),
        1,
    )
    warmup_steps = min(
        int(steps_per_epoch * config.get("warmup_epochs", 0.0)),
        max(scheduler_steps - 1, 0),
    )
    min_ratio = config.get("min_lr_ratio", 0.05)

    def scale(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        if step >= scheduler_steps:
            return min_ratio
        progress = (step - warmup_steps) / max(scheduler_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
