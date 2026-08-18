import torch

from vid_vpr.config import project_path
from vid_vpr.models.student import ImageOnlyStudent
from vid_vpr.models.teacher import VLMConditionedTeacher
from vid_vpr.models.vid_vpr import VIDVPR
from vid_vpr.training.checkpoints import load_model_checkpoint


def build_teacher(config, load_foundation=False):
    foundation_path = config.get("foundation_path") if load_foundation else None
    return VLMConditionedTeacher(
        output_dim=config.get("output_dim", 4096),
        vlm_dim=config.get("vlm_dim", 2048),
        crossattn_heads=config.get("crossattn_heads", 16),
        crossattn_every_n=config.get("crossattn_every_n", 2),
        crossattn_layer_mode=config.get("crossattn_layer_mode", "late_stride"),
        foundation_model_path=(str(project_path(foundation_path)) if foundation_path else None),
    )


def build_image_student(config, load_foundation=False):
    foundation_path = config.get("foundation_path") if load_foundation else None
    return ImageOnlyStudent(
        output_dim=config.get("output_dim", 4096),
        vlm_dim=config.get("vlm_dim", 2048),
        crossattn_heads=config.get("crossattn_heads", 16),
        crossattn_every_n=config.get("crossattn_every_n", 2),
        crossattn_layer_mode=config.get("crossattn_layer_mode", "late_stride"),
        bank_size=config.get("bank_size", 256),
        router_dim=config.get("router_dim", 512),
        top_k=config.get("top_k", 8),
        router_layers=config.get("router_layers", (8, 12, 16, 20)),
        foundation_model_path=(str(project_path(foundation_path)) if foundation_path else None),
    )


build_student = build_image_student


def build_vid_vpr(config):
    return VIDVPR(
        student=build_image_student(config),
        num_clusters=config.get("num_clusters", 64),
        cluster_dim=config.get("cluster_dim", 128),
        token_dim=config.get("token_dim", 256),
        dropout=config.get("dropout", 0.3),
        sinkhorn_iters=config.get("sinkhorn_iters", 3),
        response_layers=config.get("response_layers", (20, 22)),
        response_projection_dim=config.get("response_projection_dim", 16),
        response_hidden_dim=config.get("response_hidden_dim", 64),
        response_max_strength=config.get("response_max_strength", 0.5),
    )


def load_teacher(config, checkpoint_path):
    model = build_teacher(config)
    load_model_checkpoint(model, project_path(checkpoint_path), strict=True)
    return model


def load_student(config, checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_vid_vpr(payload.get("model_config", config))
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict, strict=True)
    return model
