import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from vid_vpr.backbones.dinov2.vision_transformer import vit_large
from vid_vpr.models.cross_attention import VLMCrossAttention
from vid_vpr.models.pooling import Flatten, GeM, L2Norm


class VLMConditionedTeacher(nn.Module):
    """DINOv2-Large VPR model conditioned by cached VLM hidden states."""

    def __init__(
        self,
        output_dim=4096,
        vlm_dim=2048,
        crossattn_heads=16,
        crossattn_every_n=2,
        crossattn_layer_mode="late_stride",
        foundation_model_path=None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.backbone = vit_large(
            patch_size=14,
            img_size=518,
            init_values=1,
            block_chunks=0,
            use_vlm_crossattn=True,
            vlm_dim=vlm_dim,
            crossattn_heads=crossattn_heads,
            crossattn_every_n=crossattn_every_n,
            crossattn_layer_mode=crossattn_layer_mode,
        )
        self.aggregation = nn.Sequential(L2Norm(), GeM(), Flatten())
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, output_dim)
        if foundation_model_path is not None:
            self.load_foundation_weights(foundation_model_path)

    def load_foundation_weights(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = self.backbone.state_dict()
        compatible = OrderedDict(
            (key, value)
            for key, value in state.items()
            if key in model_state and model_state[key].shape == value.shape
        )
        result = self.backbone.load_state_dict(compatible, strict=False)
        required_missing = [
            key
            for key in result.missing_keys
            if not key.startswith(("vlm_projector.", "vlm_crossattns."))
        ]
        if required_missing:
            raise RuntimeError(f"foundation checkpoint is missing DINOv2 weights: {required_missing[:10]}")
        return result

    def set_trainable_layers(self, unfreeze_last_n_layers=0):
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.vlm_projector.parameters():
            parameter.requires_grad = True
        for module in self.backbone.vlm_crossattns:
            for parameter in module.parameters():
                parameter.requires_grad = True
        for parameter in self.linear1.parameters():
            parameter.requires_grad = True
        for parameter in self.aggregation.parameters():
            parameter.requires_grad = True
        for parameter in self.linear2.parameters():
            parameter.requires_grad = True
        if unfreeze_last_n_layers > 0:
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad = True
            start = max(len(self.backbone.blocks) - unfreeze_last_n_layers, 0)
            for block in self.backbone.blocks[start:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_injection_trainable(self):
        """Freeze VPR features and tune only the VLM injection path."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.vlm_projector.parameters():
            parameter.requires_grad = True
        for module in self.backbone.vlm_crossattns:
            if isinstance(module, VLMCrossAttention):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        images,
        vl_embeds,
        vl_attention_mask=None,
        return_xattn_deltas=False,
        return_feature_tokens=False,
        return_dict=False,
    ):
        features = self.backbone(
            images,
            vl_embeds=vl_embeds,
            vl_attention_mask=vl_attention_mask,
            return_xattn_deltas=return_xattn_deltas,
        )
        patch_tokens = features["x_norm_patchtokens"]
        batch_size, patch_count, dim = patch_tokens.shape
        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise ValueError(f"GeM requires a square patch grid, got {patch_count} tokens")
        patches = self.linear1(
            patch_tokens.view(batch_size, side, side, dim)
        ).permute(0, 3, 1, 2)
        descriptor = F.normalize(self.linear2(self.aggregation(patches)), p=2, dim=-1)
        if return_dict or return_xattn_deltas or return_feature_tokens:
            output = {"descriptor": descriptor}
            if return_xattn_deltas:
                output["xattn_deltas"] = features["xattn_deltas"]
            if return_feature_tokens:
                output["feature_tokens"] = patch_tokens
            return output
        return descriptor
