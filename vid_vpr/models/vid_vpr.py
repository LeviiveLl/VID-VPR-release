import math

import torch
from torch import nn

from vid_vpr.models.aggregation import PriorUtilityTransportAggregator
from vid_vpr.models.student import ImageOnlyStudent


class VIDVPR(nn.Module):
    """Final image-only VID-VPR model used for descriptor extraction."""

    def __init__(
        self,
        student=None,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        dropout=0.3,
        sinkhorn_iters=3,
        response_layers=(20, 22),
        response_projection_dim=16,
        response_hidden_dim=64,
        response_max_strength=0.5,
        **student_config,
    ):
        super().__init__()
        self.student = student if student is not None else ImageOnlyStudent(**student_config)
        self.response_layers = tuple(int(index) for index in response_layers)
        self.aggregator = PriorUtilityTransportAggregator(
            num_channels=1024,
            num_clusters=num_clusters,
            cluster_dim=cluster_dim,
            token_dim=token_dim,
            dropout=dropout,
            sinkhorn_iters=sinkhorn_iters,
            response_layers=self.response_layers,
            response_projection_dim=response_projection_dim,
            response_hidden_dim=response_hidden_dim,
            max_response_strength=response_max_strength,
        )
        self.output_dim = num_clusters * cluster_dim + token_dim

    def forward(
        self,
        images,
        return_xattn_deltas=False,
        return_diagnostics=False,
    ):
        output = self.student.forward_feature_tokens(
            images,
            return_xattn_deltas=True,
            xattn_delta_indices=self.response_layers,
            return_router_state=return_diagnostics,
        )
        if return_diagnostics:
            tokens = output["tokens"]
            deltas = output["xattn_deltas"]
            router_state = output["router_state"]
        else:
            tokens, deltas = output
            router_state = None

        register_count = int(getattr(self.student.backbone, "num_register_tokens", 0))
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1 + register_count :]
        side = math.isqrt(patch_tokens.shape[1])
        if side * side != patch_tokens.shape[1]:
            raise ValueError(
                "Layer-aware aggregation requires a square patch grid, "
                f"got {patch_tokens.shape[1]} tokens"
            )
        feature_map = patch_tokens.reshape(
            patch_tokens.shape[0], side, side, patch_tokens.shape[-1]
        ).permute(0, 3, 1, 2)
        patch_deltas = {
            index: deltas[index][:, 1 + register_count :]
            for index in self.response_layers
        }
        aggregation = self.aggregator(
            (feature_map, cls_token),
            patch_deltas=patch_deltas,
            return_diagnostics=return_diagnostics,
        )
        if not return_diagnostics and not return_xattn_deltas:
            return aggregation
        result = aggregation if return_diagnostics else {"descriptor": aggregation}
        if return_xattn_deltas:
            result["xattn_deltas"] = deltas
        if return_diagnostics:
            result["router_state"] = router_state
        return result


def load_vid_vpr(checkpoint_path, map_location="cpu", strict=True):
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    config = payload.get("model_config", {}) if isinstance(payload, dict) else {}
    model = VIDVPR(**config)
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict, strict=strict)
    return model
