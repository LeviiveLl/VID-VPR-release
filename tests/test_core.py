import torch

from vid_vpr.models.aggregation import PriorUtilityTransportAggregator
from vid_vpr.models.student import SyntheticPriorRouter


def test_synthetic_prior_router_shapes():
    router = SyntheticPriorRouter(
        visual_dim=8, vlm_dim=6, bank_size=16, router_dim=4, top_k=3
    )
    tokens, stats = router(torch.randn(2, 5, 8), layer_index=2)
    assert tokens.shape == (2, 3, 6)
    assert torch.isfinite(tokens).all()
    assert "router_entropy" in stats


def test_layer_aware_aggregation_is_normalized():
    model = PriorUtilityTransportAggregator(
        num_channels=8,
        num_clusters=2,
        cluster_dim=3,
        token_dim=4,
        dropout=0.0,
        response_layers=(2, 4),
        response_projection_dim=2,
        response_hidden_dim=4,
    ).eval()
    feature_map = torch.randn(2, 8, 2, 2)
    cls_token = torch.randn(2, 8)
    deltas = {2: torch.randn(2, 4, 8), 4: torch.randn(2, 4, 8)}
    descriptor = model((feature_map, cls_token), patch_deltas=deltas)
    assert descriptor.shape == (2, 10)
    torch.testing.assert_close(descriptor.norm(dim=-1), torch.ones(2))
