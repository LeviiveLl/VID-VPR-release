import math

import torch
import torch.nn.functional as F
from torch import nn

# Balanced Sinkhorn assignment is used to aggregate local place evidence.


def log_optimal_transport(log_a, log_b, scores, num_iters=3, reg=1.0):
    """Differentiable Sinkhorn scaling used for balanced local assignment."""
    scores = scores / reg
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(num_iters):
        u = log_a - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
    return scores + u.unsqueeze(2) + v.unsqueeze(1)


def get_prior_matching_probs(scores, dustbin_scores, num_iters=3, reg=1.0):
    """Compute patch-dependent assignments with stable mixed-precision numerics."""
    batch_size, cluster_count, patch_count = scores.shape
    if patch_count <= cluster_count:
        raise ValueError(
            "balanced transport requires more patches than clusters, "
            f"got {patch_count} patches and {cluster_count} clusters"
        )
    if dustbin_scores.shape != (batch_size, patch_count):
        raise ValueError(
            "dustbin_scores must have shape "
            f"{(batch_size, patch_count)}, got {tuple(dustbin_scores.shape)}"
        )

    augmented = torch.empty(
        batch_size,
        cluster_count + 1,
        patch_count,
        dtype=scores.dtype,
        device=scores.device,
    )
    augmented[:, :cluster_count] = scores
    augmented[:, cluster_count] = dustbin_scores

    # Keep this scalar in FP32. This
    # important under autocast because it promotes the Sinkhorn iterations and
    # their gradients to FP32.
    norm = -torch.tensor(
        math.log(patch_count + cluster_count),
        dtype=torch.float32,
        device=scores.device,
    )
    log_a = norm.expand(cluster_count + 1).contiguous()
    log_b = norm.expand(patch_count).contiguous()
    log_a[-1] += math.log(patch_count - cluster_count)
    log_a = log_a.expand(batch_size, -1)
    log_b = log_b.expand(batch_size, -1)
    return log_optimal_transport(log_a, log_b, augmented, num_iters, reg) - norm


class PriorRoutedTransportAggregator(nn.Module):
    """Balanced local aggregation conditioned on the internal IVP prior.

    The appearance branches retain parameter names and operations compatible with
    a balanced-aggregation checkpoint. Zero-initialized residual scales make the initial output
    identical while allowing the IVP prior to learn assignment, rejection, and
    global-token corrections during adaptation.
    """

    def __init__(
        self,
        num_channels=1024,
        prior_dim=1024,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        prior_rank=64,
        dropout=0.3,
        sinkhorn_iters=3,
        use_token_reader=False,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.prior_dim = int(prior_dim)
        self.num_clusters = int(num_clusters)
        self.cluster_dim = int(cluster_dim)
        self.token_dim = int(token_dim)
        self.prior_rank = int(prior_rank)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.use_token_reader = bool(use_token_reader)

        local_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        score_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # These four modules intentionally match the strong balanced-aggregation
        # initialization used by the existing Stage2 adaptation checkpoint.
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim),
        )
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            local_dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1),
        )
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            score_dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

        self.prior_norm = nn.LayerNorm(self.prior_dim)
        self.prior_patch_keys = nn.Conv2d(
            self.num_channels,
            self.prior_rank,
            1,
            bias=False,
        )
        self.prior_cluster_queries = nn.Linear(
            self.prior_dim,
            self.num_clusters * self.prior_rank,
        )
        self.assignment_scale = nn.Parameter(torch.zeros(()))

        self.reliability_patch = nn.Conv2d(
            self.num_channels,
            self.prior_rank,
            1,
            bias=False,
        )
        self.reliability_prior = nn.Linear(
            self.prior_dim,
            self.prior_rank,
            bias=False,
        )
        self.reliability_bias = nn.Parameter(torch.zeros(()))
        self.dustbin_scale = nn.Parameter(torch.zeros(()))

        self.global_prior = nn.Sequential(
            nn.LayerNorm(self.num_channels + self.prior_dim),
            nn.Linear(self.num_channels + self.prior_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.token_dim),
        )
        self.global_scale = nn.Parameter(torch.zeros(()))

        if self.use_token_reader:
            self.prior_token_norm = nn.LayerNorm(self.prior_dim)
            self.token_query = nn.Linear(
                self.num_channels,
                self.prior_rank,
                bias=False,
            )
            self.token_key = nn.Linear(
                self.prior_dim,
                self.prior_rank,
                bias=False,
            )
            self.token_value = nn.Linear(
                self.prior_dim,
                self.prior_rank,
                bias=False,
            )
            self.token_assignment = nn.Sequential(
                nn.LayerNorm(self.prior_rank),
                nn.Linear(self.prior_rank, self.num_clusters, bias=False),
            )
            self.token_assignment_scale = nn.Parameter(torch.zeros(()))

    def _prior_summary(self, prior_tokens, batch_size, device, dtype):
        if prior_tokens is None:
            return torch.zeros(batch_size, self.prior_dim, device=device, dtype=dtype)
        if prior_tokens.dim() != 3 or prior_tokens.shape[0] != batch_size:
            raise ValueError(
                "prior_tokens must have shape [batch, tokens, channels], "
                f"got {tuple(prior_tokens.shape)}"
            )
        if prior_tokens.shape[-1] != self.prior_dim:
            raise ValueError(
                f"expected prior dimension {self.prior_dim}, "
                f"got {prior_tokens.shape[-1]}"
            )
        return self.prior_norm(prior_tokens.sum(dim=1))

    def forward(self, inputs, prior_tokens=None, return_diagnostics=False):
        feature_map, cls_token = inputs
        batch_size = feature_map.shape[0]
        prior = self._prior_summary(
            prior_tokens,
            batch_size,
            feature_map.device,
            feature_map.dtype,
        )

        local_features = self.cluster_features(feature_map).flatten(2)
        appearance_scores = self.score(feature_map).flatten(2)

        patch_keys = self.prior_patch_keys(feature_map).flatten(2)
        cluster_queries = self.prior_cluster_queries(prior).view(
            batch_size,
            self.num_clusters,
            self.prior_rank,
        )
        prior_scores = torch.einsum("bkr,brn->bkn", cluster_queries, patch_keys)
        prior_scores = prior_scores / math.sqrt(self.prior_rank)
        assignment_strength = torch.tanh(self.assignment_scale)
        scores = appearance_scores + assignment_strength * prior_scores

        token_assignment_strength = scores.new_zeros(())
        token_attention_entropy = scores.new_zeros(())
        if self.use_token_reader and prior_tokens is not None:
            normalized_tokens = self.prior_token_norm(prior_tokens)
            patch_features = feature_map.flatten(2).transpose(1, 2)
            patch_queries = self.token_query(patch_features)
            prior_keys = self.token_key(normalized_tokens)
            prior_values = self.token_value(normalized_tokens)
            token_attention = torch.matmul(patch_queries, prior_keys.transpose(1, 2))
            token_attention = token_attention / math.sqrt(self.prior_rank)
            token_attention = token_attention.softmax(dim=-1)
            token_context = torch.matmul(token_attention, prior_values)
            token_scores = self.token_assignment(token_context).transpose(1, 2)
            token_assignment_strength = torch.tanh(self.token_assignment_scale)
            scores = scores + token_assignment_strength * token_scores
            token_attention_entropy = -(
                token_attention.clamp_min(1e-8)
                * token_attention.clamp_min(1e-8).log()
            ).sum(dim=-1).mean()

        reliability_keys = self.reliability_patch(feature_map).flatten(2)
        reliability_query = self.reliability_prior(prior)
        reliability_logits = torch.einsum(
            "br,brn->bn",
            reliability_query,
            reliability_keys,
        ) / math.sqrt(self.prior_rank)
        reliability = torch.sigmoid(reliability_logits + self.reliability_bias)
        dustbin_strength = torch.tanh(self.dustbin_scale)
        dustbin_scores = self.dust_bin + dustbin_strength * (1.0 - reliability)

        log_assignment = get_prior_matching_probs(
            scores,
            dustbin_scores,
            num_iters=self.sinkhorn_iters,
        )
        assignment = log_assignment.exp()
        cluster_assignment = assignment[:, :-1]

        local_descriptor = torch.einsum(
            "bcn,bkn->bck",
            local_features,
            cluster_assignment,
        )
        local_descriptor = F.normalize(local_descriptor, p=2, dim=1).flatten(1)

        global_descriptor = self.token_features(cls_token)
        global_residual = self.global_prior(torch.cat([cls_token, prior], dim=-1))
        global_descriptor = global_descriptor + torch.tanh(self.global_scale) * global_residual
        descriptor = torch.cat(
            [
                F.normalize(global_descriptor, p=2, dim=-1),
                local_descriptor,
            ],
            dim=-1,
        )
        descriptor = F.normalize(descriptor, p=2, dim=-1)

        if not return_diagnostics:
            return descriptor
        return {
            "descriptor": descriptor,
            "prior_reliability": reliability,
            "dustbin_probability": assignment[:, -1],
            "assignment_strength": assignment_strength,
            "dustbin_strength": dustbin_strength,
            "global_strength": torch.tanh(self.global_scale),
            "token_assignment_strength": token_assignment_strength,
            "token_attention_entropy": token_attention_entropy,
        }


class PriorConsensusTransportAggregator(nn.Module):
    """Calibrate local evidence transport with an internal IVP prior.

    The appearance path is checkpoint-compatible with the balanced transport
    baseline. Zero-initialized residual gates recover that path exactly, while
    the prior path exposes assignment agreement and reliability explicitly.
    """

    def __init__(
        self,
        num_channels=1024,
        prior_dim=1024,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        prior_rank=64,
        dropout=0.3,
        sinkhorn_iters=3,
        appearance_temperature=1.0,
        prior_temperature=1.0,
        enable_prior_assignment=True,
        enable_consensus_dustbin=True,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.prior_dim = int(prior_dim)
        self.num_clusters = int(num_clusters)
        self.cluster_dim = int(cluster_dim)
        self.token_dim = int(token_dim)
        self.prior_rank = int(prior_rank)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.appearance_temperature = float(appearance_temperature)
        self.prior_temperature = float(prior_temperature)
        self.enable_prior_assignment = bool(enable_prior_assignment)
        self.enable_consensus_dustbin = bool(enable_consensus_dustbin)
        if self.appearance_temperature <= 0 or self.prior_temperature <= 0:
            raise ValueError("consensus temperatures must be positive")

        local_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        score_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim),
        )
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            local_dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1),
        )
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            score_dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

        self.prior_norm = nn.LayerNorm(self.prior_dim)
        self.prior_patch_keys = nn.Conv2d(
            self.num_channels,
            self.prior_rank,
            1,
            bias=False,
        )
        self.prior_cluster_queries = nn.Linear(
            self.prior_dim,
            self.num_clusters * self.prior_rank,
        )
        self.assignment_scale = nn.Parameter(torch.zeros(()))
        self.consensus_dustbin_scale = nn.Parameter(torch.zeros(()))

    def _prior_summary(self, prior_tokens, batch_size, device, dtype):
        if prior_tokens is None:
            return None
        if prior_tokens.dim() != 3 or prior_tokens.shape[0] != batch_size:
            raise ValueError(
                "prior_tokens must have shape [batch, tokens, channels], "
                f"got {tuple(prior_tokens.shape)}"
            )
        if prior_tokens.shape[-1] != self.prior_dim:
            raise ValueError(
                f"expected prior dimension {self.prior_dim}, "
                f"got {prior_tokens.shape[-1]}"
            )
        return self.prior_norm(
            prior_tokens.to(device=device, dtype=dtype).sum(dim=1)
        )

    @staticmethod
    def _normalize_prior_scores(prior_scores):
        centered = prior_scores - prior_scores.mean(dim=1, keepdim=True)
        variance = centered.square().mean(dim=1, keepdim=True)
        return centered * torch.rsqrt(variance + 1e-6)

    @staticmethod
    def consensus_reliability(
        appearance_scores,
        prior_scores,
        appearance_temperature=1.0,
        prior_temperature=1.0,
        detach_appearance=True,
    ):
        appearance_source = (
            appearance_scores.detach() if detach_appearance else appearance_scores
        )
        appearance_assignment = (
            appearance_source / appearance_temperature
        ).softmax(dim=1)
        prior_assignment = (prior_scores / prior_temperature).softmax(dim=1)
        consensus = torch.sqrt(
            appearance_assignment.clamp_min(1e-8)
            * prior_assignment.clamp_min(1e-8)
        ).sum(dim=1)
        prior_entropy = -(
            prior_assignment.clamp_min(1e-8)
            * prior_assignment.clamp_min(1e-8).log()
        ).sum(dim=1)
        cluster_count = prior_assignment.shape[1]
        prior_confidence = 1.0 - prior_entropy / math.log(cluster_count)
        prior_confidence = prior_confidence.clamp(0.0, 1.0)
        reliability = (consensus * prior_confidence).clamp(0.0, 1.0)
        return reliability, consensus, prior_confidence, appearance_assignment, prior_assignment

    def forward(self, inputs, prior_tokens=None, return_diagnostics=False):
        feature_map, cls_token = inputs
        batch_size = feature_map.shape[0]
        prior = self._prior_summary(
            prior_tokens,
            batch_size,
            feature_map.device,
            feature_map.dtype,
        )

        local_features = self.cluster_features(feature_map).flatten(2)
        appearance_scores = self.score(feature_map).flatten(2)
        assignment_strength = torch.tanh(self.assignment_scale)
        consensus_strength = torch.tanh(self.consensus_dustbin_scale)

        if prior is None:
            prior_scores = torch.zeros_like(appearance_scores)
            reliability = appearance_scores.new_ones(
                batch_size,
                appearance_scores.shape[-1],
            )
            consensus = reliability
            prior_confidence = appearance_scores.new_zeros(reliability.shape)
            appearance_assignment = appearance_scores.softmax(dim=1)
            prior_assignment = prior_scores.softmax(dim=1)
            scores = appearance_scores
            dustbin_scores = self.dust_bin.expand_as(reliability)
        else:
            patch_keys = self.prior_patch_keys(feature_map).flatten(2)
            cluster_queries = self.prior_cluster_queries(prior).view(
                batch_size,
                self.num_clusters,
                self.prior_rank,
            )
            prior_scores = torch.einsum(
                "bkr,brn->bkn",
                cluster_queries,
                patch_keys,
            ) / math.sqrt(self.prior_rank)
            prior_scores = self._normalize_prior_scores(prior_scores)
            (
                reliability,
                consensus,
                prior_confidence,
                appearance_assignment,
                prior_assignment,
            ) = self.consensus_reliability(
                appearance_scores,
                prior_scores,
                appearance_temperature=self.appearance_temperature,
                prior_temperature=self.prior_temperature,
            )
            if self.enable_prior_assignment:
                scores = appearance_scores + assignment_strength * prior_scores
            else:
                scores = appearance_scores
                assignment_strength = scores.new_zeros(())
            if self.enable_consensus_dustbin:
                dustbin_scores = self.dust_bin + consensus_strength * (
                    1.0 - reliability
                )
            else:
                dustbin_scores = self.dust_bin.expand_as(reliability)
                consensus_strength = scores.new_zeros(())

        log_assignment = get_prior_matching_probs(
            scores,
            dustbin_scores,
            num_iters=self.sinkhorn_iters,
        )
        assignment = log_assignment.exp()
        cluster_assignment = assignment[:, :-1]
        local_descriptor = torch.einsum(
            "bcn,bkn->bck",
            local_features,
            cluster_assignment,
        )
        local_descriptor = F.normalize(local_descriptor, p=2, dim=1).flatten(1)
        global_descriptor = self.token_features(cls_token)
        descriptor = torch.cat(
            [
                F.normalize(global_descriptor, p=2, dim=-1),
                local_descriptor,
            ],
            dim=-1,
        )
        descriptor = F.normalize(descriptor, p=2, dim=-1)

        if not return_diagnostics:
            return descriptor
        base_dustbin_scores = self.dust_bin.expand_as(reliability)
        base_assignment = get_prior_matching_probs(
            appearance_scores,
            base_dustbin_scores,
            num_iters=self.sinkhorn_iters,
        ).exp()
        return {
            "descriptor": descriptor,
            "appearance_scores": appearance_scores,
            "prior_scores": prior_scores,
            "appearance_assignment": appearance_assignment,
            "prior_assignment": prior_assignment,
            "prior_consensus": consensus,
            "prior_confidence": prior_confidence,
            "prior_reliability": reliability,
            "dustbin_probability": assignment[:, -1],
            "transport_assignment": cluster_assignment,
            "base_dustbin_probability": base_assignment[:, -1],
            "base_transport_assignment": base_assignment[:, :-1],
            "assignment_strength": assignment_strength,
            "consensus_strength": consensus_strength,
        }


class PriorResponseTransportAggregator(nn.Module):
    """Use spatial IVP injection responses to calibrate local transport.

    The appearance path is parameter-compatible with the balanced transport
    baseline. Cross-attention deltas provide patch-wise prior evidence that
    directly reweights post-Sinkhorn patch contributions in FP32. A
    zero-initialized residual gate exactly recovers the appearance-only path.
    """

    def __init__(
        self,
        num_channels=1024,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        dropout=0.3,
        sinkhorn_iters=3,
        response_layers=(20, 22),
        response_hidden_dim=16,
        max_response_strength=2.0,
        initial_response_strength=0.0,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.num_clusters = int(num_clusters)
        self.cluster_dim = int(cluster_dim)
        self.token_dim = int(token_dim)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.response_layers = tuple(int(index) for index in response_layers)
        self.max_response_strength = float(max_response_strength)
        if not self.response_layers:
            raise ValueError("response_layers must contain at least one layer")
        if response_hidden_dim <= 0:
            raise ValueError("response_hidden_dim must be positive")
        if self.max_response_strength <= 0:
            raise ValueError("max_response_strength must be positive")
        initial_response_strength = float(initial_response_strength)
        if abs(initial_response_strength) >= self.max_response_strength:
            raise ValueError(
                "abs(initial_response_strength) must be smaller than "
                "max_response_strength"
            )

        local_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        score_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim),
        )
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            local_dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1),
        )
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            score_dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

        self.response_layer_logits = nn.Parameter(torch.zeros(len(self.response_layers)))
        # Do not let the auxiliary branch consume the RNG stream that controls
        # the baseline's data augmentation and dropout trajectory.
        with torch.random.fork_rng(devices=[]):
            self.response_reliability = nn.Sequential(
                nn.Linear(2, response_hidden_dim),
                nn.GELU(),
                nn.Linear(response_hidden_dim, 1),
            )
        initial_gate = math.atanh(
            initial_response_strength / self.max_response_strength
        )
        self.response_gate = nn.Parameter(torch.tensor(initial_gate))

    @staticmethod
    def _normalize_patch_signal(signal):
        mean = signal.mean(dim=1, keepdim=True)
        variance = (signal - mean).square().mean(dim=1, keepdim=True)
        return (signal - mean) * torch.rsqrt(variance + 1e-6)

    def response_features(self, feature_map, patch_deltas):
        if patch_deltas is None:
            raise ValueError("patch_deltas are required for response aggregation")
        missing = [index for index in self.response_layers if index not in patch_deltas]
        if missing:
            raise KeyError(f"missing response deltas for layers: {missing}")

        batch_size, channels, height, width = feature_map.shape
        patch_count = height * width
        feature_energy = (
            feature_map.detach()
            .flatten(2)
            .float()
            .square()
            .mean(dim=1)
            .add(1e-8)
            .sqrt()
        )
        relative_energies = []
        normalized_deltas = []
        for index in self.response_layers:
            delta = patch_deltas[index]
            expected = (batch_size, patch_count, channels)
            if delta.shape != expected:
                raise ValueError(
                    f"response delta {index} must have shape {expected}, "
                    f"got {tuple(delta.shape)}"
                )
            delta_float = delta.detach().float()
            delta_energy = delta_float.square().mean(dim=-1).add(1e-8).sqrt()
            relative_energies.append(delta_energy / feature_energy.clamp_min(1e-6))
            normalized_deltas.append(F.normalize(delta_float, p=2, dim=-1))

        layer_weights = self.response_layer_logits.float().softmax(dim=0)
        energy_stack = torch.stack(relative_energies, dim=1)
        response_magnitude = torch.einsum(
            "l,bln->bn",
            layer_weights,
            energy_stack,
        )
        normalized_magnitude = self._normalize_patch_signal(
            response_magnitude.log1p()
        )

        pairwise_consistency = []
        for left in range(len(normalized_deltas)):
            for right in range(left + 1, len(normalized_deltas)):
                pairwise_consistency.append(
                    (normalized_deltas[left] * normalized_deltas[right]).sum(dim=-1)
                )
        if pairwise_consistency:
            response_consistency = torch.stack(
                pairwise_consistency,
                dim=1,
            ).mean(dim=1)
            response_consistency = (response_consistency + 1.0) * 0.5
        else:
            response_consistency = torch.ones_like(response_magnitude)

        reliability_inputs = torch.stack(
            [normalized_magnitude, response_consistency],
            dim=-1,
        )
        reliability = self.response_reliability(reliability_inputs).squeeze(-1)
        reliability = reliability.sigmoid()
        return {
            "response_magnitude": response_magnitude.to(feature_map.dtype),
            "response_consistency": response_consistency.to(feature_map.dtype),
            "response_reliability": reliability.to(feature_map.dtype),
            "response_layer_weights": layer_weights.to(feature_map.dtype),
        }

    def forward(self, inputs, patch_deltas=None, return_diagnostics=False):
        feature_map, cls_token = inputs
        local_features = self.cluster_features(feature_map).flatten(2)
        appearance_scores = self.score(feature_map).flatten(2)
        response = self.response_features(feature_map, patch_deltas)
        reliability = response["response_reliability"]
        response_strength = (
            self.max_response_strength * torch.tanh(self.response_gate)
        )
        response_source = response.get("response_logits", reliability)
        response_signal = self._normalize_patch_signal(
            response_source.float()
        ).clamp(min=-3.0, max=3.0)
        response_patch_weight = torch.exp(
            response_strength.float() * response_signal
        )

        assignment = get_prior_matching_probs(
            appearance_scores,
            self.dust_bin.expand_as(reliability),
            num_iters=self.sinkhorn_iters,
        ).exp()
        base_cluster_assignment = assignment[:, :-1]
        cluster_assignment = (
            base_cluster_assignment.float()
            * response_patch_weight.unsqueeze(1)
        )
        expanded_assignment = cluster_assignment.unsqueeze(1).repeat(
            1,
            self.cluster_dim,
            1,
            1,
        )
        expanded_features = local_features.unsqueeze(2).repeat(
            1,
            1,
            self.num_clusters,
            1,
        )
        local_descriptor = F.normalize(
            (expanded_features * expanded_assignment).sum(dim=-1),
            p=2,
            dim=1,
        ).flatten(1)
        global_descriptor = self.token_features(cls_token)
        descriptor = torch.cat(
            [
                F.normalize(global_descriptor, p=2, dim=-1),
                local_descriptor,
            ],
            dim=-1,
        )
        descriptor = F.normalize(descriptor, p=2, dim=-1)

        if not return_diagnostics:
            return descriptor
        base_expanded_assignment = base_cluster_assignment.unsqueeze(1).repeat(
            1,
            self.cluster_dim,
            1,
            1,
        )
        base_local_descriptor = F.normalize(
            (expanded_features * base_expanded_assignment).sum(dim=-1),
            p=2,
            dim=1,
        ).flatten(1)
        base_descriptor = F.normalize(
            torch.cat(
                [
                    F.normalize(global_descriptor, p=2, dim=-1),
                    base_local_descriptor,
                ],
                dim=-1,
            ),
            p=2,
            dim=-1,
        )
        diagnostics = {
            "descriptor": descriptor,
            "base_descriptor": base_descriptor,
            "appearance_scores": appearance_scores,
            "response_magnitude": response["response_magnitude"],
            "response_consistency": response["response_consistency"],
            "response_reliability": reliability,
            "response_layer_weights": response["response_layer_weights"],
            "response_strength": response_strength,
            "response_signal": response_signal,
            "response_patch_weight": response_patch_weight,
            "dustbin_probability": assignment[:, -1],
            "transport_assignment": cluster_assignment,
            "base_dustbin_probability": assignment[:, -1],
            "base_transport_assignment": base_cluster_assignment,
        }
        if "response_logits" in response:
            diagnostics["response_logits"] = response["response_logits"]
        return diagnostics


class PriorUtilityTransportAggregator(PriorResponseTransportAggregator):
    """Predict VLM-sensitive patch utility from inference-time Stage-2 features.

    Unlike the scalar response probe, this head retains low-rank directional
    information from each selected IVP delta and the corresponding appearance
    token. It can therefore imitate a training-only teacher utility map without
    requiring VLM features during inference.
    """

    def __init__(
        self,
        num_channels=1024,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        dropout=0.3,
        sinkhorn_iters=3,
        response_layers=(20, 22),
        response_projection_dim=16,
        response_hidden_dim=64,
        max_response_strength=2.0,
        initial_response_strength=0.0,
    ):
        super().__init__(
            num_channels=num_channels,
            num_clusters=num_clusters,
            cluster_dim=cluster_dim,
            token_dim=token_dim,
            dropout=dropout,
            sinkhorn_iters=sinkhorn_iters,
            response_layers=response_layers,
            response_hidden_dim=response_hidden_dim,
            max_response_strength=max_response_strength,
            initial_response_strength=initial_response_strength,
        )
        self.response_projection_dim = int(response_projection_dim)
        if self.response_projection_dim <= 0:
            raise ValueError("response_projection_dim must be positive")

        with torch.random.fork_rng(devices=[]):
            self.response_delta_projections = nn.ModuleDict(
                {
                    str(index): nn.Linear(
                        self.num_channels,
                        self.response_projection_dim,
                        bias=False,
                    )
                    for index in self.response_layers
                }
            )
            self.response_appearance_projection = nn.Linear(
                self.num_channels,
                self.response_projection_dim,
                bias=False,
            )
            pair_count = (
                len(self.response_layers) * (len(self.response_layers) - 1) // 2
            )
            input_dim = (
                (len(self.response_layers) + 1) * self.response_projection_dim
                + len(self.response_layers)
                + pair_count
            )
            self.response_reliability = nn.Sequential(
                nn.Linear(input_dim, response_hidden_dim),
                nn.GELU(),
                nn.Linear(response_hidden_dim, response_hidden_dim),
                nn.GELU(),
                nn.Linear(response_hidden_dim, 1),
            )

    def response_features(self, feature_map, patch_deltas):
        if patch_deltas is None:
            raise ValueError("patch_deltas are required for utility aggregation")
        missing = [index for index in self.response_layers if index not in patch_deltas]
        if missing:
            raise KeyError(f"missing response deltas for layers: {missing}")

        batch_size, channels, height, width = feature_map.shape
        patch_count = height * width
        appearance = (
            feature_map.detach()
            .flatten(2)
            .transpose(1, 2)
            .float()
        )
        feature_energy = (
            appearance.square().mean(dim=-1).add(1e-8).sqrt()
        )
        projected_features = [
            F.normalize(
                self.response_appearance_projection(appearance),
                p=2,
                dim=-1,
            )
        ]
        layer_weights = self.response_layer_logits.float().softmax(dim=0)
        relative_energies = []
        normalized_deltas = []
        for layer_position, index in enumerate(self.response_layers):
            delta = patch_deltas[index]
            expected = (batch_size, patch_count, channels)
            if delta.shape != expected:
                raise ValueError(
                    f"response delta {index} must have shape {expected}, "
                    f"got {tuple(delta.shape)}"
                )
            delta_float = delta.detach().float()
            delta_energy = delta_float.square().mean(dim=-1).add(1e-8).sqrt()
            relative_energy = delta_energy / feature_energy.clamp_min(1e-6)
            relative_energies.append(relative_energy)
            normalized_delta = F.normalize(delta_float, p=2, dim=-1)
            normalized_deltas.append(normalized_delta)
            projected_features.append(
                F.normalize(
                    self.response_delta_projections[str(index)](delta_float),
                    p=2,
                    dim=-1,
                )
                * layer_weights[layer_position]
            )

        energy_stack = torch.stack(relative_energies, dim=1)
        response_magnitude = torch.einsum(
            "l,bln->bn",
            layer_weights,
            energy_stack,
        )
        scalar_features = [
            (
                self._normalize_patch_signal(energy.log1p())
                * layer_weights[layer_position]
            ).unsqueeze(-1)
            for layer_position, energy in enumerate(relative_energies)
        ]
        pairwise_consistency = []
        for left in range(len(normalized_deltas)):
            for right in range(left + 1, len(normalized_deltas)):
                consistency = (
                    normalized_deltas[left] * normalized_deltas[right]
                ).sum(dim=-1)
                pairwise_consistency.append(consistency)
                scalar_features.append(consistency.unsqueeze(-1))
        if pairwise_consistency:
            response_consistency = torch.stack(
                pairwise_consistency,
                dim=1,
            ).mean(dim=1)
            response_consistency = (response_consistency + 1.0) * 0.5
        else:
            response_consistency = torch.ones_like(response_magnitude)

        utility_inputs = torch.cat(projected_features + scalar_features, dim=-1)
        response_logits = self.response_reliability(utility_inputs).squeeze(-1)
        reliability = response_logits.sigmoid()
        return {
            "response_magnitude": response_magnitude.to(feature_map.dtype),
            "response_consistency": response_consistency.to(feature_map.dtype),
            "response_reliability": reliability.to(feature_map.dtype),
            "response_logits": response_logits,
            "response_layer_weights": layer_weights.to(feature_map.dtype),
        }
