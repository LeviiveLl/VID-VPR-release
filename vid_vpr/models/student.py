import logging
import math
import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from vid_vpr.backbones.dinov2.vision_transformer import vit_large
from vid_vpr.models.cross_attention import VLMCrossAttention
from vid_vpr.models.pooling import ChannelWiseGeM, Flatten, GeM, L2Norm


class ResidualAggregationAdapter(nn.Module):
    """Expand token processing capacity without changing the descriptor interface."""

    def __init__(self, dim=1024, hidden_dim=2048):
        super().__init__()
        if hidden_dim <= dim:
            raise ValueError(
                f"aggregation adapter hidden_dim must exceed {dim}, got {hidden_dim}"
            )
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.down = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, tokens):
        residual = self.down(self.activation(self.up(self.norm(tokens))))
        return tokens + residual


class WideTokenExpansion(nn.Module):
    """Widen patch channels while preserving the legacy path at initialization."""

    def __init__(self, input_dim=1024, output_dim=2048, hidden_dim=4096):
        super().__init__()
        if output_dim % input_dim:
            raise ValueError(
                f"wide output_dim must be a multiple of {input_dim}, got {output_dim}"
            )
        self.repeat_factor = output_dim // input_dim
        self.refinement = ResidualAggregationAdapter(
            dim=output_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, tokens):
        repeats = [1] * tokens.ndim
        repeats[-1] = self.repeat_factor
        expanded = tokens.repeat(*repeats)
        return self.refinement(expanded)


class GeMAggregationExpert(nn.Module):
    """An independently trainable high-capacity GeM descriptor branch."""

    def __init__(self, token_dim=1024, hidden_dim=2048, output_dim=4096):
        super().__init__()
        self.adapter = ResidualAggregationAdapter(
            dim=token_dim,
            hidden_dim=hidden_dim,
        )
        self.aggregation = nn.Sequential(L2Norm(), GeM(), Flatten())
        self.projection = nn.Linear(token_dim, output_dim)

    def forward(self, patches):
        patches = self.adapter(patches).permute(0, 3, 1, 2)
        return self.projection(self.aggregation(patches))


class SyntheticPriorRouter(nn.Module):
    """Generate depth-conditioned synthetic prior tokens from visual features."""

    def __init__(
        self,
        visual_dim=1024,
        vlm_dim=2048,
        bank_size=256,
        router_dim=512,
        top_k=8,
        ablation_mode="full",
        max_layer_index=24,
        dynamic_residual=False,
    ):
        super().__init__()
        if top_k <= 0 or top_k > bank_size:
            raise ValueError(f"top_k must be in [1, {bank_size}], got {top_k}")
        self.visual_dim = visual_dim
        self.vlm_dim = vlm_dim
        self.bank_size = bank_size
        self.router_dim = router_dim
        self.top_k = top_k
        self.ablation_mode = ablation_mode
        self.dynamic_residual = bool(dynamic_residual)
        layer_count = max(max_layer_index, 1)

        self.memory_bank = nn.Parameter(torch.randn(bank_size, vlm_dim) * 0.02)
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, router_dim),
            nn.GELU(),
            nn.Linear(router_dim, router_dim),
        )
        self.memory_key_proj = nn.Linear(vlm_dim, router_dim)
        self.layer_embedding = nn.Embedding(layer_count, visual_dim)
        self.static_logits = nn.Parameter(torch.zeros(1, bank_size))
        self.direct_generator = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, router_dim),
            nn.GELU(),
            nn.Linear(router_dim, top_k * vlm_dim),
        )
        if self.dynamic_residual:
            self.dynamic_visual_proj = nn.Sequential(
                nn.LayerNorm(visual_dim),
                nn.Linear(visual_dim, router_dim),
                nn.GELU(),
                nn.Linear(router_dim, router_dim),
            )
            self.dynamic_memory_key_proj = nn.Linear(vlm_dim, router_dim)
            self.dynamic_layer_embedding = nn.Embedding(
                layer_count,
                router_dim,
            )
            self.dynamic_slot_queries = nn.Parameter(
                torch.randn(top_k, router_dim) * 0.02
            )
            self.dynamic_log_temperature = nn.Parameter(torch.zeros(layer_count))
            self.dynamic_residual_gate = nn.Parameter(torch.zeros(layer_count))
            nn.init.zeros_(self.dynamic_layer_embedding.weight)

    @torch.no_grad()
    def initialize_dynamic_from_legacy(self):
        if not self.dynamic_residual:
            return
        self.dynamic_visual_proj.load_state_dict(self.visual_proj.state_dict())
        self.dynamic_memory_key_proj.load_state_dict(self.memory_key_proj.state_dict())

    def _dynamic_route(self, visual_inputs, layer_index):
        visual_summary = self._summarize(visual_inputs, layer_index=None)
        visual_query = self.dynamic_visual_proj(visual_summary)
        index = min(
            max(int(layer_index or 0), 0),
            self.dynamic_layer_embedding.num_embeddings - 1,
        )
        layer_query = self.dynamic_layer_embedding.weight[index].to(
            dtype=visual_query.dtype,
            device=visual_query.device,
        )
        queries = (
            visual_query[:, None, :]
            + self.dynamic_slot_queries[None].to(dtype=visual_query.dtype)
            + layer_query[None, None, :]
        )
        queries = F.normalize(queries, p=2, dim=-1)
        memory_keys = F.normalize(
            self.dynamic_memory_key_proj(self.memory_bank),
            p=2,
            dim=-1,
        )
        temperature = self.dynamic_log_temperature[index].exp().clamp(0.1, 5.0)
        scores = torch.matmul(queries, memory_keys.T) / temperature
        probabilities = scores.softmax(dim=-1)
        dynamic_tokens = torch.matmul(probabilities, self.memory_bank)
        dynamic_tokens = dynamic_tokens / self.top_k
        return dynamic_tokens, probabilities, visual_summary, temperature

    @staticmethod
    def _as_list(visual_inputs):
        if isinstance(visual_inputs, torch.Tensor):
            return [visual_inputs]
        return list(visual_inputs)

    def _summarize(self, visual_inputs, layer_index=None):
        summaries = []
        for item in self._as_list(visual_inputs):
            summaries.append(item.mean(dim=1) if item.dim() == 3 else item)
        summary = summaries[0] if len(summaries) == 1 else torch.stack(summaries).mean(dim=0)
        if layer_index is not None:
            index = min(max(int(layer_index), 0), self.layer_embedding.num_embeddings - 1)
            summary = summary + self.layer_embedding.weight[index].to(
                dtype=summary.dtype,
                device=summary.device,
            )
        return summary

    def _usage_stats(self, weights, indices=None):
        eps = torch.finfo(weights.dtype).eps
        entropy = -(weights.clamp_min(eps) * weights.clamp_min(eps).log()).sum(dim=-1).mean()
        if indices is None:
            usage = weights.detach().mean(dim=0)
            return {
                "router_entropy": entropy,
                "router_topk_unique": (usage > 0).sum().to(dtype=weights.dtype),
                "bank_usage_max_frac": usage.max(),
            }
        counts = torch.bincount(
            indices.detach().reshape(-1),
            minlength=self.bank_size,
        ).to(dtype=weights.dtype, device=weights.device)
        return {
            "router_entropy": entropy,
            "router_topk_unique": (counts > 0).sum().to(dtype=weights.dtype),
            "bank_usage_max_frac": counts.max() / counts.sum().clamp_min(1.0),
        }

    @torch.no_grad()
    def initialize_memory_bank(self, vlm_tokens):
        expected = (None, self.vlm_dim)
        if vlm_tokens.dim() != 2 or vlm_tokens.shape[-1] != self.vlm_dim:
            raise ValueError(
                f"expected VLM tokens shaped {expected}, got {tuple(vlm_tokens.shape)}"
            )
        if vlm_tokens.shape[0] == 0:
            raise ValueError("cannot initialize the memory bank from zero tokens")
        if vlm_tokens.shape[0] < self.bank_size:
            repeats = math.ceil(self.bank_size / vlm_tokens.shape[0])
            vlm_tokens = vlm_tokens.repeat(repeats, 1)
        self.memory_bank.copy_(
            vlm_tokens[: self.bank_size].to(
                device=self.memory_bank.device,
                dtype=self.memory_bank.dtype,
            )
        )

    def forward(self, visual_inputs, layer_index=None, return_details=False):
        summary = self._summarize(visual_inputs, layer_index)
        if self.ablation_mode == "no_vsyn":
            zero = summary.new_tensor(0.0)
            stats = {
                "router_entropy": zero,
                "router_topk_unique": zero,
                "bank_usage_max_frac": zero,
            }
            if return_details:
                return None, stats, {
                    "visual_summary": summary,
                    "routing_weights": None,
                    "topk_indices": None,
                }
            return None, stats
        if self.ablation_mode == "router_without_memory":
            tokens = self.direct_generator(summary).view(
                summary.shape[0],
                self.top_k,
                self.vlm_dim,
            )
            weights = tokens.new_full(
                (tokens.shape[0], self.top_k),
                1.0 / self.top_k,
            )
            stats = self._usage_stats(weights)
            if return_details:
                return tokens, stats, {
                    "visual_summary": summary,
                    "routing_weights": weights,
                    "topk_indices": None,
                }
            return tokens, stats

        if self.ablation_mode == "static_vsyn":
            scores = self.static_logits.expand(summary.shape[0], -1)
        elif self.ablation_mode == "full":
            memory_keys = F.normalize(self.memory_key_proj(self.memory_bank), p=2, dim=-1)
            visual_query = F.normalize(self.visual_proj(summary), p=2, dim=-1)
            scores = torch.matmul(visual_query, memory_keys.T) / math.sqrt(self.router_dim)
        else:
            raise ValueError(f"unknown IVP ablation mode: {self.ablation_mode}")

        topk_scores, topk_indices = scores.topk(self.top_k, dim=-1)
        weights = topk_scores.softmax(dim=-1)
        selected_memory = self.memory_bank[topk_indices]
        tokens = selected_memory * weights.unsqueeze(-1)
        stats = self._usage_stats(weights, topk_indices)
        dynamic_state = {}
        if self.dynamic_residual:
            dynamic_tokens, probabilities, visual_summary, temperature = (
                self._dynamic_route(visual_inputs, layer_index)
            )
            dynamic_layer_index = min(
                max(int(layer_index or 0), 0),
                self.dynamic_residual_gate.numel() - 1,
            )
            dynamic_strength = torch.tanh(
                self.dynamic_residual_gate[dynamic_layer_index]
            )
            tokens = tokens + dynamic_strength * dynamic_tokens
            dynamic_entropy = -(
                probabilities.clamp_min(1e-8)
                * probabilities.clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
            stats = dict(stats)
            stats.update(
                {
                    "dynamic_router_entropy": dynamic_entropy,
                    "dynamic_router_strength": dynamic_strength,
                    "dynamic_router_temperature": temperature,
                }
            )
            dynamic_state = {
                "dynamic_probabilities": probabilities,
                "dynamic_tokens": dynamic_tokens,
                "dynamic_visual_summary": visual_summary,
                "dynamic_strength": dynamic_strength,
                "dynamic_temperature": temperature,
                "dynamic_layer_index": dynamic_layer_index,
            }
        if return_details:
            state = {
                "visual_summary": summary,
                "routing_weights": weights,
                "topk_indices": topk_indices,
            }
            state.update(dynamic_state)
            return tokens, stats, state
        return tokens, stats


class ImageOnlyStudent(nn.Module):
    """VLM-free DINOv2-Large student with a synthetic-prior router."""

    def __init__(
        self,
        output_dim=4096,
        vlm_dim=2048,
        crossattn_heads=16,
        crossattn_every_n=2,
        crossattn_layer_mode="late_stride",
        bank_size=256,
        router_dim=512,
        top_k=8,
        router_layers=(8, 12, 16, 20),
        ablation_mode="full",
        foundation_model_path=None,
        teacher_checkpoint=None,
        strict_no_vlm=True,
        dynamic_router_residual=False,
        aggregation_adapter_dim=0,
        aggregation_num_branches=1,
        aggregation_branch_dim=None,
        aggregation_branch_noise=0.001,
        aggregation_wide_dim=0,
        aggregation_wide_hidden_dim=4096,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.vlm_dim = vlm_dim
        self.strict_no_vlm = strict_no_vlm
        self.crossattn_layer_mode = crossattn_layer_mode
        self.router_layer_indices = tuple(int(value) for value in router_layers)
        self.ablation_mode = ablation_mode
        self.aggregation_adapter_dim = int(aggregation_adapter_dim)
        self.aggregation_wide_dim = int(aggregation_wide_dim)
        self.aggregation_wide_hidden_dim = int(aggregation_wide_hidden_dim)
        self.aggregation_num_branches = int(aggregation_num_branches)
        if self.aggregation_num_branches <= 0:
            raise ValueError("aggregation_num_branches must be positive")
        if self.aggregation_wide_dim > 0:
            if self.aggregation_num_branches != 1:
                raise ValueError("Wide GeM supports exactly one aggregation branch")
            aggregation_branch_dim = output_dim
        elif aggregation_branch_dim is None:
            if output_dim % self.aggregation_num_branches:
                raise ValueError(
                    "output_dim must be divisible by aggregation_num_branches"
                )
            aggregation_branch_dim = output_dim // self.aggregation_num_branches
        self.aggregation_branch_dim = int(aggregation_branch_dim)
        if self.aggregation_branch_dim * self.aggregation_num_branches != output_dim:
            raise ValueError(
                "output_dim must equal aggregation_branch_dim * "
                "aggregation_num_branches"
            )
        self.aggregation_branch_noise = float(aggregation_branch_noise)

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
        self.router = SyntheticPriorRouter(
            visual_dim=1024,
            vlm_dim=vlm_dim,
            bank_size=bank_size,
            router_dim=router_dim,
            top_k=top_k,
            ablation_mode=ablation_mode,
            max_layer_index=len(self.backbone.blocks),
            dynamic_residual=dynamic_router_residual,
        )
        self.linear1 = nn.Linear(1024, 1024)
        if self.aggregation_wide_dim > 0:
            self.aggregation_adapter = WideTokenExpansion(
                input_dim=1024,
                output_dim=self.aggregation_wide_dim,
                hidden_dim=self.aggregation_wide_hidden_dim,
            )
            self.aggregation = nn.Sequential(
                L2Norm(),
                ChannelWiseGeM(self.aggregation_wide_dim),
                Flatten(),
            )
            self.linear2 = nn.Linear(self.aggregation_wide_dim, output_dim)
        else:
            self.aggregation_adapter = (
                ResidualAggregationAdapter(
                    dim=1024,
                    hidden_dim=self.aggregation_adapter_dim,
                )
                if self.aggregation_adapter_dim > 0
                else nn.Identity()
            )
            self.aggregation = nn.Sequential(L2Norm(), GeM(), Flatten())
            self.linear2 = nn.Linear(1024, self.aggregation_branch_dim)
        self.aggregation_branches = nn.ModuleList(
            [
                GeMAggregationExpert(
                    token_dim=1024,
                    hidden_dim=self.aggregation_adapter_dim,
                    output_dim=self.aggregation_branch_dim,
                )
                for _ in range(self.aggregation_num_branches - 1)
            ]
        )

        if foundation_model_path is not None:
            self.load_foundation_weights(foundation_model_path)
        if teacher_checkpoint is not None:
            self.load_teacher_weights(teacher_checkpoint)

    @staticmethod
    def _checkpoint_state_dict(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            return checkpoint
        state_dict = checkpoint.get("student_state_dict")
        if state_dict is None:
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        if state_dict and next(iter(state_dict)).startswith("module."):
            state_dict = OrderedDict(
                (key.removeprefix("module."), value)
                for key, value in state_dict.items()
            )
        return state_dict

    @staticmethod
    def _extract_crossattn_indices(state_dict):
        pattern = re.compile(r"backbone\.vlm_crossattns\.(\d+)\.")
        return sorted(
            {
                int(match.group(1))
                for key in state_dict
                if (match := pattern.search(key))
            }
        )

    def _remap_legacy_crossattn_keys(self, state_dict):
        model_state = self.state_dict()
        source_indices = self._extract_crossattn_indices(state_dict)
        target_indices = self._extract_crossattn_indices(model_state)
        if source_indices != list(range(0, 16, 2)) or target_indices != list(range(8, 24, 2)):
            return state_dict
        remapped = OrderedDict()
        pattern = re.compile(r"(backbone\.vlm_crossattns\.)(\d+)(\..+)")
        for key, value in state_dict.items():
            match = pattern.match(key)
            new_key = key
            if match:
                candidate = f"{match.group(1)}{int(match.group(2)) + 8}{match.group(3)}"
                if candidate in model_state and model_state[candidate].shape == value.shape:
                    new_key = candidate
            remapped[new_key] = value
        return remapped

    def _load_compatible_state_dict(self, state_dict, source_name):
        model_state = self.state_dict()
        compatible = OrderedDict(
            (key, value)
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        )
        result = self.load_state_dict(compatible, strict=False)
        logging.info(
            "Loaded %d compatible tensors from %s; missing=%d, unexpected=%d",
            len(compatible),
            source_name,
            len(result.missing_keys),
            len(result.unexpected_keys),
        )
        return result

    def load_foundation_weights(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        prefixed = OrderedDict(
            (f"backbone.{key}", value)
            for key, value in state_dict.items()
        )
        return self._load_compatible_state_dict(prefixed, checkpoint_path)

    def load_teacher_weights(self, checkpoint_path):
        state_dict = self._checkpoint_state_dict(checkpoint_path)
        state_dict = self._remap_legacy_crossattn_keys(state_dict)
        return self._load_compatible_state_dict(state_dict, checkpoint_path)

    def load_student_checkpoint(self, checkpoint_path, strict=True):
        state_dict = self._checkpoint_state_dict(checkpoint_path)
        state_dict = self.prepare_checkpoint_state_dict(state_dict)
        return self.load_state_dict(state_dict, strict=strict)

    def prepare_checkpoint_state_dict(self, state_dict):
        state_dict = OrderedDict(state_dict)
        if self.aggregation_wide_dim <= 0:
            return state_dict
        old_weight = state_dict.get("linear2.weight")
        old_bias = state_dict.get("linear2.bias")
        old_p = state_dict.get("aggregation.1.p")
        if (
            old_weight is None
            or old_bias is None
            or old_p is None
            or old_weight.shape == self.linear2.weight.shape
        ):
            return state_dict
        if self.aggregation_wide_dim % old_weight.shape[1]:
            raise ValueError("Wide GeM input dimension is incompatible with checkpoint")
        if self.output_dim % old_weight.shape[0]:
            raise ValueError("Wide GeM output dimension is incompatible with checkpoint")
        input_repeats = self.aggregation_wide_dim // old_weight.shape[1]
        output_repeats = self.output_dim // old_weight.shape[0]
        expanded_weight = old_weight.repeat(output_repeats, input_repeats)
        expanded_weight = expanded_weight / math.sqrt(float(input_repeats))
        if self.aggregation_branch_noise > 0:
            noise_scale = (
                old_weight.detach().float().std().to(dtype=old_weight.dtype)
                * self.aggregation_branch_noise
            )
            expanded_weight = expanded_weight + (
                torch.randn_like(expanded_weight) * noise_scale
            )
        state_dict["linear2.weight"] = expanded_weight
        state_dict["linear2.bias"] = old_bias.repeat(output_repeats)
        state_dict["aggregation.1.p"] = old_p.reshape(-1)[0].repeat(
            self.aggregation_wide_dim
        )
        logging.info(
            "Migrated legacy GeM head %dx%d to Wide GeM %dx%d",
            old_weight.shape[1],
            old_weight.shape[0],
            self.aggregation_wide_dim,
            self.output_dim,
        )
        return state_dict

    @torch.no_grad()
    def initialize_aggregation_branches_from_base(self):
        if not self.aggregation_branches:
            return
        weight_scale = self.linear2.weight.detach().float().std().to(
            dtype=self.linear2.weight.dtype
        )
        for index, branch in enumerate(self.aggregation_branches, start=1):
            branch.aggregation[1].p.copy_(self.aggregation[1].p)
            branch.projection.weight.copy_(self.linear2.weight)
            branch.projection.bias.copy_(self.linear2.bias)
            if self.aggregation_branch_noise > 0:
                noise_scale = (
                    weight_scale
                    * self.aggregation_branch_noise
                    * float(index)
                )
                branch.projection.weight.add_(
                    torch.randn_like(branch.projection.weight) * noise_scale
                )

    def set_train_stage(
        self,
        stage,
        unfreeze_last_n_layers=0,
        train_aggregation_head=False,
    ):
        if stage not in {1, 2}:
            raise ValueError(f"supported training stages are 1 and 2, got {stage}")
        unfreeze_last_n_layers = int(unfreeze_last_n_layers)
        if unfreeze_last_n_layers < 0:
            raise ValueError("unfreeze_last_n_layers must be non-negative")
        for parameter in self.parameters():
            parameter.requires_grad = False
        if self.ablation_mode != "no_vsyn":
            for parameter in self.router.parameters():
                parameter.requires_grad = True
        if stage == 2:
            for parameter in self.backbone.vlm_projector.parameters():
                parameter.requires_grad = True
            for module in self.backbone.vlm_crossattns:
                if isinstance(module, VLMCrossAttention):
                    for parameter in module.parameters():
                        parameter.requires_grad = True
        if train_aggregation_head:
            for module in (
                self.linear1,
                self.aggregation_adapter,
                self.aggregation,
                self.linear2,
                self.aggregation_branches,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        if unfreeze_last_n_layers:
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad = True
            effective_last_index = len(self.backbone.blocks) - 1
            if self.ablation_mode != "no_vsyn":
                injection_indices = [
                    index
                    for index, module in enumerate(self.backbone.vlm_crossattns)
                    if isinstance(module, VLMCrossAttention)
                ]
                if injection_indices:
                    effective_last_index = max(injection_indices)
            start = max(effective_last_index - unfreeze_last_n_layers + 1, 0)
            for block in self.backbone.blocks[start : effective_last_index + 1]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _patch_tokens(tokens):
        return tokens[:, 1:]

    def _router_forward(
        self,
        visual_inputs,
        target_dtype,
        layer_index,
        return_state=False,
    ):
        router_output = self.router(
            visual_inputs,
            layer_index=layer_index,
            return_details=return_state,
        )
        if return_state:
            tokens, stats, state = router_output
        else:
            tokens, stats = router_output
            state = None
        if tokens is None:
            if return_state:
                return None, stats, state
            return None, stats
        projected = self.backbone.vlm_projector(
            tokens.to(dtype=self.backbone.vlm_projector.weight.dtype)
        )
        projected = projected.to(dtype=target_dtype)
        if state is not None:
            state = dict(state)
            state["synthetic_tokens"] = projected
            state["layer_index"] = int(layer_index)
            return projected, stats, state
        return projected, stats

    @staticmethod
    def _aggregate_router_stats(stats_list, device):
        if not stats_list:
            zero = torch.tensor(0.0, device=device)
            return {
                "router_entropy": zero,
                "router_topk_unique": zero,
                "bank_usage_max_frac": zero,
            }
        return {
            key: torch.stack([stats[key] for stats in stats_list]).mean()
            for key in stats_list[0]
        }

    def _aggregate_descriptor(self, features):
        normalized = self.backbone.norm(features)
        patch_tokens = normalized[:, 1:]
        batch_size, patch_count, dim = patch_tokens.shape
        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise ValueError(f"GeM requires a square patch grid, got {patch_count} tokens")
        patches = self.linear1(patch_tokens.view(batch_size, side, side, dim))
        base_patches = self.aggregation_adapter(patches).permute(0, 3, 1, 2)
        descriptor_parts = [self.linear2(self.aggregation(base_patches))]
        descriptor_parts.extend(
            branch(patches) for branch in self.aggregation_branches
        )
        descriptor = torch.cat(descriptor_parts, dim=-1)
        return F.normalize(descriptor, p=2, dim=-1)

    def _forward_plain_features(self, images):
        features = self.backbone.prepare_tokens_with_masks(images, None)
        for block in self.backbone.blocks:
            features = block(features)
        return features

    def _forward_plain_dino(self, images):
        return self._aggregate_descriptor(self._forward_plain_features(images))

    def _forward_ivp_features(
        self,
        images,
        return_xattn_deltas=False,
        xattn_delta_indices=None,
        return_router_state=False,
    ):
        x = self.backbone.prepare_tokens_with_masks(images, None)
        y = x.clone() if self.crossattn_layer_mode == "all_even" else None
        router_contexts = []
        router_stats = []
        xattn_deltas = {}
        final_router_state = None
        layer_router_states = {}

        for index, block in enumerate(self.backbone.blocks):
            x = block(x)
            if index in self.router_layer_indices:
                router_contexts.append(self._patch_tokens(x))
            if self.crossattn_layer_mode != "all_even" and index == 7:
                y = x.clone()
                continue
            if y is None:
                continue
            module = self.backbone.vlm_crossattns[index]
            if not isinstance(module, VLMCrossAttention):
                continue
            contexts = router_contexts if router_contexts else [self._patch_tokens(x)]
            router_output = self._router_forward(
                contexts,
                (y + x).dtype,
                index,
                return_state=return_router_state,
            )
            if return_router_state:
                synthetic_tokens, stats, router_state = router_output
            else:
                synthetic_tokens, stats = router_output
                router_state = None
            router_stats.append(stats)
            if router_state is not None:
                final_router_state = router_state
                layer_router_states[index] = router_state
            if synthetic_tokens is not None:
                delta = module(y + x, synthetic_tokens)
                y = y + delta
                if return_xattn_deltas and (
                    xattn_delta_indices is None or index in xattn_delta_indices
                ):
                    xattn_deltas[index] = delta

        if final_router_state is not None:
            final_router_state = dict(final_router_state)
            final_router_state["layer_states"] = layer_router_states
        return y if y is not None else x, router_stats, xattn_deltas, final_router_state

    def forward_feature_tokens(
        self,
        images,
        return_xattn_deltas=False,
        xattn_delta_indices=None,
        return_router_state=False,
    ):
        """Return the normalized token sequence immediately before descriptor pooling."""
        if self.ablation_mode == "no_vsyn":
            features = self.backbone.prepare_tokens_with_masks(images, None)
            for block in self.backbone.blocks:
                features = block(features)
            xattn_deltas = {}
            router_state = None
        else:
            features, _, xattn_deltas, router_state = self._forward_ivp_features(
                images,
                return_xattn_deltas=return_xattn_deltas,
                xattn_delta_indices=xattn_delta_indices,
                return_router_state=return_router_state,
            )
        normalized = self.backbone.norm(features)
        if return_router_state:
            return {
                "tokens": normalized,
                "router_state": router_state,
                "xattn_deltas": xattn_deltas,
            }
        if return_xattn_deltas:
            return normalized, xattn_deltas
        return normalized

    def forward(
        self,
        images,
        vl_embeds=None,
        vl_attention_mask=None,
        return_diagnostics=False,
        return_xattn_deltas=False,
        return_feature_tokens=False,
    ):
        if self.strict_no_vlm and (vl_embeds is not None or vl_attention_mask is not None):
            raise ValueError("ImageOnlyStudent cannot consume VLM inputs at inference")
        if self.ablation_mode == "no_vsyn":
            features = self._forward_plain_features(images)
            descriptor = self._aggregate_descriptor(features)
            if return_diagnostics or return_xattn_deltas or return_feature_tokens:
                output = {
                    "descriptor": descriptor,
                    "xattn_deltas": {},
                    "diagnostics": self._aggregate_router_stats([], descriptor.device),
                }
                if return_feature_tokens:
                    output["feature_tokens"] = self.backbone.norm(features)[:, 1:]
                return output
            return descriptor

        features, router_stats, xattn_deltas, _ = self._forward_ivp_features(
            images,
            return_xattn_deltas=return_xattn_deltas,
        )
        descriptor = self._aggregate_descriptor(features)
        if return_diagnostics or return_xattn_deltas or return_feature_tokens:
            output = {
                "descriptor": descriptor,
                "xattn_deltas": xattn_deltas,
                "diagnostics": self._aggregate_router_stats(
                    router_stats,
                    descriptor.device,
                ),
            }
            if return_feature_tokens:
                output["feature_tokens"] = self.backbone.norm(features)[:, 1:]
            return output
        return descriptor
