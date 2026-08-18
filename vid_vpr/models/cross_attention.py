import torch
from torch import nn, Tensor

class VLMCrossAttention(nn.Module):
    """Inject VLM hidden states into visual tokens through a gated residual."""

    def __init__(self, dim, vlm_dim, num_heads=8, qkv_bias=True, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(vlm_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(vlm_dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(vlm_dim)

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: Tensor,
        vl_embeds: Tensor,
        vl_attention_mask: Tensor = None,
        return_delta: bool = False,
        return_attn: bool = False,
    ) -> Tensor:
        B, N, D = x.shape
        x_normed = self.norm_q(x)
        vl_normed = self.norm_kv(vl_embeds)

        q = self.q_proj(x_normed).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(vl_normed).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(vl_normed).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if vl_attention_mask is not None:
            attn_mask = vl_attention_mask[:, None, None, :].to(dtype=torch.bool, device=attn.device)
            attn = attn.masked_fill(~attn_mask, torch.finfo(attn.dtype).min)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.out_drop(self.out_proj(out))
        gated_out = self.gate.tanh() * out
        if return_delta or return_attn:
            outputs = {"delta": gated_out}
            if return_attn:
                outputs["attn"] = attn
            return outputs
        return gated_out
