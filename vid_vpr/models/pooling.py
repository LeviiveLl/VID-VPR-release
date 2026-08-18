import torch
import torch.nn.functional as F
from torch import nn


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        pooled = F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1)),
        )
        return pooled.pow(1.0 / self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


class ChannelWiseGeM(nn.Module):
    def __init__(self, channels, p=3.0, eps=1e-6):
        super().__init__()
        self.channels = int(channels)
        self.p = nn.Parameter(torch.full((self.channels,), float(p)))
        self.eps = eps

    def forward(self, x):
        if x.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {x.shape[1]}"
            )
        p = self.p.to(dtype=x.dtype).view(1, self.channels, 1, 1)
        pooled = F.avg_pool2d(
            x.clamp(min=self.eps).pow(p),
            (x.size(-2), x.size(-1)),
        )
        return pooled.pow(1.0 / p)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(channels={self.channels}, "
            f"mean_p={self.p.detach().mean().item():.4f}, eps={self.eps})"
        )


class Flatten(nn.Module):
    def forward(self, x):
        if x.shape[2:] != (1, 1):
            raise ValueError(f"expected [B, C, 1, 1], got {tuple(x.shape)}")
        return x[:, :, 0, 0]
