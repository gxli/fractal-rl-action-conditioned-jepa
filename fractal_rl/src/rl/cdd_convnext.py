"""SA-JEPA-compatible CDD scale-aware ConvNeXt feature extractor.

Adapted from the MIT-licensed gxli/SA-JEPA CDDScaleAwareConvNeXtEncoder.  The
input is a precomputed full-image CDD cube sampled into local FOVs: B,S,H,W.
"""
import torch
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__(); self.norm = nn.LayerNorm(channels, eps=eps)
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GRN(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__(); self.gamma = nn.Parameter(torch.zeros(1, 1, 1, channels)); self.beta = nn.Parameter(torch.zeros(1, 1, 1, channels)); self.eps = eps
    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        return self.gamma * (x * gx / (gx.mean(dim=-1, keepdim=True) + self.eps)) + self.beta + x


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, expansion=4):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Sequential(nn.ReflectionPad2d(pad), nn.Conv2d(channels, channels, kernel_size, groups=channels))
        self.norm, self.pw1, self.act = nn.LayerNorm(channels), nn.Linear(channels, expansion * channels), nn.GELU()
        self.grn, self.pw2 = GRN(expansion * channels), nn.Linear(expansion * channels, channels)
        self.gamma = nn.Parameter(1e-6 * torch.ones(channels))
    def forward(self, x):
        residual = x; x = self.dw(x).permute(0, 2, 3, 1); x = self.pw2(self.grn(self.act(self.pw1(self.norm(x))))); return residual + (self.gamma * x).permute(0, 3, 1, 2)


class CDDScaleAwareConvNeXt(nn.Module):
    """Full-FOV CDD ConvNeXt point encoder.

    The complete cached CDD patch enters the SA-JEPA per-scale adapter, then a
    *single* depthwise ConvNeXt stage.  Only the center latent vector is
    returned; no dense atlas, crop, downsampling pyramid, or global pooling is
    retained for online RL.
    """
    def __init__(self, scales, hidden_channels=32, latent_channels=32, kernel_size=7, scale_features=8):
        super().__init__()
        self.scales = tuple(float(s) for s in scales); self.num_scales = len(self.scales)
        logs = torch.log(torch.tensor(self.scales)); logs = (logs - logs.mean()) / logs.std(unbiased=False).clamp_min(1e-6)
        self.register_buffer('scale_codes', logs.view(1, self.num_scales, 1, 1), persistent=False)
        self.adapter = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(3, scale_features, 3), LayerNorm2d(scale_features), nn.GELU(), nn.Conv2d(scale_features, scale_features, 1), LayerNorm2d(scale_features), nn.GELU())
        self.stem = nn.Sequential(nn.Conv2d(self.num_scales * scale_features, hidden_channels, 1), LayerNorm2d(hidden_channels), nn.GELU())
        self.block = ConvNeXtBlock(hidden_channels, kernel_size)
        self.head, self.norm = nn.Conv2d(hidden_channels, latent_channels, 1), LayerNorm2d(latent_channels)
    def forward(self, fields):
        if fields.ndim != 4 or fields.shape[1] != self.num_scales:
            raise ValueError(f'expected CDD fields B,{self.num_scales},H,W; got {tuple(fields.shape)}')
        b, s, h, w = fields.shape
        codes = self.scale_codes.expand(b, -1, h, w)
        x = torch.stack([fields, torch.zeros_like(fields), codes], dim=2).reshape(b * s, 3, h, w)
        x = self.adapter(x).reshape(b, s, -1, h, w).flatten(1, 2)
        x = self.norm(self.head(self.block(self.stem(x))))
        return x[:, :, x.shape[-2] // 2, x.shape[-1] // 2]
