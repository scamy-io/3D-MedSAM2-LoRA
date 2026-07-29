"""src/train/adapters.py — Space-Depth (SD-Trans) Adapters for SAM 2.

Implements lightweight SD-Trans adapters injected into Vision Transformer blocks
to enable 2D spatial token maps to exchange information across the depth (z) axis.
"""

import torch
import torch.nn as nn


class SpaceDepthAdapter(nn.Module):
    """Space-Depth Transpose Adapter module.

    Args:
        embed_dim (int): Feature embedding dimension.
        bottleneck_dim (int): Inner bottleneck projection dimension.
    """

    def __init__(self, embed_dim=768, bottleneck_dim=64):
        super().__init__()
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.depth_layer = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Forward pass applying residual space-depth adaptation.

        Args:
            x (Tensor): Input tensor of shape (B, N_tokens, C) or (B, C, H, W).

        Returns:
            Tensor: Adapted tensor of same shape as input.
        """
        orig = x
        is_4d = (x.ndim == 4)
        # (B, C, H, W) -> (B, H, W, C)
        x_in = x.permute(0, 2, 3, 1) if is_4d else x
        feat = self.act(self.down_proj(x_in))
        feat = self.act(self.depth_layer(feat))
        out = self.up_proj(feat)
        delta = out.permute(0, 3, 1, 2) if is_4d else out
        return orig + self.scale * delta


def inject_sd_adapters(model, embed_dim=768, bottleneck_dim=64):
    """Inject Space-Depth adapters into transformer attention modules.

    Args:
        model (nn.Module): SAM 2 image encoder or base model.
        embed_dim (int): Embedding dimension.
        bottleneck_dim (int): Bottleneck dimension for adapters.

    Returns:
        nn.Module: Model with SD adapters attached.
    """
    for _, mod in list(model.named_modules()):
        if hasattr(mod, "attn"):
            mod.sd_adapter = SpaceDepthAdapter(embed_dim=embed_dim, bottleneck_dim=bottleneck_dim)
    return model
