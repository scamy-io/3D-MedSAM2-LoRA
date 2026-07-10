# [Week 4] src/train/lora.py
import torch
import torch.nn as nn


# ── Option A — peft (recommended starting point) ─────────────────────────────

def add_lora_peft(image_encoder, r=8, alpha=16):
    """Inject LoRA adapters into the SAM 2 image encoder using the peft library.

    Specification:
    - Import LoraConfig and get_peft_model from peft.
    - Target the fused "qkv" linear layer inside each Hiera attention block.
    - Config: r=r, lora_alpha=alpha, target_modules=["qkv"],
              lora_dropout=0.05, bias="none".
    - Return get_peft_model(image_encoder, config).
    - After calling this, trainable_report() should show << 1 % trainable params.

    Args:
        image_encoder: SAM 2 Hiera ViT (predictor.model.image_encoder).
        r     (int):   LoRA rank (default 8).
        alpha (int):   LoRA scaling factor (effective scale = alpha / r).

    Returns:
        image_encoder wrapped with peft LoRA adapters (modified in-place).
    """
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["qkv"],
        lora_dropout=0.05,
        bias="none",
    )
    return get_peft_model(image_encoder, cfg)


# ── Option B — custom Q & V only (stretch goal, no peft dependency) ──────────

class LoRALinear(nn.Module):
    """Wrap a fused QKV linear layer, applying low-rank updates to Q and V only.

    The Hiera attention QKV linear has out_features = 3 × in_features
    (layout [Q | K | V] along the output dimension).

    Specification for __init__:
    - Store base; freeze all its parameters (requires_grad = False).
    - self.r, self.scale = r, alpha / r
    - self.dim = base.in_features   (size of each Q / K / V head group)
    - self.A = nn.Parameter(torch.empty(r, base.in_features)); init Kaiming uniform.
    - self.B = nn.Parameter(torch.zeros(base.out_features, r)); zero-init so
               LoRA starts as identity (no output change at step 0).

    Specification for forward(x):
    - delta = (x @ A.t()) @ B.t() * scale   → same shape as base(x)
    - If self.qv_only: zero the middle third of delta (the K slice):
          delta[..., self.dim : 2 * self.dim] = 0.0
    - Return base(x) + delta

    Args:
        base    (nn.Linear): The original fused QKV projection to wrap.
        r       (int):       LoRA rank.
        alpha   (int):       LoRA alpha.
        qv_only (bool):      If True, zero the K-head delta (Q and V only).
    """

    def __init__(self, base: nn.Linear, r=8, alpha=16, qv_only=True):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.qv_only = qv_only
        self.r = r
        self.scale = alpha / r
        self.dim = base.in_features
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        lora_term = torch.matmul(torch.matmul(x, self.A.t()), self.B.t()) * self.scale
        if self.qv_only:
            lora_term[..., self.dim : 2 * self.dim] = 0.0
        return self.base(x) + lora_term


def inject_lora_qv(model, r=8, alpha=16):
    """Replace every attention QKV linear in the model with a LoRALinear.

    Specification:
    - Walk model.named_modules().
    - For any module that has an `attn` attribute with a `qkv` nn.Linear,
      replace attn.qkv with LoRALinear(attn.qkv, r, alpha, qv_only=True).
    - Also handle modules that expose `qkv` directly as an nn.Linear.
    - Return the modified model.

    Args:
        model         : SAM 2 image encoder (Hiera ViT).
        r     (int):    LoRA rank.
        alpha (int):    LoRA alpha.

    Returns:
        model with all QKV linears replaced by LoRALinear (modified in-place).
    """
    for _, module in list(model.named_modules()):
        if hasattr(module, "attn") and hasattr(module.attn, "qkv") and isinstance(module.attn.qkv, nn.Linear):
            module.attn.qkv = LoRALinear(module.attn.qkv, r=r, alpha=alpha, qv_only=True)
        elif hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
            module.qkv = LoRALinear(module.qkv, r=r, alpha=alpha, qv_only=True)
    return model


def trainable_report(model):
    """Print the count and fraction of trainable parameters.

    Specification:
    - Compute trainable = sum of numel for params where requires_grad is True.
    - Compute total = sum of numel for all params.
    - Print: "Trainable: {trainable:,} / {total:,} = {pct:.3f}%"
    - Expected: trainable well under 1 % after LoRA injection.

    Args:
        model: Any nn.Module (typically predictor.model after LoRA injection).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = (trainable / total * 100) if total > 0 else 0.0
    print(f"Trainable: {trainable:,} / {total:,} = {pct:.3f}%")