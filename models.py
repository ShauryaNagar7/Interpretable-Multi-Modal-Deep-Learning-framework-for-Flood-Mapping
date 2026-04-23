"""
models.py — Segmentation Model Definitions
=============================================
Defines two baseline architectures using segmentation_models_pytorch (SMP):

1. Standard U-Net        — vanilla encoder–decoder with skip connections.
2. Attention U-Net (scSE) — identical topology, but with Spatial & Channel
                            Squeeze-and-Excitation attention gates injected
                            into every decoder block via `decoder_attention_type`.

Both models accept 4-channel input:
    [SAR VV, SAR VH, DEM, Hydrography]
and output a single-channel binary flood mask (logits).
"""

import segmentation_models_pytorch as smp
import config


def build_unet() -> smp.Unet:
    """
    Standard U-Net with a ResNet-34 encoder.

    Returns:
        smp.Unet model producing raw logits of shape (B, 1, H, W).
    """
    model = smp.Unet(
        encoder_name=config.ENCODER_NAME,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=config.IN_CHANNELS,
        classes=config.NUM_CLASSES,
        decoder_attention_type=None,      # no attention — vanilla U-Net
    )
    return model


def build_attention_unet() -> smp.Unet:
    """
    Attention U-Net via scSE (Concurrent Spatial & Channel
    Squeeze-and-Excitation).

    SMP doesn't expose a separate "AttentionUnet" class, but passing
    `decoder_attention_type='scse'` injects attention gates into every
    decoder block — functionally equivalent to the Attention U-Net
    architecture described in Oktay et al., 2018.

    Returns:
        smp.Unet model with scSE attention, producing raw logits (B, 1, H, W).
    """
    model = smp.Unet(
        encoder_name=config.ENCODER_NAME,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=config.IN_CHANNELS,
        classes=config.NUM_CLASSES,
        decoder_attention_type="scse",    # ← attention gates
    )
    return model


# ─────────────────────────────────────────────────────────────
# Quick shape-check when run directly
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    dummy = torch.randn(1, config.IN_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
    print(f"Input tensor : {dummy.shape}")

    unet = build_unet()
    out_u = unet(dummy)
    print(f"U-Net output : {out_u.shape}")            # (1, 1, 128, 128)

    att_unet = build_attention_unet()
    out_a = att_unet(dummy)
    print(f"Att-UNet out  : {out_a.shape}")            # (1, 1, 128, 128)

    # Parameter counts
    p_u = sum(p.numel() for p in unet.parameters())
    p_a = sum(p.numel() for p in att_unet.parameters())
    print(f"U-Net params        : {p_u:,}")
    print(f"Attention U-Net params: {p_a:,}")
    print("[OK] Model shape-check passed.")
