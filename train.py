"""
train.py — PyTorch Lightning Training Module & Execution
==========================================================
Houses the LightningModule that encapsulates:
    • forward pass
    • training & validation steps
    • loss computation (Dice + BCE combined)
    • AdamW optimiser configuration

Run this file to execute a `fast_dev_run` sanity check:
    python train.py

fast_dev_run=True runs exactly 1 batch of training + 1 batch of validation,
verifying tensor shapes, loss backprop, and memory fitness (no real training).
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

import config
from models import build_unet, build_attention_unet
from dataset import build_dataset, build_dataloader


# ═══════════════════════════════════════════════════════════════
# Lightning Module
# ═══════════════════════════════════════════════════════════════
class FloodSegmentationModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for flood segmentation training.

    Handles:
        - Model instantiation (U-Net or Attention U-Net)
        - Combined Dice + BCE loss
        - AdamW optimiser
        - Training and validation step logging
    """

    def __init__(self, model_type: str = "unet"):
        """
        Args:
            model_type: 'unet' for vanilla U-Net,
                        'attention_unet' for scSE Attention U-Net.
        """
        super().__init__()
        self.save_hyperparameters()

        # ── Model ──
        if model_type == "attention_unet":
            self.model = build_attention_unet()
        else:
            self.model = build_unet()

        # ── Loss ──
        self.loss_fn = self._build_loss()

    # ───────────────── Loss Construction ─────────────────
    def _build_loss(self) -> nn.Module:
        """
        Build the loss function based on config.LOSS_TYPE.

        Returns a module that accepts raw logits and binary masks.
        """
        if config.LOSS_TYPE == "dice":
            return smp.losses.DiceLoss(
                mode="binary",
                from_logits=True,
            )
        elif config.LOSS_TYPE == "bce":
            return nn.BCEWithLogitsLoss()
        else:  # "combined"
            return CombinedLoss()

    # ───────────────── Forward ─────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits of shape (B, 1, H, W)."""
        return self.model(x)

    # ───────────────── Training Step ─────────────────
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]       # (B, C, H, W) float32
        masks = batch["mask"]         # (B, 1, H, W) or (B, H, W)

        # Ensure masks are float and have channel dim
        masks = self._prepare_mask(masks)

        logits = self(images)         # (B, 1, H, W)
        loss = self.loss_fn(logits, masks)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ───────────────── Validation Step ─────────────────
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        masks = batch["mask"]
        masks = self._prepare_mask(masks)

        logits = self(images)
        loss = self.loss_fn(logits, masks)

        # IoU / F1 for monitoring
        preds = (torch.sigmoid(logits) > 0.5).float()
        tp = (preds * masks).sum()
        fp = (preds * (1 - masks)).sum()
        fn = ((1 - preds) * masks).sum()
        iou = tp / (tp + fp + fn + 1e-8)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/iou", iou, prog_bar=True, on_epoch=True)
        return loss

    # ───────────────── Optimiser ─────────────────
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        return optimizer

    # ───────────────── Helpers ─────────────────
    @staticmethod
    def _prepare_mask(mask: torch.Tensor) -> torch.Tensor:
        """
        Normalise the mask tensor:
            - Add channel dim if missing → (B, 1, H, W)
            - Cast to float32
            - Clamp 'no-data' pixels (255) to 0 so they don't explode the loss
        """
        if mask.ndim == 3:              # (B, H, W) → (B, 1, H, W)
            mask = mask.unsqueeze(1)
        mask = mask.float()
        # MMFlood encodes missing pixels as 255; zero them out
        mask[mask == config.IGNORE_INDEX] = 0.0
        return mask


# ═══════════════════════════════════════════════════════════════
# Combined Dice + BCE Loss
# ═══════════════════════════════════════════════════════════════
class CombinedLoss(nn.Module):
    """
    Weighted sum of SMP DiceLoss and BCEWithLogitsLoss.
    Both operate on raw logits → no need to apply sigmoid before the loss.
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce = nn.BCEWithLogitsLoss()
        self.w_dice = dice_weight
        self.w_bce = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.w_dice * self.dice(logits, targets) + \
               self.w_bce * self.bce(logits, targets)


# ═══════════════════════════════════════════════════════════════
# Execution — Fast Dev Run Sanity Check
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  MMFlood Segmentation — Phase 1 Sanity Check")
    print("=" * 60)

    # ── Data ──
    print("\n[1/4] Building datasets …")
    train_ds = build_dataset(split="train")
    val_ds = build_dataset(split="val")

    print("[2/4] Building dataloaders …")
    train_dl = build_dataloader(train_ds)
    val_dl = build_dataloader(val_ds, shuffle=False)

    # ── Model ──
    print("[3/4] Initialising model …")
    # Change to "attention_unet" to test the scSE variant
    module = FloodSegmentationModule(model_type="unet")
    print(f"       Architecture : {module.hparams.model_type}")
    print(f"       In channels  : {config.IN_CHANNELS}")
    print(f"       Image size   : {config.IMAGE_SIZE}")
    print(f"       Batch size   : {config.BATCH_SIZE}")
    print(f"       Precision    : {config.PRECISION}")

    # ── Trainer ──
    print("[4/4] Launching fast_dev_run …\n")
    trainer = pl.Trainer(
        fast_dev_run=config.FAST_DEV_RUN,    # ← 1 batch train + 1 batch val
        accelerator=config.ACCELERATOR,
        precision=config.PRECISION,
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
        max_epochs=config.MAX_EPOCHS,
        enable_checkpointing=False,          # no checkpoint for dev run
    )

    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)

    print("\n" + "=" * 60)
    print("  [OK] fast_dev_run PASSED -- tensors flow, no OOM, no shape errors")
    print("    Phase 1 is complete.  Push to GitHub for Phase 2!")
    print("=" * 60)
