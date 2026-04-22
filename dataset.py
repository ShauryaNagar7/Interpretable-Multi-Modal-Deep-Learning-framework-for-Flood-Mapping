"""
dataset.py — Data Pipeline for MMFlood
========================================
Uses TorchGeo's native MMFlood GeoDataset class to ingest multimodal
satellite imagery (Sentinel-1 SAR + DEM + Hydrography) and output
standardized, memory-safe tensors via RandomGeoSampler.

Key Design Decisions:
    - MMFlood is a GeoDataset (spatially indexed), so we MUST use
      torchgeo.samplers.RandomGeoSampler instead of integer indexing.
    - stack_samples collate_fn merges per-sample dicts into batched tensors.
    - IMAGE_SIZE crops are handled natively by the sampler's `size` param,
      so the large, variable-resolution rasters never fully load into RAM.
"""

import torch
from torch.utils.data import DataLoader

from torchgeo.datasets import MMFlood, stack_samples
from torchgeo.samplers import RandomGeoSampler

import config


def build_dataset(split: str = "train") -> MMFlood:
    """
    Initialise the MMFlood GeoDataset for a given split.

    Args:
        split: One of 'train', 'val', or 'test'.

    Returns:
        A configured MMFlood dataset instance.
    """
    dataset = MMFlood(
        root=str(config.DATA_ROOT),
        split=split,
        include_dem=config.INCLUDE_DEM,
        include_hydro=config.INCLUDE_HYDRO,
        download=config.DOWNLOAD_DATA,
    )
    return dataset


def build_dataloader(
    dataset: MMFlood,
    batch_size: int | None = None,
    samples_per_epoch: int | None = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    Wrap an MMFlood GeoDataset in a PyTorch DataLoader with spatial sampling.

    The RandomGeoSampler handles the spatial cropping:
        - `size=IMAGE_SIZE` extracts fixed-size patches from the rasters.
        - `length=SAMPLES_PER_EPOCH` controls how many patches per epoch.
    This means we never load an entire satellite tile into memory.

    Args:
        dataset:          An MMFlood dataset instance.
        batch_size:       Override config.BATCH_SIZE if needed.
        samples_per_epoch: Override config.SAMPLES_PER_EPOCH if needed.
        shuffle:          Whether the sampler should randomise patch locations.

    Returns:
        A ready-to-iterate DataLoader yielding batched dicts with
        'image' (B, C, H, W) and 'mask' (B, 1, H, W) tensors.
    """
    _batch_size = batch_size or config.BATCH_SIZE
    _length = samples_per_epoch or config.SAMPLES_PER_EPOCH

    sampler = RandomGeoSampler(
        dataset,
        size=config.IMAGE_SIZE,       # spatial crop (pixels)
        length=_length,               # patches per epoch
    )

    dataloader = DataLoader(
        dataset,
        batch_size=_batch_size,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        collate_fn=stack_samples,     # required for GeoDataset dicts
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader


# ─────────────────────────────────────────────────────────────
# Quick smoke-test when run directly
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building training dataset …")
    train_ds = build_dataset(split="train")
    print(f"  Dataset: {train_ds}")

    print("Building dataloader …")
    train_dl = build_dataloader(train_ds)

    print("Fetching one batch …")
    batch = next(iter(train_dl))
    print(f"  image shape : {batch['image'].shape}")   # (B, C, H, W)
    print(f"  mask  shape : {batch['mask'].shape}")     # (B, 1, H, W)
    print(f"  image dtype : {batch['image'].dtype}")
    print(f"  mask  dtype : {batch['mask'].dtype}")
    print("[OK] Data pipeline smoke-test passed.")
