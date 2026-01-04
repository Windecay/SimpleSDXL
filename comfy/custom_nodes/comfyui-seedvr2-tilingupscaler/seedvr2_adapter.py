"""Adapter for calling SeedVR2 VideoUpscaler from the tiling upscaler.

This module provides a simple interface to execute SeedVR2 upscaling
using the DiT and VAE configurations from the loader nodes.
"""

from __future__ import annotations

import logging
import torch
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


def get_upscaler_class():
    """Get the SeedVR2VideoUpscaler class from ComfyUI's node registry.

    Returns:
        The SeedVR2VideoUpscaler class

    Raises:
        RuntimeError: If SeedVR2VideoUpscaler is not found
    """
    import nodes

    for node_class in nodes.NODE_CLASS_MAPPINGS.values():
        if getattr(node_class, "__name__", "") == "SeedVR2VideoUpscaler":
            return node_class

    raise RuntimeError(
        "SeedVR2VideoUpscaler node not found. "
        "Please install ComfyUI-SeedVR2_VideoUpscaler v2.5 or later."
    )


def execute_seedvr2(
    *,
    images: torch.Tensor,
    dit_config: Dict[str, Any],
    vae_config: Dict[str, Any],
    seed: int,
    resolution: int,
    batch_size: int = 1,
    color_correction: str = "lab",
) -> torch.Tensor:
    """Execute SeedVR2 upscaling on a batch of images.

    Args:
        images: Input images tensor (N, H, W, C) in [0, 1] range
        dit_config: DiT model configuration from SeedVR2LoadDiTModel node
        vae_config: VAE model configuration from SeedVR2LoadVAEModel node
        seed: Random seed for reproducibility
        resolution: Target resolution for the shortest edge
        batch_size: Number of frames to process together
        color_correction: Color correction method (lab, wavelet, wavelet_adaptive, hsv, adain, none)

    Returns:
        Upscaled images tensor (N, H', W', C) in [0, 1] range
    """
    upscaler_cls = get_upscaler_class()

    # Call the SeedVR2VideoUpscaler execute method
    result = upscaler_cls.execute(
        image=images,
        dit=dit_config,
        vae=vae_config,
        seed=seed,
        resolution=resolution,
        max_resolution=0,  # No limit
        batch_size=batch_size,
        uniform_batch_size=False,
        temporal_overlap=0,
        prepend_frames=0,
        color_correction=color_correction,
        input_noise_scale=0.0,
        latent_noise_scale=0.0,
        offload_device=dit_config.get("offload_device", "none"),
        enable_debug=False,
    )

    # Extract tensor from io.NodeOutput
    if hasattr(result, "values"):
        tensor = result.values[0] if isinstance(result.values, (list, tuple)) else result.values
    elif hasattr(result, "__getitem__"):
        tensor = result[0]
    else:
        tensor = result

    return tensor
