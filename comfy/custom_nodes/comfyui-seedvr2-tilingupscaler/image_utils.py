"""Image utility functions for tensor/PIL conversions."""

import torch
import numpy as np
from PIL import Image


def tensor_to_pil(tensor):
    """Convert a tensor to PIL Image.

    Args:
        tensor: Input tensor with shape (1, H, W, C) or (H, W, C)

    Returns:
        PIL Image in RGB mode
    """
    # Only squeeze the batch dimension (dim 0), not all dimensions
    # This prevents accidentally removing spatial dimensions of size 1
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension only: (1, H, W, C) -> (H, W, C)
    image_np = tensor.mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, 'RGB')
    return image


def pil_to_tensor(image):
    """Convert a PIL Image to tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
