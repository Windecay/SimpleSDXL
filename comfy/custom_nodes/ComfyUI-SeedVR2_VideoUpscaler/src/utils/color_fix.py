import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from ..common.half_precision_fixes import safe_pad_operation, safe_interpolate_operation
from torchvision.transforms import ToTensor, ToPILImage

def adain_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply adaptive instance normalization
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def wavelet_color_fix(target: Image, source: Image, debug):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply wavelet reconstruction
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor, debug)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def calc_mean_std(feat: Tensor, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat:Tensor, style_feat:Tensor):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur using dilated convolution.
    Automatically limits radius to prevent numerical instability at high resolutions.
    """
    # Prevent excessive dilation that causes OOM/numerical issues
    # Conservative limit: 1/8 of smallest dimension
    max_safe_radius = max(1, min(image.shape[-2:]) // 8)
    if radius > max_safe_radius:
        radius = max_safe_radius
    
    # Create Gaussian-like blur kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None].repeat(3, 1, 1, 1)
    
    # Apply padding and convolution
    # safe_pad_operation already handles FP16 conversion if needed
    image = safe_pad_operation(image, (radius, radius, radius, radius), mode='replicate')
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    
    return output

def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to extract frequency components.
    Returns high frequency (detail) and low frequency (color) components.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq

def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor, debug):
    """
    Apply wavelet-based color transfer from style to content.
    Preserves high-frequency details from content while adopting low-frequency color from style.
    Note: Operates in normalized [-1,1] space for SDR images.
    
    Args:
        content_feat: Target tensor with desired details [B, C, H, W]
        style_feat: Source tensor with desired colors [B, C, H, W]  
        debug: Debug instance for logging
        
    Returns:
        Reconstructed tensor with content details and style colors
    """
    # Handle dimension mismatch if needed
    if content_feat.shape != style_feat.shape:
        debug.log(f"Dimension mismatch: content {content_feat.shape} vs style {style_feat.shape}", 
                  level="WARNING", category="precision", force=True)
        
        # Resize style to match content spatial dimensions
        if len(content_feat.shape) >= 3:
            # safe_interpolate_operation handles FP16 conversion automatically
            style_feat = safe_interpolate_operation(
                style_feat, 
                size=content_feat.shape[-2:],
                mode='bilinear', 
                align_corners=False
            )
            debug.log(f"Style resized to: {style_feat.shape}", category="precision", force=True)
    
    # Decompose both features into frequency components
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq  # Free memory immediately
    
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)  
    del style_high_freq  # Free memory immediately
    
    # Safety check (should not happen after resize)
    if content_high_freq.shape != style_low_freq.shape:
        debug.log(f"Final dimension adjustment needed", level="WARNING", category="precision", force=True)
        style_low_freq = safe_interpolate_operation(
            style_low_freq,
            size=content_high_freq.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
    
    # Reconstruct: content details + style color
    result = content_high_freq + style_low_freq
    
    # Safety clamp for normalized SDR range
    # This prevents numerical errors from propagating
    # Note: For HDR support, this would need to be removed
    return torch.clamp(result, -1.0, 1.0)