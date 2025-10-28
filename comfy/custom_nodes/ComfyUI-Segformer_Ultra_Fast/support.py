import os
import math
import torch
import shutil
import folder_paths

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import kornia.morphology as kornia_morph
import kornia.filters as kornia_filters

from PIL import Image
from functools import lru_cache

from huggingface_hub import snapshot_download
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor, VitMatteForImageMatting, VitMatteImageProcessorFast, VitMatteImageProcessor

# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
any_typ = AnyType("*")

def log(message:str, message_type:str='info'):
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    else:
        message = '\033[1;33m' + message + '\033[m'
    print(f"{message}")

def pil2tensor(image:Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
        
def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

@lru_cache(maxsize=1)
def load_segmentation_model(model_name='mattmdjaga/segformer_b2_clothes', device='cpu'):
    old_model_dir = os.path.join(folder_paths.models_dir, model_name.split("/")[-1])
    model_dir = os.path.join(folder_paths.models_dir, "segformer", model_name.replace("/", "-"))
    if not os.path.exists(model_dir) and os.path.exists(old_model_dir):
        log("Old model folder found, moving to new location...")
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        shutil.move(old_model_dir, model_dir)
    if not os.path.exists(model_dir):
        log(f"File Not Found: {model_dir}", message_type='info')
        # snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)
        
    log(f"Loading segmentation model '{model_name}' to device '{device}'...", message_type='info')
    processor = SegformerImageProcessor.from_pretrained(model_dir)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_dir).to(device)
    model.eval()
    log("Segmentation model loaded successfully.", message_type='finish')
    return model, processor

@lru_cache(maxsize=1)
def load_VITMatte_model(model_name="hustvl/vitmatte-small-composition-1k", fast=True):
    old_model_dir = os.path.join(folder_paths.models_dir, model_name.split("/")[-1])
    model_dir = os.path.join(folder_paths.models_dir, "vitmatte")
    if not os.path.exists(model_dir) and os.path.exists(old_model_dir):
        log("Old model folder found, moving to new location...")
        os.rename(old_model_dir, model_dir)
    if not os.path.exists(model_dir):
        log(f"Downloading VITMatte model '{model_name}' from huggingface...", message_type='info')
        snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)
    
    log(f"Loading VITMatte model '{model_name}'...", message_type='info')
    if fast:
        processor = VitMatteImageProcessorFast.from_pretrained(model_dir)
    else:
        processor = VitMatteImageProcessor.from_pretrained(model_dir)
    model = VitMatteForImageMatting.from_pretrained(model_dir)
    log("VITMatte model loaded successfully.", message_type='finish')
    return {"model": model, "processor": processor}

def get_segmentation_batch(image_batch_tensor: torch.Tensor, model, processor, device: str) -> torch.Tensor:
    images_pil = [tensor2pil(img) for img in image_batch_tensor]
    target_size = images_pil[0].size[::-1]
    inputs = processor(images=images_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(
        logits, size=target_size, mode="bilinear", align_corners=False
    )
    pred_seg_batch = upsampled_logits.argmax(dim=1)
    return pred_seg_batch

def histogram_remap_batch(image_tensor: torch.Tensor, blackpoint: float, whitepoint: float) -> torch.Tensor:
    bp = min(blackpoint, whitepoint - 0.001)
    scale = 1.0 / (whitepoint - bp)
    return torch.clamp((image_tensor - bp) * scale, 0.0, 1.0)

def generate_VITMatte_trimap_batch(mask_batch: torch.Tensor, erode_kernel_size: int, dilate_kernel_size: int) -> torch.Tensor:
    device = mask_batch.device
    erode_kernel = torch.ones(erode_kernel_size, erode_kernel_size, device=device)
    dilate_kernel = torch.ones(dilate_kernel_size, dilate_kernel_size, device=device)
    
    eroded_mask = kornia_morph.erosion(mask_batch, erode_kernel)
    dilated_mask = kornia_morph.dilation(mask_batch, dilate_kernel)
    
    trimap = torch.full_like(mask_batch, 0.5)
    trimap[dilated_mask == 0] = 0.0
    trimap[eroded_mask == 1] = 1.0
    
    return trimap

def generate_VITMatte_batch(image_batch_tensor: torch.Tensor, trimap_batch_tensor: torch.Tensor, device: str = "cpu", max_megapixels: float = 2.0, fast_mode: bool = True) -> torch.Tensor:
    vit_matte_model_pack = load_VITMatte_model(fast=fast_mode)
    model = vit_matte_model_pack["model"].to(device)
    processor = vit_matte_model_pack["processor"]

    image_batch_bchw = image_batch_tensor.permute(0, 3, 1, 2)
    B, C, H, W = image_batch_bchw.shape
    orig_H, orig_W = H, W

    max_pixels = max_megapixels * 1024 * 1024
    if H * W > max_pixels:
        ratio = W / H
        target_W = int(math.sqrt(ratio * max_pixels))
        target_H = int(target_W / ratio)
        image_batch_bchw = F.interpolate(image_batch_bchw, size=(target_H, target_W), mode='bilinear', align_corners=False)
        trimap_batch_tensor = F.interpolate(trimap_batch_tensor, size=(target_H, target_W), mode='bilinear', align_corners=False)
        H, W = target_H, target_W

    images_pil = [tensor2pil(img.permute(1, 2, 0)) for img in image_batch_bchw]
    trimaps_pil = [tensor2pil(trimap) for trimap in trimap_batch_tensor]
    inputs = processor(images=images_pil, trimaps=trimaps_pil, return_tensors="pt")
    
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        alphas = model(**inputs).alphas
    alphas = alphas[:, :, :H, :W]
    
    if H != orig_H or W != orig_W:
        alphas = F.interpolate(alphas, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
        
    return alphas

def guided_filter_alpha_batch(image_batch: torch.Tensor, mask_batch: torch.Tensor, filter_radius: int) -> torch.Tensor:
    device = image_batch.device; mask_batch = mask_batch.to(device); image_bchw = image_batch.permute(0, 3, 1, 2); kernel_size = filter_radius * 2 + 1
    filtered_mask = kornia_filters.guided_blur(guidance=image_bchw, input=mask_batch, kernel_size=kernel_size, eps=0.01)
    
    return filtered_mask

def mask_edge_detail_batch(image_batch: torch.Tensor, mask_batch: torch.Tensor, detail_range:int=8, black_point:float=0.01, white_point:float=0.99) -> torch.Tensor:
    from pymatting import fix_trimap, estimate_alpha_cf
    device = image_batch.device; mask_batch = mask_batch.to(device); d = detail_range * 5 + 1
    if not bool(d % 2): d += 1
    blurred_mask_batch = kornia_filters.gaussian_blur2d(mask_batch, (d, d), (detail_range, detail_range))
    images_np = image_batch.cpu().numpy().astype(np.float64); blurred_masks_np = blurred_mask_batch.cpu().numpy().astype(np.float64); alpha_results = []
    
    for i in range(images_np.shape[0]):
        img_np = images_np[i]; trimap_np = blurred_masks_np[i, 0, :, :]
        trimap_fixed = fix_trimap(trimap_np, black_point, white_point)
        alpha = estimate_alpha_cf(img_np, trimap_fixed, laplacian_kwargs={"epsilon": 1e-6}, cg_kwargs={"maxiter": 200})
        alpha_results.append(alpha)
    alpha_batch_np = np.stack(alpha_results, axis=0); alpha_batch_tensor = torch.from_numpy(alpha_batch_np.astype(np.float32)).unsqueeze(1).to(device)
    
    return alpha_batch_tensor

import torch
import kornia.morphology as kornia_morph
import kornia.filters as kornia_filters

def expand_mask_batch(mask_batch_4d: torch.Tensor, expand_pixels: int, tapered_corners: bool) -> torch.Tensor:
    if expand_pixels == 0:
        return mask_batch_4d
    
    device = mask_batch_4d.device
    abs_expand = abs(expand_pixels)
    
    c = 0 if tapered_corners else 1
    kernel_np = np.array([[c, 1, c], [1, 1, 1], [c, 1, c]])
    kernel = torch.from_numpy(kernel_np).float().to(device)
    processed_mask = mask_batch_4d
    
    for _ in range(abs_expand):
        if expand_pixels > 0:
            processed_mask = kornia_morph.dilation(processed_mask, kernel)
        else:
            processed_mask = kornia_morph.erosion(processed_mask, kernel)
    
    return processed_mask

def get_device_list():
    devs = []
    try:
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
            devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
            devs += [f"mps:{i}" for i in range(torch.mps.device_count())]
    except Exception:
        pass
    return devs+["cpu"]