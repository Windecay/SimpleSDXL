import os
import sys
import torch
import numpy as np
from PIL import Image
import io
import base64
import safetensors.torch
import modules.config as config
from modules.model_loader import load_file_from_url
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from enhanced.birefnet.models.birefnet import BiRefNet
except ImportError as e:
    logger.error(f"Error importing BiRefNet modules: {e}")
    raise e

interpolation_modes_mapping = {
    "nearest": 0,
    "bilinear": 2,
    "bicubic": 3,
    "nearest-exact": 0,
}

class ImagePreprocessor:
    def __init__(self, resolution, upscale_method="bilinear") -> None:
        interpolation = interpolation_modes_mapping.get(upscale_method, 2)
        self.transform_image = transforms.Compose([
            transforms.Resize(resolution, interpolation=interpolation),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def proc(self, image) -> torch.Tensor:
        image = self.transform_image(image)
        return image

def filter_mask(mask, threshold=0.5):
    mask[mask >= threshold] = 1
    mask[mask < threshold] = 0
    return mask

def simple_upscale(image, width, height, method):
    """
    Simplified upscale function to replace comfy.utils.common_upscale
    """
    if method == "bislerp":
        return torch.nn.functional.interpolate(image, size=(height, width), mode='bilinear', align_corners=False)
    elif method == "bilinear":
        return torch.nn.functional.interpolate(image, size=(height, width), mode='bilinear', align_corners=False)
    elif method == "bicubic":
        return torch.nn.functional.interpolate(image, size=(height, width), mode='bicubic', align_corners=False)
    elif method == "nearest-exact":
        try:
            return torch.nn.functional.interpolate(image, size=(height, width), mode='nearest-exact')
        except:
            return torch.nn.functional.interpolate(image, size=(height, width), mode='nearest')
    elif method == "area":
        return torch.nn.functional.interpolate(image, size=(height, width), mode='area')
    else:
        return torch.nn.functional.interpolate(image, size=(height, width), mode='nearest')

# --- Global Cache ---
_model_cache = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

VERSION = ["old", "v1"]

def get_birefnet_model(model_name="General"):
    global _model_cache
    if _model_cache is None:
        logger.info(f"Loading BiRefNet model: {model_name}...")

        possible_paths = [
            os.path.join(config.path_models_root, "rembg", f"{model_name}.safetensors"),
            os.path.join(config.path_models_root, "birefnet", f"{model_name}.safetensors"),
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
             logger.warning(f"Model {model_name} not found locally. Attempting download...")
             download_url = f"https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/rembg/General.safetensors"
             target_dir = os.path.join(config.path_models_root, "rembg")
             if not os.path.exists(target_dir):
                 os.makedirs(target_dir, exist_ok=True)
             
             try:
                 load_file_from_url(download_url, model_dir=target_dir, file_name=f"{model_name}.safetensors")
                 model_path = os.path.join(target_dir, f"{model_name}.safetensors")
             except Exception as e:
                 logger.error(f"[Download failed: {e}")
                 raise e
        
        logger.info(f"Found model at: {model_path}")

        bb_index = 6
        biRefNet_model = BiRefNet(bb_pretrained=False, bb_index=bb_index)
        
        state_dict = safetensors.torch.load_file(model_path, device=_device)
        biRefNet_model.load_state_dict(state_dict)
        biRefNet_model.to(_device)
        biRefNet_model.eval()

        _model_cache = (biRefNet_model, VERSION[1])
        
    return _model_cache

def check_model_availability():
    """Check if the model infrastructure is ready."""
    try:
        if BiRefNet:
            return {
                "available": True, 
                "reason": "ready", 
                "message": "Model engine is ready. Model will be downloaded on first use if missing."
            }
    except Exception as e:
        return {"available": False, "reason": "error", "message": str(e)}
    return {"available": False, "reason": "unknown", "message": "Unknown status"}

def get_mask(model_data, images, width=1024, height=1024, upscale_method='bilinear', mask_threshold=0.000):
    model, version = model_data
    model_device_type = next(model.parameters()).device.type
    b, h, w, c = images.shape
    image_bchw = images.permute(0, 3, 1, 2)

    image_preproc = ImagePreprocessor(resolution=(height, width), upscale_method=upscale_method)
    im_tensor = image_preproc.proc(image_bchw)

    _mask_bchw = []
    for each_image in im_tensor:
        with torch.no_grad():
            each_mask = model(each_image.unsqueeze(0).to(model_device_type))[-1].sigmoid().cpu()
        _mask_bchw.append(each_mask)

    mask_bchw = torch.cat(_mask_bchw, dim=0)
    # Resize mask back to original size
    mask = simple_upscale(mask_bchw, w, h, upscale_method)
    # (b, 1, h, w)
    if mask_threshold > 0:
        mask = filter_mask(mask, threshold=mask_threshold)

    return mask.squeeze(1),

def process_matting(image_base64, threshold=0.5):
    """
    Process matting request using BiRefNet.
    
    Args:
        image_base64 (str): Base64 encoded image string (with or without prefix).
        threshold (float): Mask threshold.
        
    Returns:
        dict: {"matted_image": base64_str, "alpha_mask": base64_str}
    """
    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != 'RGB':
            image_input = image.convert('RGB')
        else:
            image_input = image

        image_np = np.array(image_input).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0) # (1, H, W, C)

        model_data = get_birefnet_model()

        mask_tensor = get_mask(
            model_data, 
            image_tensor, 
            width=1024, 
            height=1024, 
            upscale_method='bilinear',
            mask_threshold=threshold
        )[0]

        mask_np = mask_tensor.squeeze().cpu().numpy()

        mask_image = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')

        if mask_image.size != image.size:
            mask_image = mask_image.resize(image.size, Image.BILINEAR)

        if image.mode != 'RGBA':
            image_rgba = image.convert('RGBA')
        else:
            image_rgba = image
            
        r, g, b, _ = image_rgba.split()
        matted_image = Image.merge('RGBA', (r, g, b, mask_image))

        matted_buf = io.BytesIO()
        matted_image.save(matted_buf, format="PNG")
        matted_b64 = "data:image/png;base64," + base64.b64encode(matted_buf.getvalue()).decode("utf-8")
        
        mask_buf = io.BytesIO()
        mask_image.save(mask_buf, format="PNG")
        mask_b64 = "data:image/png;base64," + base64.b64encode(mask_buf.getvalue()).decode("utf-8")
        
        return {
            "matted_image": matted_b64,
            "alpha_mask": mask_b64
        }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise e
