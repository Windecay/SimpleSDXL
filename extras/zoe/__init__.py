import os
import sys
from pathlib import Path

# Add the current directory to sys.path so that custom_midas_repo and custom_timm can be imported
current_dir = str(Path(__file__).parent.resolve())
if current_dir not in sys.path:
    sys.path.append(current_dir)

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image

from .util import HWC3, common_input_validate, resize_image_with_pad, custom_hf_download, HF_MODEL_NAME
from .zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from .zoedepth.utils.config import get_config

import ldm_patched.modules.model_management as model_management
from ldm_patched.modules.model_patcher import ModelPatcher

class ZoeDetector:
    _CACHE = None

    def __init__(self, model):
        self.model = model
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=HF_MODEL_NAME, filename="ZoeD_M12_N.pt"):
        if cls._CACHE is None:
            try:
                from modules.config import downloading_controlnet_zoe
                model_path = downloading_controlnet_zoe()
            except ImportError:
                model_path = custom_hf_download(pretrained_model_or_path, filename)
                
            conf = get_config("zoedepth", "infer")
            model = ZoeDepth.build_from_config(conf)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
            model.eval()

            load_device = model_management.get_torch_device()
            offload_device = model_management.text_encoder_offload_device()
            patcher = ModelPatcher(model, load_device=load_device, offload_device=offload_device)
            cls._CACHE = (model, patcher)
        
        return cls(cls._CACHE[0])

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self
    
    def __call__(self, input_image, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        _, patcher = self._CACHE
        model_management.load_model_gpu(patcher)
        self.to(model_management.get_torch_device())

        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().to(self.device)
            image_depth = image_depth / 255.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model.infer(image_depth)

            depth = depth[0, 0].cpu().numpy()

            vmin = np.percentile(depth, 2)
            vmax = np.percentile(depth, 85)

            depth -= vmin
            depth /= vmax - vmin
            depth = 1.0 - depth
            depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = remove_pad(HWC3(depth_image))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
