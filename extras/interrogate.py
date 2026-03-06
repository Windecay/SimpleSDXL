import os
import torch
import numpy as np
import ldm_patched.modules.model_management as model_management

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from modules.model_loader import load_file_from_url
from modules.config import paths_clip_vision
from ldm_patched.modules.model_patcher import ModelPatcher
from extras.BLIP.models.blip import blip_decoder
from PIL import Image
import modules.util as util


blip_image_eval_size = 384
blip_repo_root = os.path.join(os.path.dirname(__file__), 'BLIP')


class Interrogator:
    def __init__(self):
        self.blip_model = None
        self.load_device = torch.device('cpu')
        self.offload_device = torch.device('cpu')
        self.dtype = torch.float32

    @torch.no_grad()
    @torch.inference_mode()
    def interrogate(self, img_rgb):
        if self.blip_model is None:
            filename = load_file_from_url(
                url='https://huggingface.co/lllyasviel/misc/resolve/main/model_base_caption_capfilt_large.pth',
                model_dir=paths_clip_vision[0],
                file_name='model_base_caption_capfilt_large.pth',
            )

            model = blip_decoder(pretrained=filename, image_size=blip_image_eval_size, vit='base',
                                 med_config=os.path.join(blip_repo_root, "configs", "med_config.json"))
            model.eval()

            self.load_device = model_management.text_encoder_device()
            self.offload_device = model_management.text_encoder_offload_device()
            self.dtype = torch.float32

            model.to(self.offload_device)

            if model_management.should_use_fp16(device=self.load_device):
                model.half()
                self.dtype = torch.float16

            self.blip_model = ModelPatcher(model, load_device=self.load_device, offload_device=self.offload_device)

        model_management.load_model_gpu(self.blip_model)

        if isinstance(img_rgb, np.ndarray):
            if img_rgb.dtype != np.uint8:
                img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
            if img_rgb.ndim == 2:
                img_rgb = util.HWC3(img_rgb)
            elif img_rgb.ndim == 3:
                if img_rgb.shape[2] in (1, 3, 4):
                    img_rgb = util.HWC3(img_rgb)
                elif img_rgb.shape[2] > 3:
                    img_rgb = img_rgb[:, :, :3]
            img_rgb = Image.fromarray(img_rgb).convert("RGB")
        elif isinstance(img_rgb, Image.Image):
            img_rgb = img_rgb.convert("RGB")

        gpu_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(img_rgb).unsqueeze(0).to(device=self.load_device, dtype=self.dtype)

        caption = self.blip_model.model.generate(gpu_image, sample=True, num_beams=1, max_length=75)[0]

        return caption


_interrogator = Interrogator()
default_interrogator = _interrogator.interrogate


def free_model():
    if _interrogator.blip_model is not None:
        _interrogator.blip_model.model.to('cpu')
        del _interrogator.blip_model
        _interrogator.blip_model = None
    import torch
    import gc
    torch.cuda.empty_cache()
    gc.collect()
