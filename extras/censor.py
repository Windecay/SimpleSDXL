import os
import threading
import logging

import numpy as np
import torch
from transformers import CLIPConfig, CLIPImageProcessor

import ldm_patched.modules.model_management as model_management
import modules.config
from extras.safety_checker.models.safety_checker import StableDiffusionSafetyChecker
from ldm_patched.modules.model_patcher import ModelPatcher

safety_checker_repo_root = os.path.join(os.path.dirname(__file__), 'safety_checker')
config_path = os.path.join(safety_checker_repo_root, "configs", "config.json")
preprocessor_config_path = os.path.join(safety_checker_repo_root, "configs", "preprocessor_config.json")

logger = logging.getLogger(__name__)


class Censor:
    def __init__(self):
        self.safety_checker_model: ModelPatcher | None = None
        self.clip_image_processor: CLIPImageProcessor | None = None
        self.load_device = torch.device('cpu')
        self.offload_device = torch.device('cpu')
        self._init_lock = threading.Lock()

    def init(self):
        if self.safety_checker_model is not None and self.clip_image_processor is not None:
            return

        with self._init_lock:
            if self.safety_checker_model is not None and self.clip_image_processor is not None:
                return

            if self.clip_image_processor is None:
                self.clip_image_processor = CLIPImageProcessor.from_json_file(preprocessor_config_path)

            if self.safety_checker_model is None:
                safety_checker_model = modules.config.downloading_safety_checker_model()
                clip_config = CLIPConfig.from_json_file(config_path)
                model = StableDiffusionSafetyChecker(clip_config)
                state_dict = torch.load(safety_checker_model, map_location="cpu")
                if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
                    state_dict = state_dict["state_dict"]
                if isinstance(state_dict, dict) and isinstance(state_dict.get("model"), dict):
                    state_dict = state_dict["model"]
                if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
                    state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
                if isinstance(state_dict, dict):
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                    if missing_keys or unexpected_keys:
                        logger.warning(
                            "NSFW safety checker weights loaded with mismatches: missing=%s unexpected=%s",
                            len(missing_keys),
                            len(unexpected_keys),
                        )
                model.eval()

                self.load_device = model_management.text_encoder_device()
                self.offload_device = model_management.text_encoder_offload_device()

                model.to(self.offload_device)

                self.safety_checker_model = ModelPatcher(model, load_device=self.load_device, offload_device=self.offload_device)

    def censor(self, images: list | np.ndarray) -> list | np.ndarray:
        try:
            self.init()
        except Exception:
            logger.exception("NSFW safety checker init failed; skip censor")
            return images

        if self.safety_checker_model is None or self.clip_image_processor is None:
            logger.warning("NSFW safety checker unavailable; skip censor")
            return images

        try:
            model_management.load_model_gpu(self.safety_checker_model)
        except Exception:
            logger.exception("NSFW safety checker load failed; skip censor")
            return images

        single = False
        if not isinstance(images, (list, np.ndarray)):
            images = [images]
            single = True

        try:
            safety_checker_input = self.clip_image_processor(images, return_tensors="pt")
            safety_checker_input.to(device=self.load_device)
            checked_images, has_nsfw_concept = self.safety_checker_model.model(
                images=images,
                clip_input=safety_checker_input.pixel_values
            )
            checked_images = [image.astype(np.uint8) for image in checked_images]
        except Exception:
            logger.exception("NSFW safety checker inference failed; skip censor")
            return images

        if single:
            checked_images = checked_images[0]

        return checked_images


default_censor = Censor().censor
