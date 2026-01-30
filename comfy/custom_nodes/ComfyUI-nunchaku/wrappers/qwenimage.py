from typing import Callable, List, Tuple, Union
from pathlib import Path

import torch
from torch import nn
import comfy.model_management
import logging

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.caching.fbcache import cache_context, create_cache_context
from ..nunchaku_code.lora_qwen import compose_loras_v2, reset_lora_v2

logger = logging.getLogger(__name__)


class ComfyQwenImageWrapper(nn.Module):
    """
    Wrapper for NunchakuQwenImageTransformer2DModel to support ComfyUI workflows.

    This wrapper separates LoRA composition from the forward pass for maximum efficiency.
    It detects changes to its `loras` attribute and recomposes the underlying model
    lazily when the forward pass is executed.
    """

    def __init__(
            self,
            model: NunchakuQwenImageTransformer2DModel,
            config,
            customized_forward: Callable = None,
            forward_kwargs: dict | None = None,
            cpu_offload_setting: str = "auto",
            vram_margin_gb: float = 4.0
    ):
        super().__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config

        self.loras: List[Tuple[Union[str, Path, dict], float]] = []
        self._applied_loras: List[Tuple[Union[str, Path, dict], float]] = None

        self.cpu_offload_setting = cpu_offload_setting
        self.vram_margin_gb = vram_margin_gb

        self.customized_forward = customized_forward
        self.forward_kwargs = forward_kwargs or {}

        self._prev_timestep = None
        self._cache_context = None

        self._img_ids_cache = {}
        self._txt_ids_cache = {}
        self._linspace_cache_h = {}
        self._linspace_cache_w = {}
        
        self._last_device = None

    def to_safely(self, device):
        """Safely move the model to the specified device."""
        if hasattr(self.model, "to_safely"):
            self.model.to_safely(device)
        else:
            self.model.to(device)
        return self

    def forward(
            self,
            x,
            timestep,
            context=None,
            y=None,
            guidance=None,
            control=None,
            transformer_options={},
            **kwargs,
    ):
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 1:
                timestep_float = timestep.item()
            else:
                timestep_float = timestep.flatten()[0].item()
        else:
            timestep_float = float(timestep)

        model_is_dirty = (
            not self.loras and
            hasattr(self.model, "_lora_slots") and self.model._lora_slots
        )
        
        loras_changed = False
        if self._applied_loras is None:
            loras_changed = True
        elif len(self._applied_loras) != len(self.loras):
            loras_changed = True
        else:
            for applied, current in zip(self._applied_loras, self.loras):
                if applied != current:
                    loras_changed = True
                    break

        try:
            current_device = next(self.model.parameters()).device
        except Exception:
            current_device = None
        device_changed = (self._last_device != current_device)
        
        if loras_changed or model_is_dirty or device_changed:

            reset_lora_v2(self.model)
            
            self._applied_loras = self.loras.copy()
            
            if loras_changed:
                self._cache_context = None
                self._prev_timestep = None

            offload_is_on = hasattr(self.model, "offload_manager") and self.model.offload_manager is not None
            should_enable_offload = offload_is_on

            if self.cpu_offload_setting == "auto" and not offload_is_on and self.loras:
                try:
                    free_vram_gb = comfy.model_management.get_free_memory() / (1024 ** 3)
                    if free_vram_gb < self.vram_margin_gb:
                        logger.info(f"Low VRAM ({free_vram_gb:.2f}GB). Enabling CPU offload for LoRA.")
                        should_enable_offload = True
                except Exception:
                    pass

            if self.loras:
                compose_loras_v2(self.model, self.loras)

            if should_enable_offload:
                if offload_is_on:
                    manager = self.model.offload_manager
                    offload_settings = {"num_blocks_on_gpu": manager.num_blocks_on_gpu, "use_pin_memory": manager.use_pin_memory}
                else:
                    offload_settings = {"num_blocks_on_gpu": 1, "use_pin_memory": False}
                
                self.model.set_offload(False)
                self.model.set_offload(True, **offload_settings)
            
            self._last_device = current_device


        use_caching = getattr(self.model, "residual_diff_threshold_multi", 0) != 0 or getattr(self.model, "_is_cached",
                                                                                              False)
        if use_caching:
            cache_invalid = self._prev_timestep is None or self._prev_timestep < timestep_float + 1e-5
            if cache_invalid:
                self._cache_context = create_cache_context()
            self._prev_timestep = timestep_float

            with cache_context(self._cache_context):
                out = self._execute_model(x, timestep, context, guidance, control, transformer_options, **kwargs)
        else:
            out = self._execute_model(x, timestep, context, guidance, control, transformer_options, **kwargs)

        if isinstance(out, tuple):
            out = out[0]

        if x.ndim == 5 and out.ndim == 4:
            out = out.unsqueeze(2)

        return out

    def _execute_model(self, x, timestep, context, guidance, control, transformer_options, **kwargs):
        model_device = next(self.model.parameters()).device
        if x.device != model_device:
            x = x.to(model_device)
        if context is not None and context.device != model_device:
            context = context.to(model_device)
       
        original_ndim = x.ndim
        if x.ndim == 4:
            x = x.unsqueeze(2)
        elif x.ndim == 5:
            pass

        if self.customized_forward:
            with torch.inference_mode():
                out = self.customized_forward(
                    self.model,
                    hidden_states=x,
                    encoder_hidden_states=context,
                    timestep=timestep,
                    guidance=guidance if self.config.get("guidance_embed", False) else None,
                    control=control,
                    transformer_options=transformer_options,
                    **self.forward_kwargs,
                    **kwargs,
                )
        else:
            with torch.inference_mode():
                out = self.model(
                    x=x,
                    context=context,
                    timestep=timestep,
                    guidance=guidance if self.config.get("guidance_embed", False) else None,
                    control=control,
                    transformer_options=transformer_options,
                    **kwargs,
                )

        if original_ndim == 4 and out.ndim == 5:
            out = out.squeeze(2)

        return out