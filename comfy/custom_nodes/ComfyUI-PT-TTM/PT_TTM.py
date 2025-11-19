# PT_TTM.py

import torch
import torch.nn.functional as F

class PT_TTM:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embeds": ("WANVIDIMAGE_EMBEDS",),
                "reference_latents": ("LATENT", {"tooltip": "Latents used as reference for TTM"}),
                "mask": ("MASK", {"tooltip": "Mask used for TTM"}),
                "reference_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, 
                                               "tooltip": "控制参考视频对生成视频动态的影响强度。0=无影响，1=完全影响"}),
                "start_step": ("INT", {"default": 0, "min": -1, "max": 1000, "step": 1, 
                                     "tooltip": "Start step for whole denoising process"}),
                "end_step": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1, 
                                   "tooltip": "The step to stop applying TTM"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add_ttm_latents"
    CATEGORY = "PT/video"
    DESCRIPTION = "TTM (Time-to-Move) reference latents with strength control"

    def add_ttm_latents(self, embeds, reference_latents, mask, reference_strength, start_step, end_step):
        if end_step < max(0, start_step):
            raise ValueError(f"`end_step` ({end_step}) must be >= `start_step` ({start_step}).")

        # 获取参考潜变量
        ref_latents = reference_latents["samples"].squeeze(0)
        
        # 应用强度控制：通过线性插值在零潜变量和参考潜变量之间进行混合
        if reference_strength < 1.0:
            # 创建与参考潜变量相同形状的零张量
            zero_latents = torch.zeros_like(ref_latents)
            # 根据强度进行线性插值
            ref_latents = zero_latents + (ref_latents - zero_latents) * reference_strength
        
        # 处理mask
        VAE_STRIDE = (4, 8, 8)  # 与原始代码保持一致
        
        mask_sampled = mask[::VAE_STRIDE[0]]
        mask_sampled = mask_sampled.unsqueeze(1).unsqueeze(0)  # [1, T, 1, H, W]

        # 上采样到潜变量分辨率
        H_latent = mask_sampled.shape[-2] // VAE_STRIDE[1]
        W_latent = mask_sampled.shape[-1] // VAE_STRIDE[1]
        mask_latent = F.interpolate(
            mask_sampled.float(),
            size=(mask_sampled.shape[2], H_latent, W_latent),
            mode="nearest"
        )

        # 同样对mask应用强度控制
        if reference_strength < 1.0:
            mask_latent = mask_latent * reference_strength

        updated = dict(embeds)
        updated["ttm_reference_latents"] = ref_latents
        updated["ttm_mask"] = mask_latent.squeeze(0).movedim(1, 0)  # [T, 1, H, W]
        updated["ttm_start_step"] = start_step
        updated["ttm_end_step"] = end_step
        updated["ttm_reference_strength"] = reference_strength  # 保存强度值供采样器使用

        return (updated,)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "PT_TTM": PT_TTM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PT_TTM": "PT TTM Reference Control"
}