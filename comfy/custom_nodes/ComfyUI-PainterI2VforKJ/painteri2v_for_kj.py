import torch
import torch.nn.functional as F
import math
from .wanvideo_utils import add_noise_to_reference_video, VAE_STRIDE, PATCH_SIZE

# ComfyUI 核心依赖
import comfy.model_management as mm
from comfy.utils import common_upscale

class PainterI2VforKJ:
    """
    完全独立的 PainterI2V 节点 for KJ 工作流
    - 不依赖 ComfyUI-WanVideoWrapper 插件
    - 直接实现核心编码逻辑
    - 保持与KJ工作流100%兼容
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_amplitude": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05, 
                                             "tooltip": "核心参数：动态幅度增强系数，>1.0增强运动减少慢动作，1.0=禁用"}),
                "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "输出视频宽度"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "输出视频高度"}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "总帧数"}),
                "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "起始帧噪声增强强度"}),
                "start_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "起始帧latent强度"}),
                "end_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "结束帧latent强度"}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "处理完成后卸载VAE到CPU"}),
            },
            "optional": {
                "vae": ("WANVAE", {"tooltip": "WanVideo VAE模型"}),
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "CLIP视觉嵌入（来自ClipVisionEncode）"}),
                "start_image": ("IMAGE", {"tooltip": "起始帧图像，I2V必需"}),
                "end_image": ("IMAGE", {"tooltip": "结束帧图像（可选）"}),
                "control_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "控制信号嵌入（Fun模型）"}),
                "fun_or_fl2v_model": ("BOOLEAN", {"default": True, "tooltip": "使用官方FLF2V或Fun模型时启用"}),
                "temporal_mask": ("MASK", {"tooltip": "时间掩码，控制每帧权重"}),
                "extra_latents": ("LATENT", {"tooltip": "额外latent（如Skyreels A2参考图）"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "使用分块VAE编码（省显存）"}),
                "add_cond_latents": ("ADD_COND_LATENTS", {"advanced": True, "tooltip": "WanVideo额外条件latent"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper/PainterI2V"
    
    OUTPUT_NODE = False

    def process(self, width, height, num_frames, force_offload, noise_aug_strength,
                start_latent_strength, end_latent_strength, motion_amplitude=1.15,
                start_image=None, end_image=None, control_embeds=None, fun_or_fl2v_model=False,
                temporal_mask=None, extra_latents=None, clip_embeds=None, tiled_vae=False, add_cond_latents=None, vae=None):
        
        if start_image is None and end_image is None and add_cond_latents is None:
            return self.create_empty_embeds(num_frames, width, height, control_embeds, extra_latents)
        
        if vae is None:
            raise ValueError("❌ VAE模型未提供，请连接WANVAE输入")
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        H, W = height, width
        lat_h, lat_w = H // vae.upsampling_factor, W // vae.upsampling_factor

        num_frames = ((num_frames - 1) // 4) * 4 + 1
        two_ref_images = start_image is not None and end_image is not None

        if start_image is None and end_image is not None:
            fun_or_fl2v_model = True

        base_frames = num_frames + (1 if two_ref_images and not fun_or_fl2v_model else 0)
        
        # 创建时间掩码
        mask = self.create_temporal_mask(temporal_mask, base_frames, lat_h, lat_w, start_image, end_image, device, vae.dtype)

        # 编码图像序列
        vae.to(device)
        concatenated = self.prepare_image_sequence(
            vae, device, start_image, end_image, H, W, num_frames, 
            noise_aug_strength, temporal_mask, fun_or_fl2v_model
        )
        
        # 执行VAE编码 -> y形状: [C, T, H, W] (4维)
        y = vae.encode([concatenated], device, end_=(end_image is not None and not fun_or_fl2v_model), tiled=tiled_vae)[0]
        del concatenated

        # 处理额外latent
        has_ref = False
        if extra_latents is not None:
            samples = extra_latents["samples"].squeeze(0)
            y = torch.cat([samples, y], dim=1)
            mask = torch.cat([torch.ones_like(mask[:, 0:samples.shape[1]]), mask], dim=1)
            num_frames += samples.shape[1] * 4
            has_ref = True
        
        # 应用强度系数
        y[:, :1] *= start_latent_strength
        if y.shape[1] > 1:
            y[:, -1:] *= end_latent_strength

        # ==================== PainterI2V 动态增强核心算法 ====================
        if motion_amplitude > 1.0 and y.shape[1] > 1:
            print(f"\n🎨 [PainterI2V] 应用动态增强: amplitude={motion_amplitude:.2f}")
            
            base_latent = y[:, 0:1]      # [C, 1, H, W]
            other_latent = y[:, 1:]      # [C, T-1, H, W]
            
            # 广播首帧
            base_latent_bc = base_latent.expand(-1, other_latent.shape[1], -1, -1)
            
            # 计算差异并增强（保持亮度稳定）
            diff = other_latent - base_latent_bc
            diff_mean = diff.mean(dim=(0, 2, 3), keepdim=True)
            diff_centered = diff - diff_mean
            scaled_other = base_latent_bc + diff_centered * motion_amplitude + diff_mean
            
            # 安全裁剪
            scaled_other = torch.clamp(scaled_other, -6, 6)
            
            # 重组
            y = torch.cat([base_latent, scaled_other], dim=1)
            print("✅ 动态增强完成\n")
        # ==================== 动态增强结束 ====================

        # 计算序列长度
        patches_per_frame = lat_h * lat_w // (PATCH_SIZE[1] * PATCH_SIZE[2])
        frames_per_stride = (num_frames - 1) // 4 + (2 if end_image is not None and not fun_or_fl2v_model else 1)
        max_seq_len = frames_per_stride * patches_per_frame

        if add_cond_latents is not None:
            add_cond_latents["ref_latent_neg"] = vae.encode(torch.zeros(1, 3, 1, H, W, device=device, dtype=vae.dtype), device)
        
        if force_offload:
            vae.model.to(offload_device)
            mm.soft_empty_cache()

        # 构建输出
        image_embeds = {
            "image_embeds": y,
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "negative_clip_context": clip_embeds.get("negative_clip_embeds", None) if clip_embeds is not None else None,
            "max_seq_len": max_seq_len,
            "num_frames": num_frames,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "control_embeds": control_embeds["control_embeds"] if control_embeds is not None else None,
            "end_image": end_image if end_image is not None else None,
            "fun_or_fl2v_model": fun_or_fl2v_model,
            "has_ref": has_ref,
            "add_cond_latents": add_cond_latents,
            "mask": mask
        }

        return (image_embeds,)
    
    def create_temporal_mask(self, temporal_mask, base_frames, lat_h, lat_w, start_image, end_image, device, dtype):
        """创建并处理时间掩码"""
        if temporal_mask is None:
            mask = torch.zeros(1, base_frames, lat_h, lat_w, device=device, dtype=dtype)
            if start_image is not None:
                mask[:, 0:start_image.shape[0]] = 1.0
            if end_image is not None:
                mask[:, -end_image.shape[0]:] = 1.0
        else:
            mask = common_upscale(temporal_mask.unsqueeze(1).to(device), lat_w, lat_h, "nearest", "disabled").squeeze(1)
            if mask.shape[0] > base_frames:
                mask = mask[:base_frames]
            elif mask.shape[0] < base_frames:
                mask = torch.cat([mask, torch.zeros(base_frames - mask.shape[0], lat_h, lat_w, device=device)])
            mask = mask.unsqueeze(0).to(device, dtype)

        # 重复掩码
        start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
        
        # 修复：根据是否有首尾帧和fun_or_fl2v_model来决定拼接逻辑
        if end_image is not None and not (start_image is None and end_image is not None):
            # 正常情况：有尾帧且不是只有尾帧的情况
            end_mask_repeated = torch.repeat_interleave(mask[:, -1:], repeats=4, dim=1)
            mask = torch.cat([start_mask_repeated, mask[:, 1:-1], end_mask_repeated], dim=1)
        else:
            # 只有尾帧或其他情况
            mask = torch.cat([start_mask_repeated, mask[:, 1:]], dim=1)

        mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w)
        return mask.movedim(1, 2)[0]
    
    def create_empty_embeds(self, num_frames, width, height, control_embeds=None, extra_latents=None):
        """创建空嵌入"""
        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        
        embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
            "control_embeds": control_embeds["control_embeds"] if control_embeds is not None else None,
        }
        if extra_latents is not None:
            embeds["extra_latents"] = [{
                "samples": extra_latents["samples"],
                "index": 0,
            }]
        return (embeds,)
    
    def prepare_image_sequence(self, vae, device, start_image, end_image, H, W, num_frames, 
                               noise_aug_strength, temporal_mask, fun_or_fl2v_model):
        """准备图像序列"""
        C = 3
        
        if start_image is not None:
            start_image = start_image[..., :3]
            if start_image.shape[1] != H or start_image.shape[2] != W:
                resized_start = common_upscale(start_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_start = start_image.permute(3, 0, 1, 2)
            resized_start = resized_start * 2.0 - 1.0
            if noise_aug_strength > 0.0:
                resized_start = add_noise_to_reference_video(resized_start, noise_aug_strength)
            T_start = resized_start.shape[1]
        else:
            resized_start, T_start = None, 0
        
        if end_image is not None:
            end_image = end_image[..., :3]
            if end_image.shape[1] != H or end_image.shape[2] != W:
                resized_end = common_upscale(end_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_end = end_image.permute(3, 0, 1, 2)
            resized_end = resized_end * 2.0 - 1.0
            if noise_aug_strength > 0.0:
                resized_end = add_noise_to_reference_video(resized_end, noise_aug_strength)
            T_end = resized_end.shape[1]
        else:
            resized_end, T_end = None, 0
        
        # 拼接
        if temporal_mask is None:
            if start_image is not None and end_image is None:
                zero_frames = torch.zeros(C, num_frames - T_start, H, W, device=device, dtype=vae.dtype)
                concatenated = torch.cat([resized_start.to(device, dtype=vae.dtype), zero_frames], dim=1)
            elif start_image is None and end_image is not None:
                zero_frames = torch.zeros(C, num_frames - T_end, H, W, device=device, dtype=vae.dtype)
                concatenated = torch.cat([zero_frames, resized_end.to(device, dtype=vae.dtype)], dim=1)
            elif start_image is None and end_image is None:
                concatenated = torch.zeros(C, num_frames, H, W, device=device, dtype=vae.dtype)
            else:
                if fun_or_fl2v_model:
                    zero_frames = torch.zeros(C, num_frames - (T_start + T_end), H, W, device=device, dtype=vae.dtype)
                else:
                    zero_frames = torch.zeros(C, num_frames - 1, H, W, device=device, dtype=vae.dtype)
                concatenated = torch.cat([resized_start.to(device, dtype=vae.dtype), zero_frames, resized_end.to(device, dtype=vae.dtype)], dim=1)
        else:
            temporal_mask = common_upscale(temporal_mask.unsqueeze(1), W, H, "nearest", "disabled").squeeze(1)
            concatenated = resized_start[:, :num_frames].to(vae.dtype)
        
        return concatenated
