# PainterVideoUpscale.py
import torch
import comfy.utils
import comfy.model_management
import node_helpers


class PainterVideoUpscale:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "video": ("IMAGE",),
                "upscale_method": (s.upscale_methods,),
                "width": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 1}),
                "length": ("INT", {"default": 81, "min": 1, "max": 16384, "step": 4}),
                "crop": (s.crop_methods,),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "Painter/Video"
    
    def execute(self, positive, negative, vae, video, upscale_method, width, height, length, crop, start_image=None, end_image=None):
        if width == 0 and height == 0:
            scaled_video = video
        else:
            samples = video.movedim(-1, 1)
            
            if width == 0:
                width = max(1, round(samples.shape[3] * height / samples.shape[2]))
            elif height == 0:
                height = max(1, round(samples.shape[2] * width / samples.shape[3]))
            
            s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
            scaled_video = s.movedim(1, -1)
        
        t = vae.encode(scaled_video)
        video_latent = {"samples": t}
        
        spacial_scale = vae.spacial_compression_encode()
        latent = torch.zeros([1, vae.latent_channels, ((length - 1) // 4) + 1, height // spacial_scale, width // spacial_scale], device=comfy.model_management.intermediate_device())
        
        if start_image is not None:
            start_image_proc = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        else:
            start_image_proc = None
            
        if end_image is not None:
            end_image_proc = comfy.utils.common_upscale(end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        else:
            end_image_proc = None
        
        image = torch.ones((length, height, width, 3)) * 0.5
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))
        
        if start_image_proc is not None:
            image[:start_image_proc.shape[0]] = start_image_proc
            mask[:, :, :start_image_proc.shape[0] + 3] = 0.0
        
        if end_image_proc is not None:
            image[-end_image_proc.shape[0]:] = end_image_proc
            mask[:, :, -end_image_proc.shape[0]:] = 0.0
        
        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        
        positive_out = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative_out = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        
        return (positive_out, negative_out, video_latent)


NODE_CLASS_MAPPINGS = {
    "PainterVideoUpscale": PainterVideoUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterVideoUpscale": "Painter Video Upscale",
}
