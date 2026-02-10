import torch
import comfy.model_management
import comfy.utils
import node_helpers
import nodes


class PainterI2V:
    """
    An enhanced Wan2.2 Image-to-Video node specifically designed to fix the slow-motion issue in 4-step LoRAs (like lightx2v).
    Supports both start_image only and start_image + end_image modes.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "motion_amplitude": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                "start_image": ("IMAGE", ),
                "end_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size,
                motion_amplitude=1.15, start_image=None, end_image=None, clip_vision_output=None):
        # 1. Strict zero latent initialization (lifeline for 4-step LoRA)
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], 
                           device=comfy.model_management.intermediate_device())
        
        if start_image is not None:
            # Process start_image
            start_image = start_image[:1]
            start_image = comfy.utils.common_upscale(
                start_image.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            
            # Create sequence: first frame real, middle gray, last frame real (if end_image)
            image = torch.ones((length, height, width, start_image.shape[-1]), 
                             device=start_image.device, dtype=start_image.dtype) * 0.5
            image[0] = start_image[0]
            
            # Handle end_image if provided
            if end_image is not None:
                end_image = end_image[:1]
                end_image = comfy.utils.common_upscale(
                    end_image.movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
                image[-1] = end_image[0]
            
            concat_latent_image = vae.encode(image[:, :, :, :3])
            
            # Mask: constrain first frame (and last if end_image)
            # Mimic WanFirstLastFrameToVideo logic for better end frame handling
            mask = torch.ones((1, 1, latent.shape[2] * 4, concat_latent_image.shape[-2], 
                             concat_latent_image.shape[-1]), 
                            device=start_image.device, dtype=start_image.dtype)
            mask[:, :, :4] = 0.0  # First frame (all 4 sub-frames)
            if end_image is not None:
                mask[:, :, -1:] = 0.0  # Last frame (only last sub-frame)
            
            mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
            
            # 2. Motion amplitude enhancement (brightness protection core algorithm)
            if motion_amplitude > 1.0:
                if end_image is not None:
                    # Mode with end_image: scale middle frames between start and end
                    start_latent = concat_latent_image[:, :, 0:1]
                    middle_latent = concat_latent_image[:, :, 1:-1]
                    end_latent = concat_latent_image[:, :, -1:]
                    
                    # Calculate diffs from start
                    diff_start = middle_latent - start_latent
                    diff_mean = diff_start.mean(dim=(1, 3, 4), keepdim=True)
                    diff_centered = diff_start - diff_mean
                    scaled_latent = start_latent + diff_centered * motion_amplitude + diff_mean
                    scaled_latent = torch.clamp(scaled_latent, -6, 6)
                    
                    concat_latent_image = torch.cat([start_latent, scaled_latent, end_latent], dim=2)
                else:
                    # Original mode: start_image only
                    base_latent = concat_latent_image[:, :, 0:1]
                    gray_latent = concat_latent_image[:, :, 1:]
                    
                    diff = gray_latent - base_latent
                    diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
                    diff_centered = diff - diff_mean
                    scaled_latent = base_latent + diff_centered * motion_amplitude + diff_mean
                    scaled_latent = torch.clamp(scaled_latent, -6, 6)
                    
                    concat_latent_image = torch.cat([base_latent, scaled_latent], dim=2)
            
            # 3. Inject into conditioning
            positive = node_helpers.conditioning_set_values(
                positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )

            # 4. Reference frame enhancement
            ref_latents = [vae.encode(start_image[:, :, :, :3])]
            if end_image is not None:
                ref_latents.append(vae.encode(end_image[:, :, :, :3]))
            
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(
                negative, {"reference_latents": [torch.zeros_like(rl) for rl in ref_latents]}, append=True
            )

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)


class PainterI2VTiled:
    """
    An enhanced Wan2.2 Image-to-Video node with Tiled VAE encoding.
    Combines the slow-motion fix for 4-step LoRAs (like lightx2v) with tiled encoding
    to prevent memory issues during long video generation.
    Supports both start_image only and start_image + end_image modes.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "motion_amplitude": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05}),
                # Tiled VAE parameters
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, 
                                         "tooltip": "Amount of frames to encode at a time."}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, 
                                            "tooltip": "Amount of frames to overlap."}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                "start_image": ("IMAGE", ),
                "end_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size,
                motion_amplitude=1.15, tile_size=512, overlap=64, 
                temporal_size=64, temporal_overlap=8,
                start_image=None, end_image=None, clip_vision_output=None):
        # 1. Strict zero latent initialization (lifeline for 4-step LoRA)
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], 
                           device=comfy.model_management.intermediate_device())
        
        if start_image is not None:
            # Process start_image
            start_image = start_image[:1]
            start_image = comfy.utils.common_upscale(
                start_image.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            
            # Create sequence: first frame real, middle gray, last frame real (if end_image)
            image = torch.ones((length, height, width, start_image.shape[-1]), 
                             device=start_image.device, dtype=start_image.dtype) * 0.5
            image[0] = start_image[0]
            
            # Handle end_image if provided
            if end_image is not None:
                end_image = end_image[:1]
                end_image = comfy.utils.common_upscale(
                    end_image.movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
                image[-1] = end_image[0]
            
            # *** TILED VAE ENCODING (instead of standard vae.encode) ***
            concat_latent_image = vae.encode_tiled(
                image[:, :, :, :3],
                tile_x=tile_size,
                tile_y=tile_size,
                overlap=overlap,
                tile_t=temporal_size,
                overlap_t=temporal_overlap
            )
            
            # Mask: constrain first frame (and last if end_image)
            # Mimic WanFirstLastFrameToVideo logic for better end frame handling
            mask = torch.ones((1, 1, latent.shape[2] * 4, concat_latent_image.shape[-2], 
                             concat_latent_image.shape[-1]), 
                            device=start_image.device, dtype=start_image.dtype)
            mask[:, :, :4] = 0.0  # First frame (all 4 sub-frames)
            if end_image is not None:
                mask[:, :, -1:] = 0.0  # Last frame (only last sub-frame)
            
            mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
            
            # 2. Motion amplitude enhancement (brightness protection core algorithm)
            if motion_amplitude > 1.0:
                if end_image is not None:
                    # Mode with end_image: scale middle frames between start and end
                    start_latent = concat_latent_image[:, :, 0:1]
                    middle_latent = concat_latent_image[:, :, 1:-1]
                    end_latent = concat_latent_image[:, :, -1:]
                    
                    # Calculate diffs from start
                    diff_start = middle_latent - start_latent
                    diff_mean = diff_start.mean(dim=(1, 3, 4), keepdim=True)
                    diff_centered = diff_start - diff_mean
                    scaled_latent = start_latent + diff_centered * motion_amplitude + diff_mean
                    scaled_latent = torch.clamp(scaled_latent, -6, 6)
                    
                    concat_latent_image = torch.cat([start_latent, scaled_latent, end_latent], dim=2)
                else:
                    # Original mode: start_image only
                    base_latent = concat_latent_image[:, :, 0:1]
                    gray_latent = concat_latent_image[:, :, 1:]
                    
                    diff = gray_latent - base_latent
                    diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
                    diff_centered = diff - diff_mean
                    scaled_latent = base_latent + diff_centered * motion_amplitude + diff_mean
                    scaled_latent = torch.clamp(scaled_latent, -6, 6)
                    
                    concat_latent_image = torch.cat([base_latent, scaled_latent], dim=2)
            
            # 3. Inject into conditioning
            positive = node_helpers.conditioning_set_values(
                positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )

            # 4. Reference frame enhancement with TILED encoding
            ref_latents = [vae.encode_tiled(
                start_image[:, :, :, :3],
                tile_x=tile_size, tile_y=tile_size, overlap=overlap,
                tile_t=1, overlap_t=0
            )]
            
            if end_image is not None:
                ref_latents.append(vae.encode_tiled(
                    end_image[:, :, :, :3],
                    tile_x=tile_size, tile_y=tile_size, overlap=overlap,
                    tile_t=1, overlap_t=0
                ))
            
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(
                negative, {"reference_latents": [torch.zeros_like(rl) for rl in ref_latents]}, append=True
            )

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)


# Node registration mapping
NODE_CLASS_MAPPINGS = {
    "PainterI2VTiled": PainterI2VTiled,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterI2VTiled": "PainterI2V Tiled (Wan2.2 + VAE Tiled)",
}
