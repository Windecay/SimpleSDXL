import torch
import nodes
import node_helpers
import comfy.model_management
import comfy.utils


def linear_interpolation(features, input_fps, output_fps, output_len=None):
    """
    features: shape=[1, T, 512]
    input_fps: fps for audio, f_a
    output_fps: fps for video, f_m
    output_len: video length
    """
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = torch.nn.functional.interpolate(
        features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


def get_audio_emb_window(audio_emb, frame_num, frame0_idx, audio_shift=2):
    zero_audio_embed = torch.zeros((audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)
    zero_audio_embed_3 = torch.zeros((3, audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)
    iter_ = 1 + (frame_num - 1) // 4
    audio_emb_wind = []
    for lt_i in range(iter_):
        if lt_i == 0:
            st = frame0_idx + lt_i - 2
            ed = frame0_idx + lt_i + 3
            wind_feat = torch.stack([
                audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                for i in range(st, ed)
            ], dim=0)
            wind_feat = torch.cat((zero_audio_embed_3, wind_feat), dim=0)
        else:
            st = frame0_idx + 1 + 4 * (lt_i - 1) - audio_shift
            ed = frame0_idx + 1 + 4 * lt_i + audio_shift
            wind_feat = torch.stack([
                audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                for i in range(st, ed)
            ], dim=0)
        audio_emb_wind.append(wind_feat)
    audio_emb_wind = torch.stack(audio_emb_wind, dim=0)
    return audio_emb_wind, ed - audio_shift


class PainterHumoAV2V:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "video": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 97, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0, "step": 0.5}),
            },
            "optional": {
                "audio_encoder_output": ("AUDIO_ENCODER_OUTPUT",),
                "ref_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    def execute(self, positive, negative, vae, video, width, height, length, fps, audio_encoder_output=None, ref_image=None):
        batch_size = 1
        
        # ImageScale functionality: resize video frames using nearest-exact, no crop
        video_resized = comfy.utils.common_upscale(
            video.movedim(-1, 1),
            width,
            height,
            "nearest-exact",
            "disabled"
        ).movedim(1, -1)
        
        # VAEEncode functionality: encode resized video to latent
        video_latent = vae.encode(video_resized)
        out_latent = {"samples": video_latent}
        
        # PainterHuMoToVideo functionality: process conditioning independently
        if ref_image is not None:
            ref_image_proc = comfy.utils.common_upscale(
                ref_image[:1].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center"
            ).movedim(1, -1)
            ref_latent = vae.encode(ref_image_proc[:, :, :, :3])
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)
        else:
            zero_latent = torch.zeros([batch_size, 16, 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [zero_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [zero_latent]}, append=True)

        if audio_encoder_output is not None:
            audio_emb = torch.stack(audio_encoder_output["encoded_audio_all_layers"], dim=2)
            audio_len = audio_encoder_output["audio_samples"] // 640
            audio_emb = audio_emb[:, :audio_len * 2]

            feat0 = linear_interpolation(audio_emb[:, :, 0: 8].mean(dim=2), 50, fps)
            feat1 = linear_interpolation(audio_emb[:, :, 8: 16].mean(dim=2), 50, fps)
            feat2 = linear_interpolation(audio_emb[:, :, 16: 24].mean(dim=2), 50, fps)
            feat3 = linear_interpolation(audio_emb[:, :, 24: 32].mean(dim=2), 50, fps)
            feat4 = linear_interpolation(audio_emb[:, :, 32], 50, fps)
            audio_emb = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]
            
            audio_emb, _ = get_audio_emb_window(audio_emb, length, frame0_idx=0)
            audio_emb = audio_emb.unsqueeze(0)
            audio_emb_neg = torch.zeros_like(audio_emb)
            positive = node_helpers.conditioning_set_values(positive, {"audio_embed": audio_emb})
            negative = node_helpers.conditioning_set_values(negative, {"audio_embed": audio_emb_neg})
        else:
            latent_t = ((length - 1) // 4) + 1
            zero_audio = torch.zeros([batch_size, latent_t + 1, 8, 5, 1280], device=comfy.model_management.intermediate_device())
            positive = node_helpers.conditioning_set_values(positive, {"audio_embed": zero_audio})
            negative = node_helpers.conditioning_set_values(negative, {"audio_embed": zero_audio})

        return (positive, negative, out_latent)


NODE_CLASS_MAPPINGS = {
    "PainterHumoAV2V": PainterHumoAV2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterHumoAV2V": "Painter Humo AV2V",
}
