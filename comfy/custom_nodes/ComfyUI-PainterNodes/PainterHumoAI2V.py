import math
import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.latent_formats
from comfy_api.latest import io


def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = torch.nn.functional.interpolate(
        features, size=output_len, align_corners=True,
        mode='linear')
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


class PainterHumoAI2V(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PainterHumoAI2V",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=97, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Float.Input("fps", default=25.0, min=1.0, max=120.0, step=0.1),
                io.AudioEncoderOutput.Input("audio_encoder", optional=True),
                io.Image.Input("start_image", optional=True),
                io.Image.Input("end_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output("high_positive"),
                io.Conditioning.Output("high_negative"),
                io.Conditioning.Output("low_positive"),
                io.Conditioning.Output("low_negative"),
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, fps, start_image=None, end_image=None, audio_encoder=None):
        spacial_scale = vae.spacial_compression_encode()
        latent = torch.zeros([batch_size, vae.latent_channels, ((length - 1) // 4) + 1, height // spacial_scale, width // spacial_scale], device=comfy.model_management.intermediate_device())
        
        if start_image is not None:
            start_image_up = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        else:
            start_image_up = None
            
        if end_image is not None:
            end_image_up = comfy.utils.common_upscale(end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        else:
            end_image_up = None

        image = torch.ones((length, height, width, 3)) * 0.5
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

        if start_image_up is not None:
            image[:start_image_up.shape[0]] = start_image_up
            mask[:, :, :start_image_up.shape[0] + 3] = 0.0

        if end_image_up is not None:
            image[-end_image_up.shape[0]:] = end_image_up
            mask[:, :, -end_image_up.shape[0]:] = 0.0

        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        high_positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        high_negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        latent_t = ((length - 1) // 4) + 1
        has_ref = False
        
        if start_image is not None:
            start_image_ref = comfy.utils.common_upscale(start_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            start_latent = vae.encode(start_image_ref[:, :, :, :3])
            low_positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [start_latent]}, append=True)
            low_negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(start_latent)]}, append=True)
            has_ref = True
        
        if end_image is not None:
            end_image_ref = comfy.utils.common_upscale(end_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            end_latent = vae.encode(end_image_ref[:, :, :, :3])
            if has_ref:
                low_positive = node_helpers.conditioning_set_values(low_positive, {"reference_latents": [end_latent]}, append=True)
                low_negative = node_helpers.conditioning_set_values(low_negative, {"reference_latents": [torch.zeros_like(end_latent)]}, append=True)
            else:
                low_positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [end_latent]}, append=True)
                low_negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(end_latent)]}, append=True)
            has_ref = True
        
        if not has_ref:
            zero_latent = torch.zeros([batch_size, 16, 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
            low_positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [zero_latent]}, append=True)
            low_negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [zero_latent]}, append=True)

        if audio_encoder is not None:
            audio_emb = torch.stack(audio_encoder["encoded_audio_all_layers"], dim=2)
            audio_len = audio_encoder["audio_samples"] // 640
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
            low_positive = node_helpers.conditioning_set_values(low_positive, {"audio_embed": audio_emb})
            low_negative = node_helpers.conditioning_set_values(low_negative, {"audio_embed": audio_emb_neg})
        else:
            zero_audio = torch.zeros([batch_size, latent_t + 1, 8, 5, 1280], device=comfy.model_management.intermediate_device())
            low_positive = node_helpers.conditioning_set_values(low_positive, {"audio_embed": zero_audio})
            low_negative = node_helpers.conditioning_set_values(low_negative, {"audio_embed": zero_audio})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(high_positive, high_negative, low_positive, low_negative, out_latent)


NODE_CLASS_MAPPINGS = {
    "PainterHumoAI2V": PainterHumoAI2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterHumoAI2V": "Painter Humo AI2V",
}
