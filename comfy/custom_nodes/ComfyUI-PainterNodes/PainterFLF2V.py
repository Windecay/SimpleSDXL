import torch
import comfy.model_management as mm
import comfy.utils
import node_helpers
import torch.nn.functional as F
from comfy_api.latest import io, ComfyExtension
from typing_extensions import override

class PainterFLF2V(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PainterFLF2V",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=4096, step=16),
                io.Int.Input("height", default=480, min=16, max=4096, step=16),
                io.Int.Input("length", default=81, min=1, max=4096, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Float.Input("motion_amplitude", default=1.0, min=1.0, max=2.0, step=0.05),
                io.ClipVisionOutput.Input("clip_vision_start_image", optional=True),
                io.ClipVisionOutput.Input("clip_vision_end_image", optional=True),
                io.Image.Input("start_image", optional=True),
                io.Image.Input("end_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size,
                motion_amplitude=1.0,
                start_image=None, end_image=None,
                clip_vision_start_image=None, clip_vision_end_image=None) -> io.NodeOutput:

        spacial_scale = vae.spacial_compression_encode()
        latent_frames = ((length - 1) // 4) + 1
        
        latent = torch.zeros([batch_size, vae.latent_channels, latent_frames, height // spacial_scale, width // spacial_scale], 
                             device=mm.intermediate_device())

        if start_image is not None:
            start_image = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        if end_image is not None:
            end_image = comfy.utils.common_upscale(
                end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

        official_image = torch.ones((length, height, width, 3), device=mm.intermediate_device()) * 0.5
        mask = torch.ones((1, 1, latent_frames * 4, height // spacial_scale, width // spacial_scale), device=mm.intermediate_device())

        if start_image is not None:
            official_image[:start_image.shape[0]] = start_image
            mask[:, :, :start_image.shape[0] + 3] = 0.0
        if end_image is not None:
            official_image[-end_image.shape[0]:] = end_image
            mask[:, :, -end_image.shape[0]:] = 0.0
            
        official_latent = vae.encode(official_image[:, :, :, :3])

        if start_image is not None and end_image is not None and length > 2:
            start_l = official_latent[:, :, 0:1]
            end_l   = official_latent[:, :, -1:]
            t = torch.linspace(0.0, 1.0, official_latent.shape[2], device=official_latent.device).view(1, 1, -1, 1, 1)
            linear_latent = start_l * (1 - t) + end_l * t
        else:
            linear_latent = official_latent 

        if length > 2 and motion_amplitude > 1.001 and start_image is not None and end_image is not None:
            diff = official_latent - linear_latent
            
            h, w = diff.shape[-2], diff.shape[-1]
            low_freq_diff = F.interpolate(diff.view(-1, vae.latent_channels, h, w), 
                                         size=(h // 8, w // 8), mode='area')
            low_freq_diff = F.interpolate(low_freq_diff, size=(h, w), mode='bilinear')
            low_freq_diff = low_freq_diff.view_as(diff)
            
            high_freq_diff = diff - low_freq_diff
            boost_scale = (motion_amplitude - 1.0) * 4.0
            concat_latent_image = official_latent + (high_freq_diff * boost_scale)
        else:
            concat_latent_image = official_latent

        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)

        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )

        clip_vision_output = None
        if clip_vision_start_image is not None:
            clip_vision_output = clip_vision_start_image

        if clip_vision_end_image is not None:
            if clip_vision_output is not None:
                states = torch.cat([clip_vision_output.penultimate_hidden_states, 
                                   clip_vision_end_image.penultimate_hidden_states], dim=-2)
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states
            else:
                clip_vision_output = clip_vision_end_image

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {"samples": latent}
        return io.NodeOutput(positive, negative, out_latent)


class PainterFLF2VExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PainterFLF2V]

async def comfy_entrypoint() -> PainterFLF2VExtension:
    return PainterFLF2VExtension()

NODE_CLASS_MAPPINGS = {"PainterFLF2V": PainterFLF2V}
NODE_DISPLAY_NAME_MAPPINGS = {"PainterFLF2V": "PainterFLF2V"}
