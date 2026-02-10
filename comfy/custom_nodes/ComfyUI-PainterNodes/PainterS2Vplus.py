import math
import torch
import nodes
import node_helpers
import comfy.model_management
import comfy.utils
import numpy as np
from comfy_api.latest import io


def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = torch.nn.functional.interpolate(
        features, size=output_len, align_corners=True, mode='linear'
    )
    return output_features.transpose(1, 2)


def get_sample_indices(original_fps, total_frames, target_fps, num_sample, fixed_start=None):
    required_duration = num_sample / target_fps
    required_origin_frames = int(np.ceil(required_duration * original_fps))
    if required_duration > total_frames / original_fps:
        raise ValueError("video length is too short")
    
    if fixed_start is not None and fixed_start >= 0:
        start_frame = fixed_start
    else:
        max_start = total_frames - required_origin_frames
        if max_start < 0:
            raise ValueError("video length is too short")
        start_frame = np.random.randint(0, max_start + 1)
    
    start_time = start_frame / original_fps
    end_time = start_time + required_duration
    time_points = np.linspace(start_time, end_time, num_sample, endpoint=False)
    frame_indices = np.round(np.array(time_points) * original_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    return frame_indices


def get_audio_embed_bucket_fps(audio_embed, fps=16, batch_frames=81, m=0, video_rate=30):
    num_layers, audio_frame_num, audio_dim = audio_embed.shape
    return_all_layers = num_layers > 1
    
    scale = video_rate / fps
    min_batch_num = int(audio_frame_num / (batch_frames * scale)) + 1
    bucket_num = min_batch_num * batch_frames
    padd_audio_num = math.ceil(min_batch_num * batch_frames / fps * video_rate) - audio_frame_num
    
    batch_idx = get_sample_indices(
        original_fps=video_rate,
        total_frames=audio_frame_num + padd_audio_num,
        target_fps=fps,
        num_sample=bucket_num,
        fixed_start=0
    )
    
    batch_audio_eb = []
    audio_sample_stride = int(video_rate / fps)
    
    for bi in batch_idx:
        if bi < audio_frame_num:
            chosen_idx = list(range(
                bi - m * audio_sample_stride, 
                bi + (m + 1) * audio_sample_stride, 
                audio_sample_stride
            ))
            chosen_idx = [0 if c < 0 else c for c in chosen_idx]
            chosen_idx = [audio_frame_num - 1 if c >= audio_frame_num else c for c in chosen_idx]
            
            if return_all_layers:
                frame_audio_embed = audio_embed[:, chosen_idx].flatten(start_dim=-2, end_dim=-1)
            else:
                frame_audio_embed = audio_embed[0][chosen_idx].flatten()
        else:
            if return_all_layers:
                frame_audio_embed = torch.zeros([num_layers, audio_dim * (2 * m + 1)], device=audio_embed.device)
            else:
                frame_audio_embed = torch.zeros([audio_dim * (2 * m + 1)], device=audio_embed.device)
        batch_audio_eb.append(frame_audio_embed)
    
    batch_audio_eb = torch.cat([c.unsqueeze(0) for c in batch_audio_eb], dim=0)
    return batch_audio_eb, min_batch_num


class PainterS2Vplus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PainterS2Vplus",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Image.Input("video"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Float.Input("video_fps", default=16.0, min=8.0, max=60.0, step=0.1),
                io.Float.Input("audio_scale", default=1.0, min=0.1, max=20.0, step=0.1),
                io.Int.Input("motion_frame_count", default=9, min=0, max=33, step=1),
                io.AudioEncoderOutput.Input("audio_encoder_output", optional=True),
                io.Image.Input("start_image", optional=True),
                io.Image.Input("previous_frames", optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Int.Output(display_name="trim_image"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, video, width, height, length, video_fps,
                audio_scale, motion_frame_count, audio_encoder_output=None,
                start_image=None, previous_frames=None) -> io.NodeOutput:
        
        # Independent VAE Encoding for video input
        if video is not None:
            video_proc = comfy.utils.common_upscale(
                video.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            encoded_latent = vae.encode(video_proc[:, :, :, :3])
            out_latent = {"samples": encoded_latent}
        else:
            latent_t = ((length - 1) // 4) + 1
            encoded_latent = torch.zeros(
                [1, 16, latent_t, height // 8, width // 8], 
                device=comfy.model_management.intermediate_device()
            )
            out_latent = {"samples": encoded_latent}
        
        trim_image = 0
        
        # Start image as reference and first frame constraint
        if start_image is not None:
            ref_img_proc = comfy.utils.common_upscale(
                start_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            ref_latent = vae.encode(ref_img_proc[:, :, :, :3])
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_latents": [ref_latent]}, append=True
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"reference_latents": [ref_latent]}, append=True
            )
            
            latent_t = ((length - 1) // 4) + 1
            image = torch.ones((length, height, width, 3), device=comfy.model_management.intermediate_device()) * 0.5
            mask = torch.ones((1, 1, latent_t * 4, height // 8, width // 8), device=comfy.model_management.intermediate_device())
            
            start_proc = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            image[:start_proc.shape[0]] = start_proc
            mask[:, :, :start_proc.shape[0] + 3] = 0.0
            
            concat_latent_image = vae.encode(image[:, :, :, :3])
            mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
            positive = node_helpers.conditioning_set_values(
                positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )
        
        # Motion frames
        if previous_frames is not None and motion_frame_count > 0:
            actual_motion_frames = min(previous_frames.shape[0], motion_frame_count)
            motion_frames = previous_frames[-actual_motion_frames:]
            motion_frames = comfy.utils.common_upscale(
                motion_frames.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            
            if motion_frames.shape[0] < 73:
                padding = torch.ones(
                    [73 - motion_frames.shape[0], height, width, 3], 
                    device=motion_frames.device, dtype=motion_frames.dtype
                ) * 0.5
                motion_frames = torch.cat([padding, motion_frames], dim=0)
            
            motion_latent = vae.encode(motion_frames[:, :, :, :3])
            motion_latent = motion_latent[:, :, -19:]
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_motion": motion_latent}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"reference_motion": motion_latent}
            )
            trim_image = actual_motion_frames
        
        # Audio processing with configurable fps
        if audio_encoder_output is not None:
            feat = torch.cat(audio_encoder_output["encoded_audio_all_layers"])
            feat = linear_interpolation(feat, input_fps=50, output_fps=video_fps)
            
            latent_t = ((length - 1) // 4) + 1
            batch_frames = latent_t * 4
            audio_embed_bucket, _ = get_audio_embed_bucket_fps(
                feat, fps=video_fps, batch_frames=batch_frames, m=0, video_rate=video_fps
            )
            audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
            
            if len(audio_embed_bucket.shape) == 3:
                audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
            elif len(audio_embed_bucket.shape) == 4:
                audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
            
            if audio_embed_bucket.shape[-1] > 0 if len(audio_embed_bucket.shape) == 3 else audio_embed_bucket.shape[3] > 0:
                scaled_audio = audio_embed_bucket * (audio_scale ** 1.5)
                positive = node_helpers.conditioning_set_values(
                    positive, {"audio_embed": scaled_audio}
                )
                negative = node_helpers.conditioning_set_values(
                    negative, {"audio_embed": torch.zeros_like(scaled_audio)}
                )
        
        return io.NodeOutput(out_latent, positive, negative, trim_image)

NODE_CLASS_MAPPINGS = {
    "PainterS2Vplus": PainterS2Vplus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterS2Vplus": "Painter S2V Plus"
}
