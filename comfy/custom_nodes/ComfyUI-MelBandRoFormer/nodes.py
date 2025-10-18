import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

import librosa
import folder_paths

from .model.mel_band_roformer import MelBandRoformer

script_directory = os.path.dirname(os.path.abspath(__file__))

from comfy import model_management as mm
from comfy.utils import load_torch_file, ProgressBar
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

def get_windowing_array(window_size, fade_size, device):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window.to(device)

class MelBandRoFormerModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),
            },
        }

    RETURN_TYPES = ("MELROFORMERMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "Mel-Band RoFormer"

    def loadmodel(self, model_name):
        model_config = {
                "dim": 384,
                "depth": 6,
                "stereo": True,
                "num_stems": 1,
                "time_transformer_depth": 1,
                "freq_transformer_depth": 1,
                "num_bands": 60,
                "dim_head": 64,
                "heads": 8,
                "attn_dropout": 0,
                "ff_dropout": 0,
                "flash_attn": True,
                "dim_freqs_in": 1025,
                "sample_rate": 44100,  # needed for mel filter bank from librosa
                "stft_n_fft": 2048,
                "stft_hop_length": 441,
                "stft_win_length": 2048,
                "stft_normalized": False,
                "mask_estimator_depth": 2,
                "multi_stft_resolution_loss_weight": 1.0,
                "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
                "multi_stft_hop_size": 147,
                "multi_stft_normalized": False,
        }
        model = MelBandRoformer(**model_config).eval()
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        model.load_state_dict(load_torch_file(model_path), strict=True)

        return (model,)
    
class MelBandRoFormerSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MELROFORMERMODEL",),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO","AUDIO",)
    RETURN_NAMES = ("vocals", "instruments")
    FUNCTION = "process"
    CATEGORY = "Mel-Band RoFormer"

    def process(self, model, audio):

        audio_input = audio["waveform"]
        sample_rate = audio["sample_rate"]

        B, audio_channels, audio_length = audio_input.shape

        sr = 44100

        if audio_channels == 1:
            # Convert mono to stereo by duplicating the channel
            audio_input = audio_input.repeat(1, 2, 1)
            audio_channels = 2
            print("Converted mono input to stereo.")

        if sample_rate != sr:
            print(f"Resampling input {sample_rate} to {sr}")
            audio_np = audio_input.cpu().numpy()
            resampled = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=sr, axis=-1)
            audio_input = torch.from_numpy(resampled)
        audio_input = original_audio = audio_input[0]

        C = 352800
        N = 2
        step = C // N
        fade_size = C // 10
        border = C - step

        if audio_length > 2 * border and border > 0:
            audio_input = F.pad(audio_input, (border, border), mode='reflect')

        windowing_array = get_windowing_array(C, fade_size, device)


        audio_input = audio_input.to(device)
        vocals = torch.zeros_like(audio_input, dtype=torch.float32).to(device)
        counter = torch.zeros_like(audio_input, dtype=torch.float32).to(device)

        total_length = audio_input.shape[1]
        num_chunks = (total_length + step - 1) // step

        model.to(device)

        comfy_pbar = ProgressBar(num_chunks)

        for i in tqdm(range(0, total_length, step), desc="Processing chunks"):
            part = audio_input[:, i:i + C]
            length = part.shape[-1]
            if length < C:
                if length > C // 2 + 1:
                    part = F.pad(input=part, pad=(0, C - length), mode='reflect')
                else:
                    part = F.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)

            x = model(part.unsqueeze(0))[0]

            window = windowing_array.clone()
            if i == 0:
                window[:fade_size] = 1
            elif i + C >= total_length:
                window[-fade_size:] = 1

            vocals[..., i:i+length] += x[..., :length] * window[..., :length]
            counter[..., i:i+length] += window[..., :length]
            comfy_pbar.update(1)

        model.to(offload_device)

        estimated_sources = vocals / counter

        if audio_length > 2 * border and border > 0:
            estimated_sources = estimated_sources[..., border:-border]

        vocals_out = {
            "waveform": estimated_sources.unsqueeze(0).cpu(),
            "sample_rate": sr,
        }
        instruments_out = {
            "waveform": (original_audio.to(device) - estimated_sources).unsqueeze(0).cpu(),
            "sample_rate": sr,
        }

        return (vocals_out, instruments_out)

NODE_CLASS_MAPPINGS = {
    "MelBandRoFormerModelLoader": MelBandRoFormerModelLoader,
    "MelBandRoFormerSampler": MelBandRoFormerSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MelBandRoFormerModelLoader": "Mel-Band RoFormer Model Loader",
    "MelBandRoFormerSampler": "Mel-Band RoFormer Sampler",
}
