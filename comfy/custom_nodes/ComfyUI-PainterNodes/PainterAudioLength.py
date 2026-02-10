import os
import torchaudio
import torch
import folder_paths

class PainterAudioLength:
    """
    Input:  ComfyUI audio dict { "sample_rate": int, "waveform": torch.Tensor }
    Output: float – duration in seconds
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("duration_seconds",)
    FUNCTION = "get_length"
    CATEGORY = "audio"

    def get_length(self, audio):
        if audio is None:
            return (0.0,)
        waveform = audio["waveform"]          # shape: (1, channels, length)
        sr       = audio["sample_rate"]
        length   = waveform.shape[-1]
        duration = length / sr
        return (round(duration, 2),)

NODE_CLASS_MAPPINGS = {
    "PainterAudioLength": PainterAudioLength,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterAudioLength": "Painter Audio Length",
}
