from __future__ import annotations

import torch
import comfy.utils

MAX_RESOLUTION = 8192

class PainterImageFromBatch:
    """
    Extract contiguous frames from an image batch with bidirectional support.
    Supports negative indexing and automatic boundary handling.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "start_from": (["beginning", "end"], {
                    "default": "beginning",
                    "tooltip": "Start counting from beginning or end of batch"
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": -4095,
                    "max": 4095,
                    "step": 1,
                    "tooltip": "Start frame index. beginning: 0=first frame; end: -1=last frame"
                }),
                "frame_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Number of frames to extract"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "from_batch"

    CATEGORY = "image/batch"
    
    DESCRIPTION = """
Extract contiguous frames from an image batch with bidirectional support.

- **beginning mode**: start_frame=0 means first frame
- **end mode**: start_frame=-1 means last frame

Examples (81 frames):
• Get frames 1-15: beginning, start_frame=0, frame_count=15
• Get last 15 frames: end, start_frame=-15, frame_count=15
• Get frames 66-81: beginning, start_frame=65, frame_count=16
"""

    def from_batch(self, image, start_from, start_frame, frame_count):
        batch_size = image.shape[0]
        
        # Calculate start index
        if start_from == "beginning":
            # Support negative indexing
            if start_frame < 0:
                start_idx = batch_size + start_frame
            else:
                start_idx = start_frame
        else:  # "end"
            # Negative values count from end
            if start_frame < 0:
                start_idx = batch_size + start_frame + 1
            else:
                # Positive values offset from end
                start_idx = batch_size - frame_count - start_frame
        
        # Calculate end index
        end_idx = start_idx + frame_count
        
        # Boundary protection
        start_idx = max(0, min(start_idx, batch_size - 1))
        end_idx = max(1, min(end_idx, batch_size))
        
        # Ensure valid range
        if start_idx >= end_idx:
            start_idx = max(0, batch_size - frame_count)
            end_idx = batch_size
        
        # Extract and clone frames
        result = image[start_idx:end_idx].clone()
        
        return (result,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "PainterImageFromBatch": PainterImageFromBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterImageFromBatch": "PainterImageFromBatch",
}
