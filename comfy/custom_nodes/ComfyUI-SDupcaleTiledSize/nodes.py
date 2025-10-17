import math
import re
import torch
from PIL import Image

def uov_tiled_size(width, height, upscale_by, tiled_block=2048):
    tiled_size = lambda x, p: int(x*p) if int(x*p) < tiled_block else int(int(x*p)/math.ceil(int(x*p)/tiled_block))
    upscale_by = upscale_by if upscale_by < 4.0 else 4.0
    tiled_width = ((tiled_size(width, upscale_by) + 15) // 16) * 16
    tiled_height = ((tiled_size(height, upscale_by) + 15) // 16) * 16
    return upscale_by, tiled_width, tiled_height

class SDupscaleTiledSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tiled_block": ("INT", {"default": 1536, "min": 512, "max": 2048, "step": 256}),
                "upscale_by": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 4.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("output_image", "upscale_by", "tiled_width", "tiled_height")
    FUNCTION = "calculate_tiled_size"
    CATEGORY = "image/upscaling"

    def calculate_tiled_size(self, image, tiled_block, upscale_by):

        height = image.shape[1]
        width = image.shape[2]

        upscale_by, tiled_width, tiled_height = uov_tiled_size(
            width, height, upscale_by, tiled_block
        )

        return (image, upscale_by, tiled_width, tiled_height)

NODE_CLASS_MAPPINGS = {
    "SDupscaleTiledSize": SDupscaleTiledSize
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDupscaleTiledSize": "SDupscaleTiledSize"
}