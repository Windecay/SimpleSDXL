from typing import Tuple

class PainterFrameCount:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_number": ("INT", {"default": 1, "min": 1, "max": 999999, "step": 1, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("aligned_frame",)
    FUNCTION = "align_frame"
    CATEGORY = "Prince"

    def align_frame(self, frame_number: int) -> Tuple[int]:
        remainder = (frame_number - 1) % 4
        
        if remainder == 0:
            aligned = frame_number
        else:
            aligned = frame_number + (4 - remainder)
        
        return (aligned,)

NODE_CLASS_MAPPINGS = {
    "PainterFrameCount": PainterFrameCount,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterFrameCount": "Painter Frame Count",
}
