"""
ComfyUI-qwenmultiangle: A 3D camera angle control node for ComfyUI
Outputs camera angle prompts for multi-angle image generation
"""

WEB_DIRECTORY = "./web"

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
