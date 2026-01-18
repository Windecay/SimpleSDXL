"""
ComfyUI-QwenMultiangle-Lightning: A 3D lighting angle control node for ComfyUI
Outputs lighting prompts for relighting image generation

Authors: aiwood, wallen
Thanks to: https://github.com/jtydhr88
"""

import os
WEB_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
