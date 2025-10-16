"""
ComfyUI Prompt Selector - Custom Node
A structured prompt generator for video generation
"""

from .prompt_selector import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 导出节点映射，供ComfyUI使用
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
