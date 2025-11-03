from .openpose_editor_nodes import OpenposeEditorNode
from .appendage_editor_nodes import AppendageEditorNode


WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "OpenposeEditorNode": OpenposeEditorNode,
    "AppendageEditorNode": AppendageEditorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenposeEditorNode": "Openpose Editor Node",
    "AppendageEditorNode": "Appendage Editor",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
