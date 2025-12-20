from .nodes.tile_splitter import CustomTileSplitter
from .nodes.tile_merger import CustomTileMerger

NODE_CLASS_MAPPINGS = {
    "CustomTileSplitter": CustomTileSplitter,
    "CustomTileMerger": CustomTileMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomTileSplitter": "🔧 Advanced Tile Splitter",
    "CustomTileMerger": "🔧 Advanced Tile Merger",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]