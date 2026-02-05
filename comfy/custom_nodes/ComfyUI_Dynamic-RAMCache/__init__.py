from .nodes import DynamicRAMCacheControl, RAMCacheExtremeCleanup

NODE_CLASS_MAPPINGS = {
    "DynamicRAMCacheControl": DynamicRAMCacheControl,
    "RAMCacheExtremeCleanup": RAMCacheExtremeCleanup
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamicRAMCacheControl": "🔥 Dynamic RAM Cache Control",
    "RAMCacheExtremeCleanup": "🧹 RAM Cache Extreme Cleanup"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
