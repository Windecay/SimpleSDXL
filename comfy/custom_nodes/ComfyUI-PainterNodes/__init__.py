import traceback
import importlib

# Define list of node modules to import
NODE_MODULES = [
    ("PainterPrompt", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterI2V", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterI2VAdvanced", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterAI2V", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterAV2V", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterSampler", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterSamplerLTXV", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterLTX2V", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterLTX2VPlus", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterFLF2V", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterMultiF2V", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterLongVideo", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterSequentialF2V", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterFluxImageEdit", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterQwenImageEditPlus", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterVRAM", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterVideoCombine", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterVideoUpscale", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterVideoInfo", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterFrameCount", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterImageLoad", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterImageFromBatch", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterCombineFromBatch", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterAudioLength", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterAudioCut", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterS2Vplus", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterHumoAV2V", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
    ("PainterHumoAI2V", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"),
]

# Initialize global mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
# Track failed modules
failed_modules = []

# Import each node module individually with error handling
for module_name, class_map_name, display_name_map_name in NODE_MODULES:
    try:
        # Use importlib for safe relative import
        full_module_name = f"{__name__}.{module_name}"
        module = importlib.import_module(full_module_name)
        
        # Get mapping dictionaries from the module
        class_mappings = getattr(module, class_map_name, {})
        display_name_mappings = getattr(module, display_name_map_name, {})
        
        # Merge to global mappings
        NODE_CLASS_MAPPINGS.update(class_mappings)
        NODE_DISPLAY_NAME_MAPPINGS.update(display_name_mappings)
        
    except Exception as e:
        # Record failed module and its error info
        failed_modules.append({
            "name": module_name,
            "error": str(e),
            "traceback": traceback.format_exc()
        })

# Print failed modules info if any
if failed_modules:
    print(f"[PainterNodes] Found {len(failed_modules)} failed modules:")
    for fm in failed_modules:
        print(f"\n[PainterNodes] Failed to import module {fm['name']}: {fm['error']}")
        print(f"Detailed error information:\n{fm['traceback']}")

# Print success message with green color (ANSI escape code)
# Green color: \033[92m, Reset: \033[0m
success_msg = f"\033[92m[PainterNodes] Loaded {len(NODE_CLASS_MAPPINGS)} nodes successfully!🎉\033[0m"
print(success_msg)

__version__ = "1.0.0"
WEB_DIRECTORY = "./web/js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "__version__",
]
