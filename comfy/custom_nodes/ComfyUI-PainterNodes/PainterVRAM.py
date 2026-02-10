from typing import Any as any_type
from comfy import model_management
import gc
import time
import random

# Attempt to import pynvml; if missing, disable auto mode
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("[PainterVRAM] Warning: pynvml not installed. Auto mode will be disabled.")

# Save / restore random state for internal use
_INITIAL_RANDOM_STATE = random.getstate()
random.seed(time.time())
_RESERVED_RANDOM_STATE = random.getstate()
random.setstate(_INITIAL_RANDOM_STATE)

def gpu_memory_info():
    """Return (total_GB, used_GB) for GPU-0.  None if unavailable."""
    if not PYNVML_AVAILABLE:
        return None, None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = info.total / (1024 ** 3)
        used  = info.used  / (1024 ** 3)
        return total, used
    except Exception as e:
        print(f"[PainterVRAM] Failed to query GPU memory: {e}")
        return None, None

class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True
    def __ne__(self, _):
        return False

any_type = AlwaysEqualProxy("*")

class PainterVRAM:
    """Manage ComfyUI EXTRA_RESERVED_VRAM with manual or auto mode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reserved": ("FLOAT", {
                    "default": 0.6,
                    "min": -2.0,
                    "step": 0.1,
                    "display": "reserved (GB)"
                }),
                "mode": (["manual", "auto"], {
                    "default": "auto",
                    "display": "Mode"
                }),
                "clean_gpu_before": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "anything": (any_type, {})
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "apply"
    CATEGORY = "VRAM"

    @staticmethod
    def force_cleanup():
        """Aggressively free GPU memory."""
        gc.collect()
        model_management.unload_all_models()
        model_management.soft_empty_cache()

    def apply(self, reserved, mode="auto", clean_gpu_before=True,
              anything=None, unique_id=None, extra_pnginfo=None):
        if clean_gpu_before:
            print("[PainterVRAM] Pre-cleanup GPU memory...")
            self.force_cleanup()
            print("[PainterVRAM] GPU cleanup finished")

        final_reserved_gb = 0.0

        if mode == "auto":
            if PYNVML_AVAILABLE:
                total, used = gpu_memory_info()
                if total is not None and used is not None:
                    auto_reserved = used + reserved
                    auto_reserved = max(0.0, auto_reserved)
                    print(f"[PainterVRAM] Set EXTRA_RESERVED_VRAM={auto_reserved:.2f} GB "
                          f"(auto: total={total:.2f} GB, used={used:.2f} GB)")
                    model_management.EXTRA_RESERVED_VRAM = int(auto_reserved * 1024 ** 3)
                    final_reserved_gb = round(auto_reserved, 2)
                else:
                    print("[PainterVRAM] Auto query failed; fallback to manual value")
                    model_management.EXTRA_RESERVED_VRAM = int(max(0.0, reserved) * 1024 ** 3)
                    final_reserved_gb = round(max(0.0, reserved), 2)
            else:
                print("[PainterVRAM] pynvml unavailable; fallback to manual value")
                model_management.EXTRA_RESERVED_VRAM = int(max(0.0, reserved) * 1024 ** 3)
                final_reserved_gb = round(max(0.0, reserved), 2)
        else:
            # Manual mode
            reserved = max(0.0, reserved)
            model_management.EXTRA_RESERVED_VRAM = int(reserved * 1024 ** 3)
            print(f"[PainterVRAM] Set EXTRA_RESERVED_VRAM={reserved:.2f} GB (manual)")
            final_reserved_gb = round(reserved, 2)

        from comfy_execution.graph import ExecutionBlocker
        output_value = anything if anything is not None else ExecutionBlocker(None)
        return (output_value,)

NODE_CLASS_MAPPINGS = {
    "PainterVRAM": PainterVRAM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterVRAM": "Painter VRAM "
}
