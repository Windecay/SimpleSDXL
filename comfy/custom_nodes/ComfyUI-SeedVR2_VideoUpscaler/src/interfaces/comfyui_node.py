# ComfyUI Node Interface
# Clean interface for SeedVR2 VideoUpscaler integration with ComfyUI
# Extracted from original seedvr2.py lines 1731-1812

import os
import time
import torch
from typing import Tuple, Dict, Any

from ..utils.constants import get_base_cache_dir, get_script_directory
from ..utils.downloads import download_weight
from ..utils.model_registry import (
    get_available_models,
    DEFAULT_MODEL
)
from ..utils.debug import Debug
from ..core.model_manager import configure_runner
from ..core.generation import (
    setup_device_environment,
    prepare_generation_context, 
    prepare_runner,
    encode_all_batches, 
    upscale_all_batches, 
    decode_all_batches, 
)
from ..optimization.memory_manager import (
    release_text_embeddings,
    complete_cleanup, 
    get_device_list
)

# Import ComfyUI progress reporting
try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

script_directory = get_script_directory()

class SeedVR2:
    """
    SeedVR2 Video Upscaler ComfyUI Node
    
    High-quality video upscaling using diffusion models with support for:
    - Multiple model variants (3B/7B, FP16/FP8)
    - Adaptive VRAM management
    - Advanced dtype compatibility
    - Optimized inference pipeline
    - Real-time progress reporting
    """
    
    def __init__(self):
        """Initialize SeedVR2 node"""
        self.runner = None
        self.current_model_name = ""
        self.ctx = None
        self.debug = None


    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Define ComfyUI input parameter types and constraints
        
        Returns:
            Dictionary defining input parameters, types, and validation
        """
        return {
            "required": {
                "images": ("IMAGE", ),
                "model": (get_available_models(), {
                    "default": DEFAULT_MODEL,
                    "tooltip": "Model variants with different sizes and precisions. Models will automatically download on first use. Additional models can be added to the ComfyUI models folder."
                }),
                "seed": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 2**32 - 1, 
                    "step": 1,
                    "tooltip": "Random seed for generation. Same seed = same output."
                }),
                "new_resolution": ("INT", {
                    "default": 1072, 
                    "min": 16, 
                    "max": 4320, 
                    "step": 16,
                    "tooltip": "Target resolution for the shortest edge. Maintains aspect ratio."
                }),
                "batch_size": ("INT", {
                    "default": 5, 
                    "min": 1, 
                    "max": 2048, 
                    "step": 4,
                    "tooltip": "Frames per batch (with 4n+1 format: 1, 5, 9, 13, 17, 21...)"
                }),
                "color_correction": (["wavelet", "adain", "none"], {
                    "default": "wavelet",
                    "tooltip": "Color correction method. Wavelet: Frequency-based for natural results (recommended). AdaIN: Statistical matching for stylized effects. None: No color correction."
                }),
                "input_noise_scale": ("FLOAT", {
                    "default": 0.00,
                    "min": 0.00,
                    "max": 1.00,
                    "step": 0.001,
                    "tooltip": "Input noise to reduce artifacts at high resolutions. Adds subtle noise to images before upscaling."
                }),
                "latent_noise_scale": ("FLOAT", {
                    "default": 0.00,
                    "min": 0.00,
                    "max": 1.00,
                    "step": 0.001,
                    "tooltip": "Latent space noise during diffusion. Affects the conditioning process and can soften details. Use only if input_noise doesn't help."
                }),
            },
            "optional": {
                "block_swap_config": ("block_swap_config", {
                    "tooltip": "Optional BlockSwap configuration for additional VRAM savings"
                }),
                "extra_args": ("extra_args", {
                    "tooltip": "Configure extra args"
                }),
            }
        }
    
    # Define return types for ComfyUI
    RETURN_NAMES = ("image", )
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "SEEDVR2"

    def execute(self, images: torch.Tensor, model: str, seed: int, new_resolution: int, 
        batch_size: int, color_correction: str, input_noise_scale: float, latent_noise_scale: float, 
        block_swap_config=None, extra_args=None) -> Tuple[torch.Tensor]:
        """Execute SeedVR2 video upscaling with progress reporting"""
        
        temporal_overlap = 0 
        
        if extra_args is None:
            tiled_vae = False
            vae_tile_size = 512
            vae_tile_overlap = 64
            preserve_vram = False
            cache_model = False
            enable_debug = False
            devices = get_device_list()
            device = devices[0]
        else:
            tiled_vae = extra_args["tiled_vae"]
            vae_tile_size = extra_args["vae_tile_size"]
            vae_tile_overlap = extra_args["vae_tile_overlap"]
            preserve_vram = extra_args["preserve_vram"]
            cache_model = extra_args["cache_model"]
            enable_debug = extra_args["enable_debug"]
            device = extra_args["device"]
        
        # Validate tiling parameters
        if vae_tile_overlap >= vae_tile_size:
            raise ValueError(f"VAE tile overlap ({vae_tile_overlap}) must be less than tile size ({vae_tile_size})")

        # Initialize or reuse debug instance based on enable_debug parameter with timestamps
        if self.debug is None:
            self.debug = Debug(enabled=enable_debug, show_timestamps=enable_debug)
        else:
            self.debug.enabled = enable_debug

        self.debug.log("", category="none", force=True)
        self.debug.log(" ╔══════════════════════════════════════════════════════════╗", category="none", force=True)
        self.debug.log(" ║ ███████ ███████ ███████ ██████  ██    ██ ██████  ███████ ║", category="none", force=True)
        self.debug.log(" ║ ██      ██      ██      ██   ██ ██    ██ ██   ██      ██ ║", category="none", force=True)
        self.debug.log(" ║ ███████ █████   █████   ██   ██ ██    ██ ██████  █████   ║", category="none", force=True)
        self.debug.log(" ║      ██ ██      ██      ██   ██  ██  ██  ██   ██ ██      ║", category="none", force=True)
        self.debug.log(" ║ ███████ ███████ ███████ ██████    ████   ██   ██ ███████ ║", category="none", force=True)
        self.debug.log(" ║                         © ByteDance Seed · NumZ · AInVFX ║", category="none", force=True)
        self.debug.log(" ╚══════════════════════════════════════════════════════════╝", category="none", force=True)
        self.debug.log("", category="none", force=True)

        self.debug.start_timer("total_execution", force=True)

        self.debug.log("━━━━━━━━━ Model Preparation ━━━━━━━━━", category="none")

        # Initial memory state
        self.debug.log_memory_state("Before model preparation", show_tensors=True, detailed_tensors=False)
        self.debug.start_timer("model_preparation")

        # Check if download succeeded
        if not download_weight(model, debug=self.debug):
            raise RuntimeError(
                f"Required files for {model} are not available. "
                "Please check the console output for manual download instructions."
            )

        cfg_scale = 1.0
        try:
            return self._internal_execute(images, model, seed, new_resolution, cfg_scale, 
                                        batch_size, color_correction, input_noise_scale, latent_noise_scale, 
                                        tiled_vae, vae_tile_size, vae_tile_overlap, 
                                        preserve_vram, temporal_overlap, 
                                        cache_model, device, block_swap_config)
        except Exception as e:
            self.cleanup(cache_model=cache_model, debug=self.debug)
            raise e
        

    def cleanup(self, cache_model: bool = False, debug=None):
        """
        Cleanup runner and free memory
        """
        # Clear progress bar if it exists
        if hasattr(self, '_pbar') and self._pbar is not None:
            self._pbar.update_absolute(0, 100)
            self._pbar = None
            
        # Get debug from runner if not provided
        if debug is None and self.runner and hasattr(self.runner, 'debug'):
            debug = self.runner.debug
        
        # Consolidated cleanup function
        if self.runner:
            complete_cleanup(runner=self.runner, debug=debug, keep_models_in_ram=cache_model)
            
            if not cache_model:
                # Delete the runner completely
                del self.runner
                self.runner = None
        
        # Clean up context text embeddings if they exist
        if self.ctx and self.ctx.get('text_embeds'):
            embeddings = []
            names = []
            for key, embeds_list in self.ctx['text_embeds'].items():
                if embeds_list:
                    embeddings.extend(embeds_list)
                    names.append(key)
            if embeddings:
                release_text_embeddings(*embeddings, debug=debug, names=names)
            self.ctx['text_embeds'] = None
        self.ctx = None
        
        if not cache_model:
            self.current_model_name = ""


    def _internal_execute(self, images, model, seed, new_resolution, cfg_scale, batch_size,
                color_correction, input_noise_scale, latent_noise_scale, tiled_vae, vae_tile_size, vae_tile_overlap,
                preserve_vram, temporal_overlap, cache_model, device, block_swap_config) -> Tuple[torch.Tensor]:
        """Internal execution logic with progress tracking"""
        
        debug = self.debug
        
        # Initialize progress bar
        if ProgressBar is not None:
            self._pbar = ProgressBar(100)
        else:
            self._pbar = None

        # Setup device environment
        device = setup_device_environment(device, debug)

        # Create generation context with the configured device
        ctx = prepare_generation_context(device=device, debug=debug)
        self.ctx = ctx

        # Prepare runner with model state management
        self.runner, model_changed = prepare_runner(
            model, get_base_cache_dir(), preserve_vram, debug,
            cache_model=cache_model,
            block_swap_config=block_swap_config,
            vae_tiling_enabled=tiled_vae,
            vae_tile_size=(vae_tile_size, vae_tile_size),
            vae_tile_overlap=(vae_tile_overlap, vae_tile_overlap),
            cached_runner=self.runner if cache_model else None
        )
        self.current_model_name = model

        debug.end_timer("model_preparation", "Model preparation", force=True, show_breakdown=True)

        debug.log("", category="none", force=True)
        debug.log("Starting video upscaling generation...", category="generation", force=True)
        debug.start_timer("generation")
       
        # Track total frames for FPS calculation
        total_frames = len(images)
        debug.log(f"   Total frames: {total_frames}, New resolution: {new_resolution}, Batch size: {batch_size}", category="generation", force=True)
        
        # Phase 1: Encode all batches
        ctx = encode_all_batches(
            self.runner, 
            ctx=ctx, 
            images=images, 
            batch_size=batch_size, 
            preserve_vram=preserve_vram,
            debug=debug, 
            progress_callback=self._progress_callback, 
            temporal_overlap=temporal_overlap, 
            res_w=new_resolution,
            input_noise_scale=input_noise_scale,
            color_correction=color_correction
        )

        # Phase 2: Upscale all batches
        ctx = upscale_all_batches(
            self.runner, 
            ctx=ctx, 
            preserve_vram=preserve_vram,
            debug=debug, 
            progress_callback=self._progress_callback, 
            cfg_scale=cfg_scale, 
            seed=seed,
            latent_noise_scale=latent_noise_scale
        )

        # Phase 3: Decode all batches
        ctx = decode_all_batches(
            self.runner, 
            ctx=ctx, 
            preserve_vram=preserve_vram,
            debug=debug, 
            progress_callback=self._progress_callback,
            color_correction=color_correction
        )

        # Get final result
        sample = ctx['final_video']
        
        debug.log("", category="none", force=True)
        debug.log("Video upscaling completed successfully!", category="generation", force=True)

        # Log performance summary before clearing
        if debug.enabled:            
            # Log BlockSwap summary if it was used
            if hasattr(self.runner, '_blockswap_active') and self.runner._blockswap_active:
                swap_summary = debug.get_swap_summary()
                if swap_summary and swap_summary.get('total_swaps', 0) > 0:
                    total_time = swap_summary.get('block_total_ms', 0) + swap_summary.get('io_total_ms', 0)
                    debug.log("", category="none")
                    debug.log(f"BlockSwap overhead: {total_time:.2f}ms", category="blockswap")
                    debug.log(f"  Total swaps: {swap_summary['total_swaps']}", category="blockswap")
                    
                    # Show block swap details
                    if 'block_swaps' in swap_summary and swap_summary['block_swaps'] > 0:
                        avg_ms = swap_summary.get('block_avg_ms', 0)
                        total_ms = swap_summary.get('block_total_ms', 0)
                        min_ms = swap_summary.get('block_min_ms', 0)
                        max_ms = swap_summary.get('block_max_ms', 0)
                        
                        debug.log(f"  Block swaps: {swap_summary['block_swaps']} "
                                f"(avg: {avg_ms:.2f}ms, min: {min_ms:.2f}ms, max: {max_ms:.2f}ms, total: {total_ms:.2f}ms)", 
                                category="blockswap")
                        
                        # Show most frequently swapped block
                        if 'most_swapped_block' in swap_summary:
                            debug.log(f"  Most swapped: Block {swap_summary['most_swapped_block']} "
                                    f"({swap_summary['most_swapped_count']} times)", category="blockswap")

        debug.end_timer("generation", "Video generation")
       
        debug.log("", category="none")
        debug.log("━━━━━━━━━ Final Cleanup ━━━━━━━━━", category="none")
        debug.start_timer("final_cleanup")
        
        # Perform cleanup (this already calls clear_memory internally)
        self.cleanup(cache_model=cache_model, debug=debug)
        
        # Ensure sample is on CPU (ComfyUI expects CPU tensors)
        if torch.is_tensor(sample) and sample.is_cuda:
            sample = sample.cpu()
        
        # Log final memory state after ALL cleanup is done
        debug.end_timer("final_cleanup", "Final cleanup", show_breakdown=True)
        debug.log_memory_state("After final cleanup", show_tensors=True, detailed_tensors=False)
        
        # Final timing summary
        debug.log("", category="none")
        debug.log("━━━━━━━━━━━━━━━━━━", category="none")
        child_times = {
            "Model preparation": debug.timer_durations.get("model_preparation", 0),
            "Video generation": debug.timer_durations.get("generation", 0),
            "Final cleanup": debug.timer_durations.get("final_cleanup", 0)
        }
        # Add phase breakdowns as separate items (they'll appear at the same level)
        if "phase1_encoding" in debug.timer_durations:
            child_times["  Phase 1: VAE encoding"] = debug.timer_durations.get("phase1_encoding", 0)
        if "phase2_upscaling" in debug.timer_durations:
            child_times["  Phase 2: DiT upscaling"] = debug.timer_durations.get("phase2_upscaling", 0)
        if "phase3_decoding" in debug.timer_durations:
            child_times["  Phase 3: VAE decoding"] = debug.timer_durations.get("phase3_decoding", 0)

        total_execution_time = debug.end_timer("total_execution", "Total execution", show_breakdown=True, custom_children=child_times)
        
        # Calculate and log FPS
        if total_execution_time > 0:
            fps = total_frames / total_execution_time
            debug.log(f"Average FPS: {fps:.2f} frames/sec", category="timing", force=True)
            
        debug.log("━━━━━━━━━━━━━━━━━━", category="none")
        
        # Clear history for next run (do this last, after all logging)
        debug.clear_history()

        # Clear progress bar and context reference
        self._pbar = None
        self.ctx = None

        return (sample,)

    def _progress_callback(self, current_step, total_steps, current_batch_frames, phase_name=""):
        """Progress callback for generation phases"""
        
        if not hasattr(self, '_pbar') or self._pbar is None:
            return
        
        # Calculate overall progress across all phases
        # Phase weights: Encode=20%, Upscale=60%, Decode=20%
        phase_weights = {"Encoding": 0.2, "Upscaling": 0.6, "Decoding": 0.2}
        phase_offset = {"Encoding": 0.0, "Upscaling": 0.2, "Decoding": 0.8}
        
        # Extract the phase type from "Phase X: Type" format
        if ":" in phase_name:
            phase_key = phase_name.split(":")[1].strip()
        else:
            phase_key = "Upscaling"
        
        weight = phase_weights.get(phase_key, 1.0)
        offset = phase_offset.get(phase_key, 0.0)
        
        # Calculate weighted progress
        phase_progress = (current_step / total_steps) if total_steps > 0 else 0
        overall_progress = offset + (phase_progress * weight)
        
        # Update the progress bar with the overall progress
        progress_value = int(overall_progress * 100)
        self._pbar.update_absolute(progress_value, 100)

    def __del__(self):
        """Destructor"""
        try:
            # Store debug reference
            debug = self.debug if hasattr(self, 'debug') else None
            
            # Full cleanup
            if hasattr(self, 'cleanup'):
                self.cleanup(cache_model=False, debug=debug)
            
            # Clear all remaining references
            for attr in ['runner', 'current_model_name', 'debug', 'ctx']:
                if hasattr(self, attr):
                    delattr(self, attr)
        except:
            pass

class SeedVR2BlockSwap:
    """Configure block swapping to reduce VRAM usage"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blocks_to_swap": (
                    "INT",
                    {
                        "default": 16,
                        "min": 0,
                        "max": 36,
                        "step": 1,
                        "tooltip": "Number of transformer blocks to swap to CPU. Start with 16 and increase until OOM errors stop. 0=disabled",
                    },
                ),
                "offload_io_components": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Offload embeddings and I/O layers to CPU. Enable if you need additional VRAM savings beyond block swapping",
                    },
                ),
            }
        }

    RETURN_TYPES = ("block_swap_config",)
    FUNCTION = "create_config"
    CATEGORY = "SEEDVR2"
    DESCRIPTION = """Configure block swapping to reduce VRAM usage during video upscaling.

BlockSwap dynamically moves transformer blocks between GPU and CPU/RAM during inference, enabling large models to run on limited VRAM systems with minimal performance impact.

Configuration Guidelines:
    - blocks_to_swap=0: Disabled (fastest, highest VRAM usage)
    - blocks_to_swap=16: Balanced mode (moderate speed/VRAM trade-off)
    - blocks_to_swap=32-36: Maximum savings (slowest, lowest VRAM)

Advanced Options:
    - offload_io_components: Moves embeddings and I/O layers to CPU for additional VRAM savings (slower)

Performance Tips:
    - Start with blocks_to_swap=16 and increase until you no longer get OOM errors or decrease if you have spare VRAM
    - Enable offload_io_components if you still need additional VRAM savings
    - Note: Even if inference succeeds, you may still OOM during VAE decoding - combine BlockSwap with VAE tiling if needed (feature in development)
    - Combine with smaller batch_size for maximum VRAM savings

The actual memory savings depend on your specific model architecture and will be shown in the debug output when enabled.
    """

    def create_config(self, blocks_to_swap, offload_io_components):
        """Create BlockSwap configuration"""
        if blocks_to_swap == 0:
            return (None,)
        config = {
            "blocks_to_swap": blocks_to_swap,
            "offload_io_components": offload_io_components,
        }
        
        return (config,)
    
class SeedVR2ExtraArgs:
    """Configure extra args"""
    
    @classmethod
    def INPUT_TYPES(cls):
        devices = get_device_list()
        return {
            "required": {
                "tiled_vae": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Process VAE in tiles to reduce VRAM usage but slower with potential artifacts. Only enable if running out of memory."
                }),
                "vae_tile_size": ("INT", {
                    "default": 512,
                    "min": 64,
                    "step": 32,
                    "tooltip": "VAE tile size in pixels. Smaller = less VRAM but more seams/artifacts and slower. Larger = more VRAM but better quality and faster."
                }),
                "vae_tile_overlap": ("INT", {
                    "default": 64,
                    "min": 0,
                    "step": 32,
                    "tooltip": "Pixel overlap between tiles to reduce visible seams. Higher = better blending but slower processing."
                }),
                "preserve_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload models between steps to save VRAM. Slower but uses less memory."
                }),
                "cache_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep model and VAE in RAM between runs. Speeds up batch processing."
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show detailed memory usage and timing information during generation."
                }),
                "device": (devices, {
                    "default": devices[0]
                }),
            }
        }
    
    RETURN_TYPES = ("extra_args",)
    FUNCTION = "create_config"
    CATEGORY = "SEEDVR2"
    DESCRIPTION = "Configure extra args."
    
    def create_config(self, tiled_vae, vae_tile_size, vae_tile_overlap, preserve_vram, cache_model, enable_debug, device):
        config = {
            "tiled_vae": tiled_vae,
            "vae_tile_size": vae_tile_size,
            "vae_tile_overlap": vae_tile_overlap,
            "preserve_vram": preserve_vram,
            "cache_model": cache_model,
            "enable_debug": enable_debug,
            "device": device,
        }
        
        return (config,)


# ComfyUI Node Mappings
NODE_CLASS_MAPPINGS = {
    "SeedVR2": SeedVR2,
    "SeedVR2BlockSwap": SeedVR2BlockSwap,
    "SeedVR2ExtraArgs": SeedVR2ExtraArgs,
}

# Human-readable node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2": "SeedVR2 Video Upscaler",
    "SeedVR2BlockSwap": "SeedVR2 BlockSwap Config",
    "SeedVR2ExtraArgs": "SeedVR2 Extra Args",
}

# Export version and metadata
__version__ = "2.0.0-modular"
__author__ = "SeedVR2 Team"
__description__ = "High-quality video upscaling using advanced diffusion models"

# Additional exports for introspection
__all__ = [
    'SeedVR2',
    'NODE_CLASS_MAPPINGS', 
    'NODE_DISPLAY_NAME_MAPPINGS'
] 
