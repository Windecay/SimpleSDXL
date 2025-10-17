"""
Model Management Module for SeedVR2

This module handles all model-related operations including:
- Model configuration and path resolution
- Model loading with format detection (SafeTensors, PyTorch)
- DiT and VAE model setup and inference configuration
- State dict management with native FP8 support
- Universal compatibility wrappers

Key Features:
- Dynamic import path resolution for different ComfyUI environments
- Native FP8 model support with optimal performance
- Automatic compatibility mode for model architectures
- Memory-efficient model loading and configuration
"""

import os
import torch
import numpy as np
from omegaconf import OmegaConf
from typing import Dict, Optional, Tuple, Any

# Import SafeTensors with fallback
try:
    from safetensors.torch import load_file as load_safetensors_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# Import Debug for type annotations
from ..utils.debug import Debug

# Import GGUF with fallback  
try:
    import gguf
    from gguf import GGMLQuantizationType
    import traceback
    from ..optimization.gguf_dequant import dequantize_tensor
    from ..optimization.gguf_ops import replace_linear_with_quantized
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    GGMLQuantizationType = None

from ..utils.constants import get_script_directory, find_model_file, suppress_tensor_warnings
from ..optimization.memory_manager import clear_memory
from ..optimization.compatibility import FP8CompatibleDiT
from ..common.config import load_config, create_object
from .infer import VideoDiffusionInfer
from ..optimization.blockswap import apply_block_swap_to_dit, cleanup_blockswap
from ..common.distributed import get_device

# Get script directory for config paths
script_directory = get_script_directory()


def configure_runner(model: str, base_cache_dir: str, preserve_vram: bool = False, debug: Optional[Debug] = None, 
                    cache_model: bool = False, block_swap_config: Optional[Dict[str, Any]] = None, 
                    cached_runner: Optional[VideoDiffusionInfer] = None, vae_tiling_enabled: bool = False,
                    vae_tile_size: Optional[Tuple[int, int]] = None, 
                    vae_tile_overlap: Optional[Tuple[int, int]] = None) -> VideoDiffusionInfer:
    """
    Configure and create a VideoDiffusionInfer runner for the specified model
    
    Args:
        model (str): Model filename (e.g., "seedvr2_ema_3b_fp16.safetensors")
        base_cache_dir (str): Base directory containing model files
        preserve_vram (bool): Whether to preserve VRAM
        debug: Debug instance for logging
        cache_model (bool): Enable model caching
        block_swap_config (dict): Optional BlockSwap configuration
        cached_runner: Optional cached runner to reuse entirely (not just DiT)
        
    Returns:
        VideoDiffusionInfer: Configured runner instance ready for inference
        
    Features:
        - Dynamic config loading based on model type (3B vs 7B)
        - Automatic import path resolution for different environments
        - VAE configuration with proper parameter handling
        - Memory optimization and RoPE cache pre-initialization
    """
    # Check if debug instance is available
    if debug is None:
        raise ValueError("Debug instance must be provided to configure_runner")
    
    # Check if we can reuse the cached runner
    if cached_runner and cache_model:

        debug.start_timer("cache_reuse")
        # Update all runtime parameters dynamically
        runtime_params = {
            'vae_tiling_enabled': vae_tiling_enabled,
            'vae_tile_size': vae_tile_size,
            'vae_tile_overlap': vae_tile_overlap
        }
        for key, value in runtime_params.items():
            setattr(cached_runner, key, value)
        
        debug.log(f"Cache hit: Reusing runner for model {model}", category="reuse", force=True)
        
        # Normalize configurations for comparison
        # None represents "BlockSwap disabled"
        desired_config = None
        if block_swap_config and block_swap_config.get("blocks_to_swap", 0) > 0:
            desired_config = (
                block_swap_config.get("blocks_to_swap"),
                block_swap_config.get("offload_io_components", False)
            )
        
        current_config = None
        if hasattr(cached_runner, "_block_swap_config"):
            current_config = (
                cached_runner._block_swap_config.get("blocks_swapped"),
                cached_runner._block_swap_config.get("offload_io_components", False)
            )
        
        # Format configs for logging
        def format_config(cfg):
            if cfg is None:
                return "disabled"
            return f"blocks={cfg[0]}, offload_io={cfg[1]}"
        
        # Apply changes only if configurations differ
        if desired_config != current_config:
            # Log the change
            debug.log(f"BlockSwap config changed: {format_config(current_config)} → {format_config(desired_config)}", 
                     category="blockswap", force=True)
            
            # Always cleanup existing BlockSwap state when config changes
            cleanup_blockswap(cached_runner, keep_state_for_cache=False)
            
            # Apply new configuration only if not disabled
            if desired_config:
                apply_block_swap_to_dit(cached_runner, block_swap_config, debug)
                
        elif desired_config and hasattr(cached_runner, "_blockswap_active") and not cached_runner._blockswap_active:
            # Config matches but was deactivated - just reactivate
            cached_runner._blockswap_active = True
            debug.log(f"BlockSwap reactivated: {format_config(desired_config)}", category="reuse", force=True)
        
        # Store debug instance on runner
        cached_runner.debug = debug

        debug.end_timer("cache_reuse", "Reuse cached runner")

        return cached_runner
        
    else:
        debug.log(f"Cache miss: Creating new runner for model {model}", category="cache", force=True)
    
    # If we reach here, create a new runner
    debug.start_timer("config_load")

    # Select config based on model type   
    if "7b" in model:
        config_path = os.path.join(script_directory, './configs_7b', 'main.yaml')
    else:
        config_path = os.path.join(script_directory, './configs_3b', 'main.yaml')
    
    config = load_config(config_path)
    debug.end_timer("config_load", "Config loading")

    # Load and configure VAE with additional parameters
    vae_config_path = os.path.join(script_directory, 'src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml')
    debug.start_timer("vae_config_load")
    vae_config = OmegaConf.load(vae_config_path)
    debug.end_timer("vae_config_load", "VAE configuration YAML parsed from disk")
    
    debug.start_timer("vae_config_set")
    # Configure VAE parameters
    spatial_downsample_factor = vae_config.get('spatial_downsample_factor', 8)
    temporal_downsample_factor = vae_config.get('temporal_downsample_factor', 4)
    vae_config.spatial_downsample_factor = spatial_downsample_factor
    vae_config.temporal_downsample_factor = temporal_downsample_factor
    debug.end_timer("vae_config_set", f"VAE downsample factors configuration")
    
    # Merge additional VAE config with main config (preserving __object__ from main config)
    debug.start_timer("vae_config_merge")
    config.vae.model = OmegaConf.merge(config.vae.model, vae_config)
    debug.end_timer("vae_config_merge", "VAE config merged with main pipeline config")
    
    debug.start_timer("runner_video_infer")
    # Create runner
    runner = VideoDiffusionInfer(config, debug, vae_tiling_enabled=vae_tiling_enabled, vae_tile_size=vae_tile_size, vae_tile_overlap=vae_tile_overlap)
    OmegaConf.set_readonly(runner.config, False)
    # Store model name for cache validation
    runner._model_name = model
    debug.end_timer("runner_video_infer", "Video diffusion inference runner initialization")
    
    # Set device
    device = str(get_device())
    #if torch.mps.is_available():
    #    device = "mps"
    
    # Configure models
    dit_checkpoint_path = find_model_file(model, base_cache_dir)
    debug.start_timer("dit_model_infer")
    runner = configure_model_inference(runner, "dit", device, dit_checkpoint_path, config,
                                   preserve_vram, debug, block_swap_config)

    debug.end_timer("dit_model_infer", "DiT model configuration")
    debug.log_memory_state("After DiT model configuration", detailed_tensors=False)

    # Set VAE dtype for MPS compatibility
    if torch.mps.is_available():
        original_vae_dtype = config.vae.dtype
        config.vae.dtype = "bfloat16"
        debug.log(f"MPS detected: Setting VAE dtype from {original_vae_dtype} to {config.vae.dtype} for compatibility", 
                    category="precision", force=True)
    
    debug.start_timer("vae_model_infer")
    vae_checkpoint_path = find_model_file(config.vae.checkpoint, base_cache_dir)
    vae_override_dtype = getattr(torch, config.vae.dtype) if torch.mps.is_available() else None
    runner = configure_model_inference(runner, "vae", device, vae_checkpoint_path, config,
                                   preserve_vram, debug=debug, override_dtype=vae_override_dtype)
    debug.log(f"VAE downsample factors configured (spatial: {spatial_downsample_factor}x, temporal: {temporal_downsample_factor}x)", category="vae")
    debug.end_timer("vae_model_infer", "VAE model configuration")
    debug.log_memory_state("After VAE model configuration", detailed_tensors=False)
    
    debug.start_timer("vae_memory_limit")
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    debug.end_timer("vae_memory_limit", "VAE memory limit set")
    
    # Check if BlockSwap is active
    blockswap_active = (
        block_swap_config and block_swap_config.get("blocks_to_swap", 0) > 0
    )
    
    # Apply BlockSwap if configured
    if blockswap_active:
        apply_block_swap_to_dit(runner, block_swap_config, debug)
    
    # Store debug instance on runner for consistent access
    runner.debug = debug
    
    return runner


def load_quantized_state_dict(checkpoint_path: str, device: str = "cpu", debug: Optional[Debug] = None) -> Dict[str, torch.Tensor]:
    """
    Load model state dictionary with optimal memory management
    
    Args:
        checkpoint_path (str): Path to checkpoint file (.safetensors or .pth)
        device (str/torch.device): Target device for tensor placement
        debug: Optional Debug instance for logging
        
    Returns:
        dict: State dictionary loaded with appropriate format handler
        
    Notes:
        - SafeTensors files use optimized loading with direct device placement
        - PyTorch files use memory-mapped loading to reduce RAM usage
    """
    if checkpoint_path.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            error_msg = (
                f"Cannot load {os.path.basename(checkpoint_path)}\n"
                f"SafeTensors library is required but not installed.\n"
                f"Please install it with: pip install safetensors"
            )
            if debug:
                debug.log(error_msg, level="ERROR", category="dit", force=True)
                debug.log("This is a one-time installation that will enable loading of .safetensors files", 
                         level="INFO", category="info", force=True)
            raise ImportError(error_msg)
        state = load_safetensors_file(checkpoint_path, device=device)
    elif checkpoint_path.endswith('.gguf'):
        if not GGUF_AVAILABLE:
            error_msg = (
                f"Cannot load {os.path.basename(checkpoint_path)}\n"
                f"GGUF library is required but not installed.\n"
                f"Please install it with: pip install gguf"
            )
            if debug:
                debug.log(error_msg, level="ERROR", category="dit", force=True)
                debug.log("This is a one-time installation that will enable loading of quantized GGUF models", 
                         level="INFO", category="info", force=True)
            raise ImportError(error_msg)
        state = _load_gguf_state(
                    checkpoint_path=checkpoint_path, 
                    device=device, 
                    debug=debug, 
                    handle_prefix="model.diffusion_model."
                )
    elif checkpoint_path.endswith('.pth'):
        state = torch.load(checkpoint_path, map_location=device, mmap=True)
    else:
        raise ValueError(f"Unsupported checkpoint format. Expected .safetensors or .pth, got: {checkpoint_path}")
    
    return state


def _load_gguf_state(checkpoint_path: str, device: str, debug: Optional[Debug],
                    handle_prefix: str = "model.diffusion_model.") -> Dict[str, torch.Tensor]:
    """
    Load GGUF state dict
    """
    reader = gguf.GGUFReader(checkpoint_path)

    # Filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)
        
    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    state_dict = {}
    total_tensors = len(reader.tensors)
    
    debug.log(f"Loading {total_tensors} tensors to {device}...", category="dit")
    
    # Suppress expected warnings: GGUF tensors are read-only numpy arrays that trigger warnings when converted
    suppress_tensor_warnings()
    
    for i, (sd_key, tensor) in enumerate(tensors):
        tensor_name = tensor.name
        
        # Create tensor directly on target device to avoid CPU->GPU copy overhead
        # For meta-initialized models, this directly materializes to the target device
        torch_tensor = torch.from_numpy(tensor.data).to(device, non_blocking=False)
            
        # Get original shape from metadata or infer from tensor shape
        shape = _get_tensor_logical_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            
        # Handle tensors based on quantization type
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # For unquantized tensors, just reshape
            torch_tensor = torch_tensor.view(*shape)
        else:
            # For quantized tensors, keep them quantized but track original shape
            torch_tensor = GGUFTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape, debug=debug)
            
        state_dict[sd_key] = torch_tensor
        
        # Progress reporting
        if (i + 1) % 100 == 0:
            debug.log(f"    Loaded {i+1}/{total_tensors} tensors...", category="dit")

    debug.log(f"Successfully loaded {len(state_dict)} tensors to {device}", category="success")

    return state_dict


def _get_tensor_logical_shape(reader: 'gguf.GGUFReader', tensor_name: str) -> Optional[torch.Size]:
    """
    Extract the logical (unquantized) shape from GGUF metadata
    """
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


class GGUFTensor(torch.Tensor):
    """
    Tensor wrapper for GGUF quantized tensors that preserves quantization info
    """
    def __init__(self, *args, tensor_type, tensor_shape, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        
    def __new__(cls, *args, tensor_type, tensor_shape, debug, **kwargs):
        # Create tensor with requires_grad=False to avoid gradient issues
        tensor = super().__new__(cls, *args, **kwargs)
        tensor.requires_grad_(False)
        tensor.tensor_type = tensor_type
        tensor.tensor_shape = tensor_shape
        tensor.debug = debug
        return tensor
    
    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", self.tensor_shape if hasattr(self, "tensor_shape") else new.shape)
        new.debug = getattr(self, "debug", None)
        new.requires_grad_(False)  # Ensure no gradients
        return new
    
    @property
    def shape(self):
        # Always return the logical tensor shape, not the quantized data shape
        if hasattr(self, "tensor_shape"):
            return self.tensor_shape
        else:
            # Fallback to actual data shape if tensor_shape is not available
            return self.size()
        
    def size(self, *args):
        # Override size() to also return logical shape
        if hasattr(self, "tensor_shape") and len(args) == 0:
            return self.tensor_shape
        elif hasattr(self, "tensor_shape") and len(args) == 1:
            return self.tensor_shape[args[0]]
        else:
            return super().size(*args)
        
    def dequantize(self, device=None, dtype=torch.float16, dequant_dtype=None):
        """Dequantize this tensor to its original shape"""
        if device is None:
            device = self.device
            
        # Suppress expected warning when converting from GGUFTensor subclass to regular tensor
        suppress_tensor_warnings()

        # Check if already unquantized
        if self.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # Return regular tensor, not GGUFTensor
            result = self.to(device, dtype)
            if isinstance(result, GGUFTensor):
                # Convert to regular tensor to avoid __torch_function__ calls
                result = torch.tensor(result, dtype=dtype, device=device, requires_grad=False)
            return result
        
        # Try fast dequantization with crash protection
        try:
            result = dequantize_tensor(self, dtype, dequant_dtype)
            final_result = result.to(device)
            
            # Ensure we return a regular tensor, not GGUFTensor
            if isinstance(final_result, GGUFTensor):
                final_result = torch.tensor(final_result.data, dtype=dtype, device=device, requires_grad=False)
                
            return final_result
        except Exception as e:
            self.debug.log(f"Fast dequantization failed: {e}", level="WARNING", category="dit", force=True)
            self.debug.log(f"Falling back to numpy dequantization", level="WARNING", category="dit", force=True)
            
        # Fallback to numpy (slower but reliable)
        try:
            numpy_data = self.cpu().numpy()
            dequantized = gguf.quants.dequantize(numpy_data, self.tensor_type)
            result = torch.from_numpy(dequantized).to(device, dtype)
            result.requires_grad_(False)
            final_result = result.reshape(self.tensor_shape)
            # from_numpy already returns a regular tensor, no conversion needed
            return final_result
        except Exception as e:
            self.debug.log(f"Numpy fallback also failed: {e}", level="WARNING", category="dit", force=True)
            self.debug.log(f"   Tensor type: {self.tensor_type}", level="WARNING", category="dit", force=True)
            self.debug.log(f"   Shape: {self.shape}", level="WARNING", category="dit", force=True)
            self.debug.log(f"   Target shape: {self.tensor_shape}", level="WARNING", category="dit", force=True)
            traceback.print_exc()
            
            # Return regular tensor as last resort
            result = self.to(device, dtype)
            if isinstance(result, GGUFTensor):
                result = torch.tensor(result.data, dtype=dtype, device=device, requires_grad=False)
            return result
        
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Override torch function calls to automatically dequantize"""
        if kwargs is None:
            kwargs = {}
        
        # Find the GGUFTensor instance(s) in args
        gguf_tensors = [arg for arg in args if isinstance(arg, cls)]
        if not gguf_tensors:
            return super().__torch_function__(func, types, args, kwargs)
        
        # Use the first GGUFTensor instance for attribute access
        self = gguf_tensors[0]
        
        # Check if the tensor is fully constructed and still quantized
        tensor_type = getattr(self, 'tensor_type', None)
        if tensor_type is None:
            # Tensor is either being constructed or already dequantized
            return super().__torch_function__(func, types, args, kwargs)
        
        # Check if tensor is already unquantized (F32/F16)
        if tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            return super().__torch_function__(func, types, args, kwargs)
        
        # Check if debug exists before using it
        debug = getattr(self, 'debug', None)
        
        # Handle linear operations specially
        if func == torch.nn.functional.linear:
            if len(args) >= 2 and isinstance(args[1], cls):  # weight is the second argument
                try:
                    weight_tensor = args[1]
                    dequantized_weight = weight_tensor.dequantize(device=args[0].device, dtype=args[0].dtype)
                    new_args = (args[0], dequantized_weight) + args[2:]
                    return func(*new_args, **kwargs)
                except Exception as e:
                    if debug:
                        debug.log(f"Error in linear dequantization: {e}", level="WARNING", category="dit", force=True)
                        debug.log(f"  Function: {func}", level="WARNING", category="dit", force=True)
                        debug.log(f"  Args: {[arg.shape if hasattr(arg, 'shape') else type(arg) for arg in args]}", level="WARNING", category="dit", force=True)
                    raise
        
        # Handle matrix multiplication operations that need dequantization
        if func in {torch.matmul, torch.mm, torch.bmm, torch.addmm, torch.addmv,
                    torch.addr, torch.baddbmm, torch.chain_matmul}:
            try:
                new_args = []
                for arg in args:
                    if isinstance(arg, cls):
                        new_args.append(arg.dequantize())
                    else:
                        new_args.append(arg)
                return func(*tuple(new_args), **kwargs)
            except Exception as e:
                if debug:
                    debug.log(f"Error in {func.__name__} dequantization: {e}", level="WARNING", category="dit", force=True)
                raise
        
        # For ALL other operations, delegate to parent WITHOUT dequantization
        # This includes .cpu(), .to(), .device, .dtype, .shape, etc.
        return super().__torch_function__(func, types, args, kwargs)


def configure_model_inference(runner: VideoDiffusionInfer, model_type: str, device: str, 
                             checkpoint_path: str, config: OmegaConf, 
                             preserve_vram: bool = False, debug: Optional[Debug] = None, 
                             block_swap_config: Optional[Dict[str, Any]] = None,
                             override_dtype: Optional[torch.dtype] = None) -> VideoDiffusionInfer:
    """
    Configure DiT or VAE model for inference with optimized memory management.
    
    Uses meta device initialization for CPU models to avoid unnecessary memory allocation
    during model creation, reducing initialization time by ~90% for large models.
    
    Args:
        runner: VideoDiffusionInfer instance to configure
        model_type: "dit" or "vae" - determines model configuration
        device (str): Target device for inference (cuda:0, cpu, etc.)
        checkpoint_path (str): Path to model checkpoint (.safetensors or .pth)
        config: Model configuration object with dit/vae sub-configs
        preserve_vram (bool): Keep model on CPU to preserve VRAM
        debug: Debug instance for logging and profiling
        block_swap_config (dict): BlockSwap configuration (DiT only)
        override_dtype (torch.dtype, optional): Override model weights dtype during loading
            (e.g., torch.bfloat16 for MPS compatibility)
        
    Returns:
        runner: Updated runner with configured model
        
    Raises:
        ValueError: If debug instance is not provided
    """
    if debug is None:
        raise ValueError(f"Debug instance must be provided to configure_{model_type}_model_inference")
    
    # Model type configuration
    is_dit = (model_type == "dit")
    model_type_upper = "DiT" if is_dit else "VAE"
    model_config = config.dit.model if is_dit else config.vae.model
    
    # Determine target device and reason for CPU usage
    blockswap_active = is_dit and block_swap_config and block_swap_config.get("blocks_to_swap", 0) > 0
    use_cpu = preserve_vram or blockswap_active
    target_device = "cpu" if use_cpu else device
    
    # Always use meta init for memory efficiency
    use_meta_init = True

    # Create descriptive reason string for logging
    cpu_reason = ""
    if target_device == "cpu":
        reasons = []
        if blockswap_active:
            reasons.append("BlockSwap")
        if preserve_vram:
            reasons.append("preserve_vram")
        cpu_reason = f" ({', '.join(reasons)})" if reasons else ""
    
    # Create and load model
    model = _create_model(model_config, target_device, use_meta_init, model_type_upper, debug)
    model = _load_model_weights(model, checkpoint_path, target_device, use_meta_init, 
                                model_type_upper, cpu_reason, debug, override_dtype)
    
    # Apply model-specific configurations
    model = _apply_model_specific_config(model, runner, config, is_dit, debug)
    
    return runner


def _create_model(model_config: OmegaConf, target_device: str, use_meta_init: bool, 
                 model_type: str, debug: Debug):
    """
    Create model with optimized initialization strategy.
    
    Uses meta device for CPU models to avoid unnecessary memory allocation,
    otherwise creates directly on target device.
    
    Args:
        model_config: Model configuration object
        target_device: Target device for the model
        use_meta_init: Whether to use meta device initialization
        model_type: Model type string for logging
        debug: Debug instance
        
    Returns:
        Created model instance
    """
    if use_meta_init:
        # Fast path: Create on meta device to avoid memory allocation
        debug.log(f"Creating {model_type} model structure on meta device (fast initialization)", 
                 category=model_type.lower(), force=True)
        debug.start_timer(f"{model_type.lower()}_model_create")
        with torch.device("meta"):
            model = create_object(model_config)
        debug.end_timer(f"{model_type.lower()}_model_create", 
                       f"{model_type} model structure creation")
    else:
        # Standard path: Create directly on target device
        debug.log(f"Creating {model_type} model on {target_device.upper()}", 
                 category=model_type.lower(), force=True)
        debug.start_timer(f"{model_type.lower()}_model_create")
        with torch.device(target_device):
            model = create_object(model_config)
        debug.end_timer(f"{model_type.lower()}_model_create", 
                       f"{model_type} model creation")
    
    return model


def _load_model_weights(model: torch.nn.Module, checkpoint_path: str, target_device: str, 
                        used_meta: bool, model_type: str, cpu_reason: str, 
                        debug: Debug, override_dtype: Optional[torch.dtype] = None) -> torch.nn.Module:
    """
    Load model weights from checkpoint file with optimized GGUF support.
    
    For meta-initialized models, materializes to target device.
    For standard models, loads weights and applies state dict.
    
    Args:
        model: Model instance (may be on meta device)
        checkpoint_path: Path to checkpoint file
        target_device: Target device for weights
        used_meta: Whether model was created on meta device
        model_type: Model type string for logging
        cpu_reason: Reason string if using CPU
        debug: Debug instance
        override_dtype: Optional dtype override for weights
        
    Returns:
        Model with loaded weights
    """
    model_type_lower = model_type.lower()
    
    # Log loading action
    action = "Materializing" if used_meta else "Loading"
    debug.log(f"{action} {model_type} weights to {target_device.upper()}{cpu_reason}: {checkpoint_path}", 
             category=model_type_lower, force=True)
    
    # Load state dict from file
    debug.start_timer(f"{model_type_lower}_weights_load")
    state = load_quantized_state_dict(checkpoint_path, target_device, debug)
    debug.end_timer(f"{model_type_lower}_weights_load", f"{model_type} weights loaded from file")
    
    # Apply dtype conversion if requested
    if override_dtype is not None:
        state = _convert_state_dtype(state, override_dtype, model_type, debug)
    
    # Log weight statistics
    _log_weight_stats(state, used_meta, model_type, debug)
    
    # Handle GGUF or standard loading
    if checkpoint_path.endswith('.gguf'):
        model = _load_gguf_weights(model, state, used_meta, model_type_lower, debug)
    else:
        model = _load_standard_weights(model, state, used_meta, model_type, model_type_lower, debug)
    
    # Clean up state dict
    del state
    
    # Initialize meta buffers if needed
    if used_meta:
        _initialize_meta_buffers_wrapped(model, target_device, debug)
    
    return model


def _convert_state_dtype(state: Dict[str, torch.Tensor], target_dtype: torch.dtype, 
                        model_type: str, debug: Debug) -> Dict[str, torch.Tensor]:
    """Convert floating point tensors in state dict to target dtype."""
    debug.log(f"Converting {model_type} weights to {target_dtype} during loading", category="precision")
    debug.start_timer(f"{model_type.lower()}_dtype_convert")
    
    for key in state:
        if torch.is_tensor(state[key]) and state[key].is_floating_point():
            state[key] = state[key].to(target_dtype)
    
    debug.end_timer(f"{model_type.lower()}_dtype_convert", f"{model_type} weights converted to {target_dtype}")
    return state


def _log_weight_stats(state: Dict[str, torch.Tensor], used_meta: bool, model_type: str, debug: Debug) -> None:
    """Log statistics about loaded weights."""
    num_params = len(state)
    total_size_mb = sum(p.nelement() * p.element_size() for p in state.values()) / (1024 * 1024)
    action = "Materializing" if used_meta else "Applying"
    debug.log(f"{action} {model_type}: {num_params} parameters, {total_size_mb:.2f}MB total", 
             category=model_type.lower())


def _load_standard_weights(model: torch.nn.Module, state: Dict[str, torch.Tensor], 
                          used_meta: bool, model_type: str, model_type_lower: str,
                          debug: Debug) -> torch.nn.Module:
    """Load standard (non-GGUF) weights into model."""
    debug.start_timer(f"{model_type_lower}_state_apply")
    model.load_state_dict(state, strict=False, assign=True)
    
    action = "materialized" if used_meta else "applied"
    debug.end_timer(f"{model_type_lower}_state_apply", f"{model_type} weights {action}")
    
    if used_meta:
        debug.log(f"{model_type} materialized directly from meta with loaded weights", category=model_type_lower)
    else:
        debug.log(f"{model_type} weights applied", category=model_type_lower)
    
    return model


def _load_gguf_weights(model: torch.nn.Module, state: Dict[str, torch.Tensor], 
                      used_meta: bool, model_type_lower: str, debug: Debug) -> torch.nn.Module:
    """
    Load GGUF quantized weights into model with architecture validation.
    
    Args:
        model: Target model
        state: GGUF state dict with quantized tensors
        used_meta: Whether model was initialized on meta device
        model_type_lower: Lowercase model type for logging
        debug: Debug instance
        
    Returns:
        Model with GGUF weights loaded
    """
    debug.log("Loading GGUF weights", category="dit")
    
    # Get model state dict for validation
    model_state = model.state_dict()
    
    # Validate architecture compatibility
    _validate_gguf_architecture(state, model_state, debug)
    
    # Load GGUF parameters
    stats = _apply_gguf_parameters(model, state, model_state, debug)
    
    # Log results
    debug.log(f"GGUF loading complete: {stats['loaded']} parameters loaded", category="success")
    debug.log(f"Quantized parameters: {stats['quantized']}", category="info")
    
    # Report any mismatches
    _report_parameter_mismatches(state, model_state, stats['loaded_names'], debug)
    
    # Replace Linear/Conv2d layers with quantized versions for optimal precision handling
    if stats['quantized'] > 0:
        debug.log("Replacing layers with GGUF-optimized versions for precision handling", category="dit")
        
        replacements, quant_types = replace_linear_with_quantized(model, debug=debug)
        
        if replacements > 0:
            debug.log(f"Replaced {replacements} layers with GGUF-optimized versions", category="success")
            
            # Show actual quantization types found and precision strategy
            if quant_types:
                qtypes_str = ', '.join([f"{qtype}:{count}" for qtype, count in quant_types.items()])
                debug.log(
                    f"GGUF precision path: {qtypes_str} → FP16 (preserve) → BF16/FP32 (compute)", 
                    category="precision"
                )
            else:
                debug.log(
                    "GGUF precision: Dequantizing to FP16 first, then converting to compute dtype", 
                    category="precision"
                )
        else:
            debug.log("Warning: No layers were replaced despite having quantized parameters", 
                     level="WARNING", category="dit", force=True)
    
    return model


def _validate_gguf_architecture(state: Dict[str, torch.Tensor], 
                                model_state: Dict[str, torch.Tensor], debug: Debug) -> None:
    """
    Validate GGUF model architecture matches target model.
    
    Raises:
        ValueError: If architecture mismatch is detected
    """
    key_params = [
        "blocks.0.attn.proj_qkv.vid.weight",
        "blocks.0.attn.proj_qkv.txt.weight", 
        "blocks.0.mlp.vid.proj_in.weight"
    ]
    
    for key in key_params:
        if key in state and key in model_state:
            model_shape = model_state[key].shape
            gguf_shape = _get_tensor_shape(state[key])
            
            if model_shape != gguf_shape:
                # Check if it's just a quantization difference
                if hasattr(state[key], 'tensor_shape') and state[key].tensor_shape == model_shape:
                    continue
                    
                raise ValueError(
                    f"GGUF model architecture mismatch: This GGUF model is incompatible with the current architecture.\n\n"
                    f"Detected mismatch:\n"
                    f"  Parameter: {key}\n"
                    f"  Expected shape: {model_shape}\n"
                    f"  GGUF shape: {gguf_shape}\n\n"
                    f"Possible solutions:\n"
                    f"1. Use a GGUF model that matches the current architecture\n"
                    f"2. Try using a regular FP16 model instead\n"
                    f"3. Verify you're using the correct model variant (3B vs 7B)"
                )
    
    debug.log(f"Architecture check complete, no shape mismatch", category="success")


def _apply_gguf_parameters(model: torch.nn.Module, state: Dict[str, torch.Tensor], 
                           model_state: Dict[str, torch.Tensor], debug: Debug) -> Dict[str, Any]:
    """
    Apply GGUF parameters to model, handling both meta and materialized models.
    
    Returns:
        Statistics dictionary with loaded count, quantized count, and parameter names
    """
    loaded_names = set()
    quantized_count = 0
    
    for name, param in state.items():
        if name not in model_state:
            continue
            
        model_param = model_state[name]
        param_shape = _get_tensor_shape(param)
        
        if param_shape != model_param.shape:
            debug.log(f"Unexpected shape mismatch for {name}: {param_shape} vs {model_param.shape}", 
                     level="ERROR", category="dit", force=True)
            raise ValueError(f"Shape mismatch for parameter {name}")
        
        # Apply parameter based on device type
        with torch.no_grad():
            if model_param.device.type == 'meta':
                _set_parameter_on_meta_model(model, name, param, debug)
            else:
                _set_parameter_on_materialized_model(model, name, param, debug)
        
        loaded_names.add(name)
        if _is_quantized_tensor(param):
            quantized_count += 1
    
    return {
        'loaded': len(loaded_names), 
        'quantized': quantized_count, 
        'loaded_names': loaded_names
    }


def _set_parameter_on_meta_model(model: torch.nn.Module, param_name: str, 
                                 param_value: torch.Tensor, debug: Debug) -> None:
    """Set parameter on meta device model."""
    module, attr_name = _navigate_to_parameter(model, param_name)
    new_param = _create_gguf_parameter(param_value, debug)
    setattr(module, attr_name, new_param)


def _set_parameter_on_materialized_model(model: torch.nn.Module, param_name: str, 
                                         param_value: torch.Tensor, debug: Debug) -> None:
    """Set parameter on already materialized model."""
    module, attr_name = _navigate_to_parameter(model, param_name)
    
    if _is_quantized_tensor(param_value):
        # For quantized tensors, replace with wrapped parameter
        new_param = _create_gguf_parameter(param_value, debug)
        setattr(module, attr_name, new_param)
    else:
        # For regular tensors, just copy
        existing_param = getattr(module, attr_name)
        existing_param.copy_(param_value)


def _navigate_to_parameter(model: torch.nn.Module, param_path: str) -> Tuple[torch.nn.Module, str]:
    """
    Navigate to the module containing a parameter.
    
    Args:
        model: Root model
        param_path: Dot-separated path to parameter
        
    Returns:
        Tuple of (parent module, parameter name)
    """
    path_parts = param_path.split('.')
    module = model
    
    # Navigate to parent module
    for part in path_parts[:-1]:
        module = getattr(module, part)
    
    return module, path_parts[-1]


def _create_gguf_parameter(tensor: torch.Tensor, debug: Optional[Debug]) -> torch.nn.Parameter:
    """
    Create a parameter from a GGUF tensor, preserving quantization info.
    
    Args:
        tensor: GGUF tensor (may be quantized)
        debug: Debug instance for logging
        
    Returns:
        Parameter with GGUF attributes and dequantize method if quantized
    """
    param = torch.nn.Parameter(tensor, requires_grad=False)
    
    # Preserve GGUF attributes if present
    if hasattr(tensor, 'tensor_type'):
        param.tensor_type = tensor.tensor_type
        param.tensor_shape = tensor.tensor_shape
        
        # Add dequantize method for runtime dequantization
        param.gguf_dequantize = _create_dequantize_method(tensor, debug)
    
    return param


def _create_dequantize_method(tensor: torch.Tensor, debug: Optional[Debug]) -> callable:
    """
    Create a dequantization method for a GGUF tensor.
    
    Args:
        tensor: GGUF quantized tensor with tensor_type and tensor_shape attributes
        debug: Debug instance
        
    Returns:
        Callable dequantization method
    """
    def dequantize(device: Optional[torch.device] = None, 
                   dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Dequantize GGUF tensor on demand."""
        if hasattr(tensor, 'dequantize'):
            return tensor.dequantize(device, dtype)
        
        try:
            # Fallback to manual dequantization using gguf library
            numpy_data = tensor.cpu().numpy()
            dequantized = gguf.quants.dequantize(numpy_data, tensor.tensor_type)
            result = torch.from_numpy(dequantized).to(device or tensor.device, dtype)
            result.requires_grad_(False)
            return result.reshape(tensor.tensor_shape)
        except Exception as e:
            if debug:
                debug.log(f"Warning: Could not dequantize tensor: {e}", level="WARNING", category="dit", force=True)
            return tensor.to(device or tensor.device, dtype)
    
    return dequantize


def _get_tensor_shape(tensor: torch.Tensor) -> torch.Size:
    """Get the logical shape of a tensor (handling GGUF quantized tensors)."""
    if hasattr(tensor, 'tensor_shape'):
        return tensor.tensor_shape
    return tensor.shape


def _is_quantized_tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is GGUF quantized."""
    return hasattr(tensor, 'tensor_type') and hasattr(tensor, 'tensor_shape')


def _report_parameter_mismatches(state: Dict[str, torch.Tensor], 
                                 model_state: Dict[str, torch.Tensor], 
                                 loaded_names: set, debug: Debug) -> None:
    """Report any parameter mismatches between GGUF and model."""
    # Check for unmatched GGUF parameters
    unmatched = [name for name in state if name not in model_state]
    if unmatched:
        debug.log(f"Warning: {len(unmatched)} parameters from GGUF not found in model", 
                 level="WARNING", category="dit", force=True)
        debug.log(f"   First few unmatched: {unmatched[:5]}", level="WARNING", category="dit", force=True)
    
    # Check for missing model parameters  
    missing = [name for name in model_state if name not in loaded_names]
    if missing:
        debug.log(f"Warning: {len(missing)} model parameters not loaded from GGUF", 
                 level="WARNING", category="dit", force=True)
        debug.log(f"   First few missing: {missing[:5]}", level="WARNING", category="dit", force=True)


def _initialize_meta_buffers_wrapped(model: torch.nn.Module, target_device: str, debug: Debug) -> None:
    """Initialize meta buffers with timing."""
    debug.start_timer("buffer_init")
    initialized = _initialize_meta_buffers(model, target_device, debug)
    if initialized > 0:
        debug.log(f"Initialized {initialized} non-persistent buffers", category="success")
    debug.end_timer("buffer_init", "Buffer initialization")


def _initialize_meta_buffers(model: torch.nn.Module, target_device: str, debug: Optional[Debug]) -> int:
    """
    Initialize any buffers still on meta device after materialization.
    
    Non-persistent buffers aren't included in state_dict and remain on meta
    device after load_state_dict. This function moves them to the target device.
    
    Args:
        model: Model potentially containing meta device buffers
        target_device: Target device for initialization
        debug: Debug instance for logging
        
    Returns:
        Number of buffers initialized
    """
    initialized_count = 0
    
    # Simply initialize all meta device buffers to zeros on target device
    for name, buffer in model.named_buffers():
        if buffer is not None and buffer.device.type == 'meta':
            # Get the module that owns this buffer
            module_path = name.rsplit('.', 1)[0] if '.' in name else ''
            buffer_name = name.rsplit('.', 1)[1] if '.' in name else name
            
            # Get the actual module
            if module_path:
                module = model
                for part in module_path.split('.'):
                    module = getattr(module, part)
            else:
                module = model
            
            # Create a zero tensor of the same shape on target device
            # This is safe for all non-persistent buffers (caches, dummy tensors, etc.)
            initialized_buffer = torch.zeros_like(buffer, device=target_device)
            module.register_buffer(buffer_name, initialized_buffer, persistent=False)
            initialized_count += 1
    
    return initialized_count


def _apply_model_specific_config(model: torch.nn.Module, runner: VideoDiffusionInfer, 
                                config: OmegaConf, is_dit: bool, 
                                debug: Optional[Debug]) -> torch.nn.Module:
    """
    Apply model-specific configurations and attach to runner.
    
    Args:
        model: Loaded model instance
        runner: Runner to attach model to
        config: Full configuration object
        is_dit: Whether this is a DiT model (vs VAE)
        debug: Debug instance
        
    Returns:
        Configured model
    """
    if is_dit:
        # DiT-specific: Apply FP8 compatibility wrapper
        if not isinstance(model, FP8CompatibleDiT):
            debug.log("Applying FP8/RoPE compatibility wrapper to DiT model", category="setup")
            debug.start_timer("FP8CompatibleDiT")
            model = FP8CompatibleDiT(model, skip_conversion=False, debug=debug)
            debug.end_timer("FP8CompatibleDiT", "FP8/RoPE compatibility wrapper application")
        runner.dit = model
        
    else:
        # VAE-specific configurations
        
        # Set to eval mode (no gradients needed for inference)
        debug.log("VAE model set to eval mode (gradients disabled)", category="vae")
        debug.start_timer("model_requires_grad")
        model.requires_grad_(False).eval()
        debug.end_timer("model_requires_grad", "VAE model set to eval mode")
        
        # Configure causal slicing if available
        if hasattr(model, "set_causal_slicing") and hasattr(config.vae, "slicing"):
            debug.log("Configuring VAE causal slicing for temporal processing", category="vae")
            debug.start_timer("vae_set_causal_slicing")
            model.set_causal_slicing(**config.vae.slicing)
            debug.end_timer("vae_set_causal_slicing", "VAE causal slicing configuration")
        
        # Propagate debug instance to submodules
        model.debug = debug
        _propagate_debug_to_modules(model, debug)
        runner.vae = model
    
    return model


def _propagate_debug_to_modules(module: torch.nn.Module, debug: Optional[Debug]) -> None:
    """
    Propagate debug instance to specific submodules that need it.
    Only targets modules that actually use debug to avoid unnecessary memory overhead.
    
    Args:
        module: Parent module to propagate through
        debug: Debug instance to attach
    """
    if debug is None:
        return  # Early exit if no debug instance
        
    target_modules = {'ResnetBlock3D', 'Upsample3D', 'InflatedCausalConv3d', 'GroupNorm'}
    
    for name, submodule in module.named_modules():
        if submodule.__class__.__name__ in target_modules:
            if not hasattr(submodule, 'debug'):  # Only set if not already present
                submodule.debug = debug