"""
Generation Logic Module for SeedVR2

This module implements a three-phase batch processing pipeline for video upscaling:
- Phase 1: Batch VAE encoding of all input frames
- Phase 2: Batch DiT upscaling of all encoded latents
- Phase 3: Batch VAE decoding of all upscaled latents

This architecture minimizes model swapping overhead by completing each phase
for all batches before moving to the next phase, significantly improving
performance especially when using model offloading.

Key Features:
- Three-phase pipeline (encode-all → upscale-all → decode-all) for efficiency
- Native FP8 pipeline support for 2x speedup and 50% VRAM reduction
- Temporal overlap support for smooth transitions between batches
- Adaptive dtype detection and optimal autocast configuration
- Memory-efficient pre-allocated batch processing
- Advanced video format handling (4n+1 constraint)
"""

import os
import torch
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils.debug import Debug
    from .infer import VideoDiffusionInfer
from torchvision.transforms import Compose, Lambda, Normalize

from ..utils.constants import get_script_directory
from ..common.distributed import get_device
from .model_manager import configure_runner
from ..optimization.memory_manager import (
    clear_memory, 
    release_text_embeddings, 
    manage_model_device, 
    complete_cleanup
)
from ..optimization.performance import (
    optimized_video_rearrange, 
    optimized_single_video_rearrange, 
    optimized_sample_to_image_format
)
from ..common.seed import set_seed
from ..data.image.transforms.divisible_crop import DivisibleCrop
from ..data.image.transforms.na_resize import NaResize
from ..utils.color_fix import wavelet_reconstruction, adaptive_instance_normalization

# Get script directory for embeddings
script_directory = get_script_directory()

def prepare_video_transforms(res_w: int) -> Compose:
    """
    Prepare optimized video transformation pipeline
    
    Args:
        res_w (int): Target resolution width
        
    Returns:
        Compose: Configured transformation pipeline
        
    Features:
        - Resolution-aware upscaling (no downsampling)
        - Proper normalization for model compatibility
        - Memory-efficient tensor operations
    """
    return Compose([
        NaResize(
            resolution=(res_w),
            mode="side",
            # Upsample image, model only trained for high res
            downsample_only=False,
        ),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),  # t c h w -> c t h w (faster than Rearrange)
    ])


def load_text_embeddings(script_directory: str, device: Union[str, torch.device], 
                        dtype: torch.dtype) -> Dict[str, List[torch.Tensor]]:
    """
    Load and prepare text embeddings for generation
    
    Args:
        script_directory (str): Script directory path
        device (str): Target device
        dtype (torch.dtype): Target dtype
        
    Returns:
        dict: Text embeddings dictionary
        
    Features:
        - Adaptive dtype handling
        - Device-optimized loading
        - Memory-efficient embedding preparation
    """
    text_pos_embeds = torch.load(os.path.join(script_directory, 'pos_emb.pt')).to(device, dtype=dtype)
    text_neg_embeds = torch.load(os.path.join(script_directory, 'neg_emb.pt')).to(device, dtype=dtype)
    
    return {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}


def calculate_optimal_batch_params(total_frames: int, batch_size: int, 
                                  temporal_overlap: int) -> Dict[str, Any]:
    """
    Calculate optimal batch processing parameters for 4n+1 constraint.
    
    Args:
        total_frames (int): Total number of frames to process
        batch_size (int): Desired batch size
        temporal_overlap (int): Number of overlapping frames between batches
        
    Returns:
        dict: {
            'step': Effective step size between batches,
            'temporal_overlap': Adjusted temporal overlap,
            'best_batch': Optimal batch size for temporal stability,
            'padding_waste': Total frames that will be padded,
            'is_optimal': Whether current batch_size causes no padding
        }
        
    The 4n+1 constraint (1, 5, 9, 13, 17, 21...) is required by the model.
    Best batch prioritizes temporal stability (larger batches) over padding waste.
    """
    # Calculate step size
    step = batch_size - temporal_overlap
    if step <= 0:
        step = batch_size
        temporal_overlap = 0
    
    # Find all valid 4n+1 batch sizes up to total_frames
    valid_sizes = [i for i in range(1, total_frames + 1) if i % 4 == 1]
    
    # Best batch: largest valid size ≤ total_frames (maximizes temporal stability)
    best_batch = max(valid_sizes) if valid_sizes else 1
    
    # Calculate padding waste for current batch_size
    padding_waste = 0
    current_frame = 0
    
    while current_frame < total_frames:
        frames_in_batch = min(batch_size, total_frames - current_frame)
        
        # Find next 4n+1 target
        if frames_in_batch % 4 == 1:
            target = frames_in_batch
        else:
            target = ((frames_in_batch - 1) // 4 + 1) * 4 + 1
        
        padding_waste += target - frames_in_batch
        current_frame += step if step > 0 else batch_size
    
    return {
        'step': step,
        'temporal_overlap': temporal_overlap,
        'best_batch': best_batch,
        'padding_waste': padding_waste,
        'is_optimal': padding_waste == 0
    }


def cut_videos(videos: torch.Tensor) -> torch.Tensor:
    """
    Correct video cutting respecting the constraint: frames % 4 == 1
    
    Args:
        videos (torch.Tensor): Video tensor to format
        
    Returns:
        torch.Tensor: Properly formatted video tensor
        
    Features:
        - Ensures frames % 4 == 1 constraint for model compatibility
        - Intelligent padding with last frame repetition
        - Memory-efficient tensor operations
    """
    t = videos.size(1)
    
    if t % 4 == 1:
        return videos
    
    # Calculate next valid number (4n + 1)
    padding_needed = (4 - (t % 4)) % 4 + 1
    
    # Apply padding to reach 4n+1 format
    last_frame = videos[:, -1:].expand(-1, padding_needed, -1, -1).contiguous()
    result = torch.cat([videos, last_frame], dim=1)
    
    return result


def check_interrupt(ctx: Dict[str, Any]) -> None:
    """Single interrupt check to avoid redundant imports"""
    if ctx.get('interrupt_fn') is not None:
        ctx['interrupt_fn']()


def prepare_generation_context(device: Union[str, torch.device], 
                              debug: Optional['Debug'] = None) -> Dict[str, Any]:
    """
    Create a generation context for shared state.
    Precision will be lazily initialized when first needed.
    
    Args:
        device: Device string (required, e.g., "cuda:0", "cpu")
        debug: Debug instance for logging
    """
    if device is None:
        raise ValueError("Device must be provided to prepare_generation_context")
    
    try:
        import comfy.model_management
        interrupt_fn = comfy.model_management.throw_exception_if_processing_interrupted
        comfyui_available = True
    except:
        interrupt_fn = None
        comfyui_available = False
    
    ctx = {
        'device': device,
        'compute_dtype': None,
        'autocast_dtype': None,
        'video_transform': None,
        'text_embeds': None,
        'all_transformed_videos': [],
        'all_latents': [],
        'all_upscaled_latents': [],
        'batch_samples': [],
        'final_video': None,
        'comfyui_available': comfyui_available,
        'interrupt_fn': interrupt_fn,  # Store the function reference
    }
    
    if debug:
        debug.log("Initialized generation context", category="setup")
    
    return ctx


def _ensure_precision_initialized(ctx: Dict[str, Any], 
                                runner: 'VideoDiffusionInfer', 
                                debug: Optional['Debug'] = None) -> None:
    """Lazily initialize precision settings if not already done"""
    if ctx.get('compute_dtype') is not None:
        return  # Already initialized
    
    try:
        # Get real dtype of loaded models
        dit_dtype = next(runner.dit.parameters()).dtype
        vae_dtype = next(runner.vae.parameters()).dtype

        # Use BFloat16 for all models
        # - FP8 models: BFloat16 required for arithmetic operations
        # - FP16 models: BFloat16 provides better numerical stability and prevents black frames
        # - BFloat16 models: Already optimal
        if dit_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            ctx['compute_dtype'] = torch.bfloat16
            ctx['autocast_dtype'] = torch.bfloat16
        elif dit_dtype == torch.float16:
            ctx['compute_dtype'] = torch.bfloat16
            ctx['autocast_dtype'] = torch.bfloat16
        else:
            ctx['compute_dtype'] = torch.bfloat16
            ctx['autocast_dtype'] = torch.bfloat16
        
        if debug:
            debug.log(f"Initialized precision: DiT={dit_dtype}, VAE={vae_dtype}, compute={ctx['compute_dtype']}, autocast={ctx['autocast_dtype']}", category="precision")
    except Exception as e:
        # Fallback to safe defaults
        ctx['compute_dtype'] = torch.bfloat16
        ctx['autocast_dtype'] = torch.bfloat16
        
        if debug:
            debug.log(f"Could not detect model dtypes: {e}, falling back to BFloat16", level="WARNING", category="model", force=True)


def setup_device_environment(device: Optional[str] = None, 
                            debug: Optional['Debug'] = None) -> str:
    """
    Setup device environment variables before model loading.
    This must be called before configure_runner.
    
    Args:
        device: Device string (e.g., "cuda:0", "none")
        debug: Debug instance for logging
        
    Returns:
        str: Processed device string
    """
    if device is None:
        device = get_device() if (torch.cuda.is_available() or torch.mps.is_available()) else "cpu"
    
    # Set LOCAL_RANK for distributed compatibility
    if device != "none" and ":" in device:
        os.environ["LOCAL_RANK"] = device.split(":")[1]
    else:
        os.environ["LOCAL_RANK"] = "0"
    
    if debug:
        debug.log(f"Device environment configured: {device}, LOCAL_RANK={os.environ['LOCAL_RANK']}", category="setup")
    
    return device


def prepare_runner(model_name: str, model_dir: str, preserve_vram: bool, 
                  debug: 'Debug', cache_model: bool = False, 
                  block_swap_config: Optional[Dict[str, Any]] = None,
                  vae_tiling_enabled: bool = False, 
                  vae_tile_size: Tuple[int, int] = (512, 512), 
                  vae_tile_overlap: Tuple[int, int] = (64, 64), 
                  cached_runner: Optional['VideoDiffusionInfer'] = None) -> Tuple['VideoDiffusionInfer', bool]:
    """
    Prepare runner with model state management.
    Handles model changes and caching logic.
    
    Args:
        model_name: Name of the model to load
        model_dir: Directory containing models  
        preserve_vram: Whether to preserve VRAM
        debug: Debug instance
        cache_model: Whether to cache model between runs
        block_swap_config: BlockSwap configuration
        vae_tiling_enabled: Enable VAE tiling
        vae_tile_size: VAE tile dimensions
        vae_tile_overlap: VAE tile overlap
        cached_runner: Existing runner instance if caching
        
    Returns:
        tuple: (runner, model_changed) - runner instance and whether model changed
    """    
    model_changed = False
    
    # Check for model change if we have a cached runner
    if cached_runner is not None:
        current_model = getattr(cached_runner, '_model_name', None)
        if current_model != model_name:
            model_changed = True
            debug.log(f"Model changed from {current_model} to {model_name}, clearing cache...", category="cache")
            complete_cleanup(runner=cached_runner, debug=debug, keep_models_in_ram=False)
            cached_runner = None
    
    # Configure runner
    debug.log("Configuring inference runner...", category="runner")
    runner = configure_runner(
        model_name, model_dir, preserve_vram, debug,
        cache_model=cache_model,
        block_swap_config=block_swap_config,
        vae_tiling_enabled=vae_tiling_enabled,
        vae_tile_size=vae_tile_size,
        vae_tile_overlap=vae_tile_overlap,
        cached_runner=cached_runner if cache_model else None
    )
    
    # Store model name for future comparisons
    runner._model_name = model_name
    
    return runner, model_changed


def encode_all_batches(runner: 'VideoDiffusionInfer', ctx: Optional[Dict[str, Any]] = None, 
                       images: Optional[torch.Tensor] = None, batch_size: int = 1, 
                       preserve_vram: bool = False, debug: Optional['Debug'] = None, 
                       progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
                       temporal_overlap: int = 0, res_w: int = 1072, 
                       input_noise_scale: float = 0.0,
                       color_correction: str = "wavelet") -> Dict[str, Any]:
    """
    Phase 1: VAE Encoding for all batches
    
    Encodes video frames to latents in batches, handling temporal overlap and 
    memory optimization. Creates context automatically if not provided.
    
    Args:
        runner: VideoDiffusionInfer instance with loaded models (required)
        ctx: Generation context from prepare_generation_context (required)
        images: Input frames tensor [T, H, W, C] in float16, range [0,1]. 
                Required if ctx doesn't contain 'input_images'
        batch_size: Frames per batch (4n+1 format: 1, 5, 9, 13...)
        preserve_vram: If True, offload VAE between operations
        debug: Debug instance for logging (required)
        progress_callback: Optional callback(current, total, frames, phase_name)
        temporal_overlap: Overlapping frames between batches for continuity
        res_w: Target resolution for shortest edge
        input_noise_scale: Scale for input noise (0.0-1.0). Adds noise to input images
                          before VAE encoding to reduce artifacts at high resolutions.
        color_correction: Color correction method - "wavelet", "adain", or "none" (default: "wavelet")
                         Determines if transformed videos need to be stored for later use.
        
    Returns:
        dict: Context containing:
            - all_transformed_videos: List of (video, original_length) tuples
            - all_latents: List of encoded latents ready for upscaling
            - Other state for subsequent phases
            
    Raises:
        ValueError: If required inputs are missing or invalid
        RuntimeError: If encoding fails
    """
    if debug is None:
        raise ValueError("Debug instance must be provided to encode_all_batches")
    
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 1: VAE encoding ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase1_encoding")

    # Context must be provided
    if ctx is None:
        raise ValueError("Generation context must be provided to encode_all_batches")
    
    # Ensure precision is initialized
    _ensure_precision_initialized(ctx, runner, debug)
    
    # Validate and store inputs
    if images is None and 'input_images' not in ctx:
        raise ValueError("Either images must be provided or ctx must contain 'input_images'")
    
    if images is not None:
        ctx['input_images'] = images
    else:
        images = ctx['input_images']
    
    total_frames = len(images)
    if total_frames == 0:
        raise ValueError("No frames to process")
    
    # Setup video transformation pipeline if not already done
    if ctx.get('video_transform') is None:
        debug.start_timer("video_transform")
        ctx['video_transform'] = prepare_video_transforms(res_w)
        debug.log(f"Initialized video transformation pipeline for {res_w}px", category="setup")
        debug.end_timer("video_transform", "Video transform pipeline initialization")
    
    # Display batch optimization tips
    if total_frames > 0:
        batch_params = calculate_optimal_batch_params(total_frames, batch_size, temporal_overlap)
        if batch_params['padding_waste'] > 0:
            debug.log("", category="none", force=True)
            debug.log(f"Padding waste: {batch_params['padding_waste']}", category="info", force=True)
            debug.log(f"   Why padding? Each batch must be 4n+1 frames (1, 5, 9, 13, 17, 21, ...)", category="info", force=True)
            debug.log(f"   Current batch_size creates partial batches that need padding to meet this constraint", category="info", force=True)
            debug.log(f"   This increases memory usage and processing time unnecessarily", category="info", force=True)
        
        if batch_params['best_batch'] != batch_size and batch_params['best_batch'] <= total_frames:
            debug.log("", category="none", force=True)
            debug.log(f"For {total_frames} frames, use batch_size={batch_params['best_batch']} for better efficiency", category="tip", force=True)
            debug.log(f"   Consider larger batches: better temporal coherence BUT uses more memory", category="tip", force=True)
        
        if batch_params['padding_waste'] > 0 or (batch_params['best_batch'] != batch_size and batch_params['best_batch'] <= total_frames):
            debug.log("", category="none", force=True)
    
    # Calculate batching parameters
    step = batch_size - temporal_overlap if temporal_overlap > 0 else batch_size
    if step <= 0:
        step = batch_size
        temporal_overlap = 0
    
    # Calculate number of batches
    num_encode_batches = 0
    for idx in range(0, total_frames, step):
        end_idx = min(idx + batch_size, total_frames)
        if idx > 0 and end_idx - idx <= temporal_overlap:
            break
        num_encode_batches += 1
    
    # Pre-allocate lists for memory efficiency
    ctx['all_latents'] = [None] * num_encode_batches
    ctx['all_ori_lengths'] = [None] * num_encode_batches
    if color_correction != "none":
        ctx['all_transformed_videos'] = [None] * num_encode_batches
    else:
        ctx['all_transformed_videos'] = None
    
    encode_idx = 0
    
    try:
        # Move VAE to GPU once for all encoding
        manage_model_device(model=runner.vae, target_device=str(ctx['device']), 
                          model_name="VAE", preserve_vram=False, debug=debug,
                          runner=runner)
        
        for batch_idx in range(0, total_frames, step):
            check_interrupt(ctx)
            
            # Calculate indices with temporal overlap
            if batch_idx == 0:
                start_idx = 0
                end_idx = min(batch_size, total_frames)
            else:
                start_idx = batch_idx
                end_idx = min(start_idx + batch_size, total_frames)
                if end_idx - start_idx <= temporal_overlap:
                    break
            
            current_frames = end_idx - start_idx
            
            debug.log(f"Encoding batch {encode_idx+1}/{num_encode_batches}", category="vae", force=True)
            debug.start_timer(f"encode_batch_{encode_idx+1}")
            
            # Process current batch
            video = images[start_idx:end_idx]
            video = video.permute(0, 3, 1, 2).to(ctx['device'], dtype=ctx['compute_dtype'])
            
            # Apply transformations
            transformed_video = ctx['video_transform'](video)

            del video

            # Apply input noise if requested (to reduce artifacts at high resolutions)
            if input_noise_scale > 0:
                debug.log(f"Applying input noise (scale: {input_noise_scale:.2f})", category="video")
                
                # Generate noise matching the video shape
                noise = torch.randn_like(transformed_video)
                
                # Subtle noise amplitude
                noise = noise * 0.05
                
                # Linear blend factor: 0 at scale=0, 0.5 at scale=1
                blend_factor = input_noise_scale * 0.5
                
                # Apply blend
                transformed_video = transformed_video * (1 - blend_factor) + (transformed_video + noise) * blend_factor
                
                del noise

            ori_length = transformed_video.size(1)
            
            # Log sequence info
            t = transformed_video.size(1)
            debug.log(f"Sequence of {t} frames", category="video", force=True)

            # Handle 4n+1 constraint
            if t % 4 != 1:
                target = ((t-1)//4+1)*4+1
                padding_frames = target - t
                debug.log(f"Applying padding: {padding_frames} frame{'s' if padding_frames != 1 else ''} added ({t} -> {target})", category="video", force=True)
                transformed_video = cut_videos(transformed_video)
            
            # Store original length for proper trimming later
            ctx['all_ori_lengths'][encode_idx] = ori_length
            
            # Store transformed video on CPU if needed for color correction
            if color_correction != "none":
                # Move to CPU to free VRAM
                transformed_video_cpu = transformed_video.to('cpu', non_blocking=False)
                ctx['all_transformed_videos'][encode_idx] = transformed_video_cpu
            
            # Encode to latents
            cond_latents = runner.vae_encode([transformed_video])
            
            # Offload latents to CPU to save VRAM between phases
            ctx['all_latents'][encode_idx] = cond_latents[0].to('cpu', non_blocking=False)
            
            del cond_latents, transformed_video
            
            debug.end_timer(f"encode_batch_{encode_idx+1}", f"Encoded batch {encode_idx+1}")
            
            if progress_callback:
                progress_callback(encode_idx+1, num_encode_batches, 
                                current_frames, "Phase 1: Encoding")
            
            encode_idx += 1
            
    except Exception as e:
        debug.log(f"Error in Phase 1 (Encoding): {e}", level="ERROR", category="error", force=True)
        raise
    finally:
        # Always offload VAE if needed
        if preserve_vram:
            manage_model_device(model=runner.vae, target_device='cpu', 
                              model_name="VAE", preserve_vram=preserve_vram, debug=debug,
                              runner=runner)
    
    debug.end_timer("phase1_encoding", "Phase 1: VAE encoding complete", show_breakdown=True)
    debug.log_memory_state("After phase 1 (VAE encoding)", show_tensors=False)
    
    return ctx


def upscale_all_batches(runner: 'VideoDiffusionInfer', ctx: Optional[Dict[str, Any]] = None, 
                        preserve_vram: bool = False, debug: Optional['Debug'] = None, 
                        progress_callback: Optional[Callable[[int, int, int, str], None]] = None, 
                        cfg_scale: float = 1.0, seed: int = 100, 
                        latent_noise_scale: float = 0.0) -> Dict[str, Any]:
    """
    Phase 2: DiT Upscaling for all encoded batches.
    
    Processes all encoded latents through the diffusion model for upscaling.
    Requires context from encode_all_batches with encoded latents.
    
    Args:
        runner: VideoDiffusionInfer instance with loaded models (required)
        ctx: Context from encode_all_batches containing latents (required)
        preserve_vram: If True, offload DiT between operations
        debug: Debug instance for logging (required)
        progress_callback: Optional callback(current, total, frames, phase_name)
        cfg_scale: Classifier-free guidance scale (default: 1.0)
        seed: Random seed for noise generation
        latent_noise_scale: Noise scale for latent space augmentation (0.0-1.0).
                           Adds noise during diffusion conditioning. Can soften details
                           but may help with certain artifacts. 0.0 = no noise (crisp),
                           1.0 = maximum noise (softer)
        
    Returns:
        dict: Updated context containing:
            - all_upscaled_latents: List of upscaled latents ready for decoding
            - Preserved state from encoding phase
            
    Raises:
        ValueError: If context is missing or has no encoded latents
        RuntimeError: If upscaling fails
    """
    if debug is None:
        raise ValueError("Debug instance must be provided to upscale_all_batches")
    
    if ctx is None:
        raise ValueError("Context is required for upscale_all_batches. Run encode_all_batches first.")
    
    # Ensure precision is initialized
    _ensure_precision_initialized(ctx, runner, debug)
    
    # Validate we have encoded latents
    if 'all_latents' not in ctx or not ctx['all_latents']:
        raise ValueError("No encoded latents found. Run encode_all_batches first.")
    
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 2: DiT upscaling ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase2_upscaling")
    
    # Load text embeddings if not already loaded
    if ctx.get('text_embeds') is None:
        ctx['text_embeds'] = load_text_embeddings(script_directory, ctx['device'], ctx['compute_dtype'])
        debug.log("Loaded text embeddings for DiT", category="dit")
    
    # Configure diffusion parameters
    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion(dtype=ctx['compute_dtype'])
    
    # Set seed for generation
    set_seed(seed)
    
    # Count valid latents
    num_valid_latents = len([l for l in ctx['all_latents'] if l is not None])

    # Safety check for empty latents
    if num_valid_latents == 0:
        debug.log("No valid latents to upscale", level="WARNING", category="dit", force=True)
        ctx['all_upscaled_latents'] = []
        return ctx
    
    # Pre-allocate list for upscaled latents
    ctx['all_upscaled_latents'] = [None] * num_valid_latents
    
    upscale_idx = 0
    
    try:
        # Move DiT to GPU once for all upscaling
        manage_model_device(model=runner.dit, target_device=str(ctx['device']), 
                            model_name="DiT", preserve_vram=False, debug=debug,
                            runner=runner)
        
        for batch_idx, latent in enumerate(ctx['all_latents']):
            if latent is None:
                continue
            
            check_interrupt(ctx)
            
            debug.log(f"Upscaling batch {upscale_idx+1}/{num_valid_latents}", category="generation", force=True)
            debug.start_timer(f"upscale_batch_{upscale_idx+1}")
            
            # Move latent from CPU to device with correct dtype
            latent = latent.to(ctx['device'], dtype=ctx['compute_dtype'], non_blocking=False)
            
            # Generate noise
            if torch.mps.is_available():
                base_noise = torch.randn_like(latent, dtype=ctx['compute_dtype'])
            else:
                with torch.cuda.device(ctx['device']):
                    base_noise = torch.randn_like(latent, dtype=ctx['compute_dtype'])
            
            noises = [base_noise]
            aug_noises = [base_noise * 0.1 + torch.randn_like(base_noise) * 0.05]
            
            # Log latent noise application if enabled
            if latent_noise_scale > 0:
                debug.log(f"Applying latent noise (scale: {latent_noise_scale:.3f})", category="generation")
            
            def _add_noise(x, aug_noise):
                if latent_noise_scale == 0.0:
                    return x
                t = torch.tensor([1000.0], device=ctx['device'], dtype=ctx['compute_dtype']) * latent_noise_scale
                shape = torch.tensor(x.shape[1:], device=ctx['device'])[None]
                t = runner.timestep_transform(t, shape)
                x = runner.schedule.forward(x, aug_noise, t)
                del t, shape
                return x
            
            # Generate condition
            condition = runner.get_condition(
                noises[0],
                task="sr",
                latent_blur=_add_noise(latent, aug_noises[0]),
            )
            conditions = [condition]
            
            # Run inference
            debug.start_timer(f"dit_inference_{upscale_idx+1}")
            with torch.no_grad():
                with torch.autocast(str(ctx['device']), ctx['autocast_dtype'], enabled=True):
                    upscaled = runner.inference(
                        noises=noises,
                        conditions=conditions,
                        **ctx['text_embeds'],
                    )
            debug.end_timer(f"dit_inference_{upscale_idx+1}", f"DiT inference {upscale_idx+1}")
            
            # Store upscaled result on CPU to save VRAM
            ctx['all_upscaled_latents'][upscale_idx] = upscaled[0].to('cpu', non_blocking=False)
            
            # Free original latent
            ctx['all_latents'][batch_idx] = None
            
            del noises, aug_noises, latent, conditions, condition, base_noise, upscaled
            
            if preserve_vram and ctx['all_upscaled_latents'][upscale_idx].shape[0] > 1:
                clear_memory(debug=debug, deep=True, force=True, timer_name=f"upscale_all_batches - batch {upscale_idx+1} - deep")
            
            debug.end_timer(f"upscale_batch_{upscale_idx+1}", f"Upscaled batch {upscale_idx+1}")
            
            if progress_callback:
                progress_callback(upscale_idx+1, num_valid_latents,
                                1, "Phase 2: Upscaling")
            
            upscale_idx += 1
            
    except Exception as e:
        debug.log(f"Error in Phase 2 (Upscaling): {e}", level="ERROR", category="error", force=True)
        raise
    finally:
        # Always offload DiT if needed
        if preserve_vram:
            manage_model_device(model=runner.dit, target_device='cpu', 
                              model_name="DiT", preserve_vram=preserve_vram, debug=debug,
                              runner=runner)
        clear_memory(debug=debug, deep=False, force=True, timer_name=f"upscale_all_batches - finally - minimal")
    
    debug.end_timer("phase2_upscaling", "Phase 2: DiT upscaling complete", show_breakdown=True)
    debug.log_memory_state("After phase 2 (DiT upscaling)", show_tensors=False)
    
    return ctx


def decode_all_batches(runner: 'VideoDiffusionInfer', ctx: Optional[Dict[str, Any]] = None, 
                       preserve_vram: bool = False, debug: Optional['Debug'] = None, 
                       progress_callback: Optional[Callable[[int, int, int, str], None]] = None, 
                       color_correction: str = "wavelet") -> Dict[str, Any]:
    """
    Phase 3: VAE Decoding and Final Video Assembly.
    
    Decodes all upscaled latents back to pixel space and assembles final video.
    Requires context from upscale_all_batches with upscaled latents.
    
    Args:
        runner: VideoDiffusionInfer instance with loaded models (required)
        ctx: Context from upscale_all_batches containing upscaled latents (required)
        preserve_vram: If True, offload VAE between operations
        debug: Debug instance for logging (required)
        progress_callback: Optional callback(current, total, frames, phase_name)
        color_correction: Color correction method - "wavelet", "adain", or "none" (default: "wavelet")
        
    Returns:
        dict: Updated context containing:
            - final_video: Assembled video tensor [T, H, W, C] in float16, range [0,1]
            - All intermediate storage cleared for memory efficiency
            
    Raises:
        ValueError: If context is missing or has no upscaled latents
        RuntimeError: If decoding fails
    """
    if debug is None:
        raise ValueError("Debug instance must be provided to decode_all_batches")
    
    if ctx is None:
        raise ValueError("Context is required for decode_all_batches. Run upscale_all_batches first.")
    
    # Ensure precision is initialized
    _ensure_precision_initialized(ctx, runner, debug)
    
    # Validate we have upscaled latents
    if 'all_upscaled_latents' not in ctx or not ctx['all_upscaled_latents']:
        raise ValueError("No upscaled latents found. Run upscale_all_batches first.")
    
    # Validate we have transformed videos if color correction is enabled
    if color_correction != "none":
        if 'all_transformed_videos' not in ctx or not ctx['all_transformed_videos']:
            raise ValueError("No transformed videos found for color correction. Context corrupted.")
    
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 3: VAE decoding ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase3_decoding")

    # Count valid latents
    num_valid_latents = len([l for l in ctx['all_upscaled_latents'] if l is not None])
    
    # Pre-allocate to match original batches (use ori_lengths which is always available)
    num_batches = len([l for l in ctx['all_ori_lengths'] if l is not None])
    ctx['batch_samples'] = [None] * num_batches
    
    decode_idx = 0
    
    try:
        # Move VAE to GPU once for all decoding
        manage_model_device(model=runner.vae, target_device=str(ctx['device']), 
                          model_name="VAE", preserve_vram=False, debug=debug,
                          runner=runner)
        
        for batch_idx, upscaled_latent in enumerate(ctx['all_upscaled_latents']):
            if upscaled_latent is None:
                continue
            
            check_interrupt(ctx)
            
            debug.log(f"Decoding batch {decode_idx+1}/{num_valid_latents}", category="vae", force=True)
            debug.start_timer(f"decode_batch_{decode_idx+1}")
            
            # Move latent to device with correct dtype for decoding
            upscaled_latent = upscaled_latent.to(ctx['device'], dtype=ctx['compute_dtype'], non_blocking=False)
            
            # Decode latent
            debug.start_timer("vae_decode")
            samples = runner.vae_decode([upscaled_latent], preserve_vram=preserve_vram)
            debug.end_timer("vae_decode", "VAE decode")
            
            # Convert to Float16 for efficiency
            if samples and len(samples) > 0 and samples[0].dtype != torch.float16:
                debug.log(f"Converting from {samples[0].dtype} to Float16", category="precision")
                samples = [sample.to(torch.float16, non_blocking=False) for sample in samples]
            
            # Process samples
            samples = optimized_video_rearrange(samples)
            
            # Post-process samples
            for i, sample in enumerate(samples):
                # Get original length for trimming (always available)
                video_idx = min(batch_idx, len(ctx['all_ori_lengths']) - 1)
                ori_length = ctx['all_ori_lengths'][video_idx]
                
                # Trim to original length if necessary
                if ori_length < sample.shape[0]:
                    sample = sample[:ori_length]
                
                # Apply color correction if enabled
                if color_correction != "none" and ctx['all_transformed_videos'] is not None:
                    # Get transformed video from CPU
                    transformed_video = ctx['all_transformed_videos'][video_idx]
                    
                    # Move to GPU for color correction
                    transformed_video = transformed_video.to(ctx['device'], non_blocking=False)
                    
                    # Adjust transformed video to handles cases when padding was applied during encoding
                    if transformed_video.shape[1] != sample.shape[0]:
                        debug.log(f"Trimming transformed video from {transformed_video.shape[1]} to {sample.shape[0]} frames to match original input", 
                                category="video")
                        transformed_video = transformed_video[:, :sample.shape[0]]
                    
                    input_video = [optimized_single_video_rearrange(transformed_video)]
                    
                    # Apply selected color correction method
                    debug.start_timer(f"color_correction_{color_correction}")
                    
                    if color_correction == "wavelet":
                        debug.log("Applying wavelet color reconstruction", category="video")
                        sample = wavelet_reconstruction(sample, input_video[0], debug)
                    elif color_correction == "adain":
                        debug.log("Applying AdaIN color correction", category="video")
                        sample = adaptive_instance_normalization(sample, input_video[0])
                    else:
                        debug.log(f"Unknown color correction method: {color_correction}", level="WARNING", category="video", force=True)
                    
                    debug.end_timer(f"color_correction_{color_correction}", f"Color correction ({color_correction})")
                    
                    # Free the transformed video
                    ctx['all_transformed_videos'][video_idx] = None
                    del input_video, transformed_video
                
                else:
                    debug.log("Color correction disabled (set to none)", category="video")
                
                # Free the original length entry
                ctx['all_ori_lengths'][video_idx] = None
                
                # Convert to final format
                sample = optimized_sample_to_image_format(sample)
                sample = sample.clip(-1, 1).mul_(0.5).add_(0.5)
                sample_cpu = sample.to(torch.float16).to("cpu")
                del sample
                
                # Store by index for pre-allocation
                ctx['batch_samples'][decode_idx] = sample_cpu
            
            # Free the upscaled latent
            ctx['all_upscaled_latents'][batch_idx] = None
            del upscaled_latent, samples
            
            debug.end_timer(f"decode_batch_{decode_idx+1}", f"Decoded batch {decode_idx+1}")
            
            if progress_callback:
                progress_callback(decode_idx+1, num_valid_latents,
                                1, "Phase 3: Decoding")
            
            decode_idx += 1
            
    except Exception as e:
        debug.log(f"Error in Phase 3 (Decoding): {e}", level="ERROR", category="error", force=True)
        raise
    finally:
        # Always offload VAE if needed
        if preserve_vram:
            manage_model_device(model=runner.vae, target_device='cpu', 
                              model_name="VAE", preserve_vram=preserve_vram, debug=debug,
                              runner=runner)
        
        # Always clean up intermediate storage
        if 'all_latents' in ctx:
            del ctx['all_latents']
        if 'all_upscaled_latents' in ctx:
            del ctx['all_upscaled_latents']
        if 'all_transformed_videos' in ctx:
            del ctx['all_transformed_videos']
        if 'all_ori_lengths' in ctx:
            del ctx['all_ori_lengths']
    
    debug.log("Assembling final video from decoded batches...", category="video")

    # Merge all batch results into final video
    if ctx['batch_samples'] and any(s is not None for s in ctx['batch_samples']):
        valid_samples = [s for s in ctx['batch_samples'] if s is not None]
        total_frames = sum(batch.shape[0] for batch in valid_samples)
        
        if total_frames > 0:
            sample_shape = valid_samples[0].shape
            H, W, C = sample_shape[1], sample_shape[2], sample_shape[3]
            
            debug.log(f"Total frames: {total_frames}, shape per frame: {H}x{W}x{C}", category="info")
            
            # Pre-allocate final tensor
            ctx['final_video'] = torch.empty((total_frames, H, W, C), dtype=torch.float16)
            
            # Copy batch results into final tensor
            current_idx = 0
            for batch in valid_samples:
                batch_frames = batch.shape[0]
                ctx['final_video'][current_idx:current_idx + batch_frames] = batch
                current_idx += batch_frames
            
            final_shape = ctx['final_video'].shape
            Hf, Wf, Cf = final_shape[1], final_shape[2], final_shape[3]

            debug.log(f"Final video assembled: Total frames: {total_frames}, shape per frame: {Hf}x{Wf}x{Cf}", category="video", force=True)
        else:
            ctx['final_video'] = torch.empty((0, 0, 0, 0), dtype=torch.float16)
            debug.log("No frame to assemble", level="WARNING", category="video", force=True)
    else:
        ctx['final_video'] = torch.empty((0, 0, 0, 0), dtype=torch.float16)
        debug.log("No samples to assemble", level="WARNING", category="video", force=True)
    
    # Clean up batch samples
    ctx['batch_samples'].clear()
    
    # Clean up video transform
    if ctx.get('video_transform'):
        for transform in ctx['video_transform'].transforms:
            if hasattr(transform, '__dict__'):
                transform.__dict__.clear()
        ctx['video_transform'] = None
    
    debug.end_timer("phase3_decoding", "Phase 3: VAE decoding complete", show_breakdown=True)
    debug.log_memory_state("After phase 3 (VAE decoding)", show_tensors=False)
    
    return ctx