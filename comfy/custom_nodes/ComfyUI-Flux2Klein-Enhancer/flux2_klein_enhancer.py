"""
FLUX.2 Klein Conditioning Enhancer - Fixed Version

Built from empirical diagnostic data:
- Conditioning shape: [batch, 512, 12288]
- Active text region: Dynamic (check attention_mask), typically ~77 tokens
- Padding region: Remaining positions (std ~2.3)
- Metadata: pooled_output, attention_mask, reference_latents (for edit)
- Text encoder: Qwen3 8B (4096 hidden → 12288 projected)

CRITICAL FIX: Previous version had mean-recentering that undid the scaling.
This version uses direct operations with no reversion loops.
"""

import torch
import gc

try:
    import comfy.model_management as mm
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False


class Flux2KleinEnhancer:
    """
    Conditioning enhancer for FLUX.2 Klein.
    
    Operations:
    - magnitude: Direct scaling of active region embeddings
    - contrast: Amplify differences between tokens
    - normalize: Equalize token magnitudes
    
    All operations modify the active text region only (padding untouched).
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        devices = ["auto", "cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")

        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "magnitude": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                    "tooltip": "Scale active region embeddings. >1 = stronger prompt, <1 = weaker."
                }),
                "contrast": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Amplify differences between tokens. >0 = sharper concepts."
                }),
                "normalize_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Equalize token magnitudes. Higher = more balanced emphasis."
                }),
            },
            "optional": {
                "edit_text_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                    "tooltip": "[IMAGE EDIT] Additional scaling for edit mode. <1 = preserve original, >1 = follow prompt."
                }),
                "active_end_override": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 512,
                    "step": 1,
                    "tooltip": "Override active region end. 0 = auto-detect from attention_mask."
                }),
                "low_vram": ("BOOLEAN", {"default": False}),
                "device": (devices, {"default": "auto"}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "enhance"
    CATEGORY = "conditioning/flux2klein"

    def _get_active_end(self, cond_shape, meta, override):
        """Determine where active tokens end."""
        if override > 0:
            return min(override, cond_shape[1])
        
        # Try to get from attention mask
        attn_mask = meta.get("attention_mask", None)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # Find last non-zero position
                nonzero = attn_mask[0].nonzero()
                if len(nonzero) > 0:
                    return int(nonzero[-1].item()) + 1
        
        # Default: assume 77 (typical CLIP-style max)
        return min(77, cond_shape[1])

    def enhance(self, conditioning, magnitude=1.0, contrast=0.0, normalize_strength=0.0,
                edit_text_weight=1.0, active_end_override=0, low_vram=False, 
                device="auto", debug=False):
        
        if not conditioning:
            return (conditioning,)
        
        # Check if anything needs to be done
        no_op = (
            magnitude == 1.0 and 
            contrast == 0.0 and 
            normalize_strength == 0.0 and
            edit_text_weight == 1.0
        )
        if no_op:
            if debug:
                print("[Flux2KleinEnhancer] All parameters neutral, passing through")
            return (conditioning,)
        
        # Device setup
        if device == "auto":
            if HAS_COMFY:
                device = mm.get_torch_device()
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        
        compute_dtype = torch.float16 if low_vram and device.type == "cuda" else torch.float32
        
        output = []
        
        for idx, (cond_tensor, meta) in enumerate(conditioning):
            original_dtype = cond_tensor.dtype
            cond = cond_tensor.to(device, dtype=compute_dtype)
            
            if len(cond.shape) != 3:
                if debug:
                    print(f"[Flux2KleinEnhancer] Item {idx}: unexpected shape {cond.shape}, skipping")
                output.append((cond_tensor, meta))
                continue
            
            batch, seq_len, embed_dim = cond.shape
            
            # Get active region
            active_end = self._get_active_end(cond.shape, meta, active_end_override)
            is_edit_mode = "reference_latents" in meta and meta["reference_latents"] is not None
            
            if debug:
                print(f"\n{'='*50}")
                print(f"Flux2KleinEnhancer Item {idx}")
                print(f"{'='*50}")
                print(f"Shape: {cond.shape}")
                print(f"Active region: 0 to {active_end}")
                print(f"Edit mode: {is_edit_mode}")
                if active_end < seq_len:
                    print(f"Active std: {cond[:, :active_end, :].std().item():.4f}")
                    print(f"Padding std: {cond[:, active_end:, :].std().item():.4f}")
            
            # Extract active region only
            active = cond[:, :active_end, :].clone()
            
            if debug:
                orig_norm = active.norm(dim=-1).mean().item()
                print(f"\nBefore modifications:")
                print(f"  Active region mean norm: {orig_norm:.4f}")
            
            # ============================================================
            # OPERATION 1: Contrast (amplify token differences)
            # ============================================================
            if contrast != 0.0:
                seq_mean = active.mean(dim=1, keepdim=True)
                deviation = active - seq_mean
                active = seq_mean + deviation * (1.0 + contrast)
                
                if debug:
                    print(f"\nContrast ({contrast:+.2f}): deviation scaled by {1.0 + contrast:.2f}")
            
            # ============================================================
            # OPERATION 2: Normalize (equalize token magnitudes)
            # ============================================================
            if normalize_strength > 0.0:
                token_norms = active.norm(dim=-1, keepdim=True)
                mean_norm = token_norms.mean()
                normalized = active / (token_norms + 1e-8) * mean_norm
                active = active * (1.0 - normalize_strength) + normalized * normalize_strength
                
                if debug:
                    norm_var_before = token_norms.var().item()
                    norm_var_after = active.norm(dim=-1).var().item()
                    print(f"\nNormalize ({normalize_strength:.2f}): norm variance {norm_var_before:.4f} -> {norm_var_after:.4f}")
            
            # ============================================================
            # OPERATION 3: Magnitude (direct scaling)
            # ============================================================
            if magnitude != 1.0:
                active = active * magnitude
                if debug:
                    print(f"\nMagnitude ({magnitude:.2f}): all active tokens scaled")
            
            # ============================================================
            # OPERATION 4: Edit mode text weight (additional scaling)
            # ============================================================
            if is_edit_mode and edit_text_weight != 1.0:
                active = active * edit_text_weight
                if debug:
                    print(f"\nEdit text weight ({edit_text_weight:.2f}): applied for image edit mode")
            
            # Write back to full tensor
            result = cond.clone()
            result[:, :active_end, :] = active
            
            if debug:
                final_norm = result[:, :active_end, :].norm(dim=-1).mean().item()
                diff = (result - cond).abs()
                print(f"\nFinal state:")
                print(f"  Active region mean norm: {orig_norm:.4f} -> {final_norm:.4f}")
                print(f"  Output change: mean={diff.mean().item():.6f}, max={diff.max().item():.6f}")
            
            output.append((result.to("cpu", dtype=original_dtype), meta))
            
            del cond, active, result
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        return (output,)


class Flux2KleinDetailController:
    """
    Regional control for FLUX.2 Klein conditioning.
    
    Divides active tokens into regions:
    - Front: Subject/main concept
    - Mid: Details/modifiers
    - End: Style/quality terms
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        devices = ["auto", "cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")

        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            },
            "optional": {
                "front_mult": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                    "tooltip": "First 25% of active tokens"
                }),
                "mid_mult": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                    "tooltip": "Middle 50% of active tokens"
                }),
                "end_mult": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                    "tooltip": "Last 25% of active tokens"
                }),
                "emphasis_start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Start of custom emphasis region"
                }),
                "emphasis_end": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "tooltip": "End of custom emphasis region (0 = disabled)"
                }),
                "emphasis_mult": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Multiplier for emphasis region"
                }),
                "low_vram": ("BOOLEAN", {"default": False}),
                "device": (devices, {"default": "auto"}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "control"
    CATEGORY = "conditioning/flux2klein"

    def control(self, conditioning, front_mult=1.0, mid_mult=1.0, end_mult=1.0,
                emphasis_start=0, emphasis_end=0, emphasis_mult=1.0,
                low_vram=False, device="auto", debug=False):
        
        if not conditioning:
            return (conditioning,)
        
        # Check if anything needs to be done
        no_op = (
            front_mult == 1.0 and 
            mid_mult == 1.0 and 
            end_mult == 1.0 and
            (emphasis_end == 0 or emphasis_mult == 1.0)
        )
        if no_op:
            if debug:
                print("[Flux2KleinDetailController] All parameters neutral, passing through")
            return (conditioning,)
        
        # Device setup
        if device == "auto":
            if HAS_COMFY:
                device = mm.get_torch_device()
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        
        compute_dtype = torch.float16 if low_vram and device.type == "cuda" else torch.float32
        
        output = []
        
        for idx, (cond_tensor, meta) in enumerate(conditioning):
            original_dtype = cond_tensor.dtype
            cond = cond_tensor.to(device, dtype=compute_dtype)
            
            if len(cond.shape) != 3:
                output.append((cond_tensor, meta))
                continue
            
            batch, seq_len, embed_dim = cond.shape
            
            # Get active region from attention mask
            attn_mask = meta.get("attention_mask", None)
            if attn_mask is not None and attn_mask.dim() == 2:
                nonzero = attn_mask[0].nonzero()
                active_end = int(nonzero[-1].item()) + 1 if len(nonzero) > 0 else 77
            else:
                active_end = min(77, seq_len)
            
            active = cond[:, :active_end, :].clone()
            num_tokens = active.shape[1]
            
            if debug:
                print(f"\n{'='*50}")
                print(f"Flux2KleinDetailController Item {idx}")
                print(f"{'='*50}")
                print(f"Active tokens: {num_tokens}")
            
            # Calculate region boundaries
            front_end = int(num_tokens * 0.25)
            mid_end = int(num_tokens * 0.75)
            
            if debug:
                print(f"Regions: front=[0:{front_end}], mid=[{front_end}:{mid_end}], end=[{mid_end}:{num_tokens}]")
            
            # Apply regional multipliers
            if front_mult != 1.0 and front_end > 0:
                active[:, :front_end, :] *= front_mult
                if debug:
                    print(f"  Front: x{front_mult}")
            
            if mid_mult != 1.0 and mid_end > front_end:
                active[:, front_end:mid_end, :] *= mid_mult
                if debug:
                    print(f"  Mid: x{mid_mult}")
            
            if end_mult != 1.0 and num_tokens > mid_end:
                active[:, mid_end:, :] *= end_mult
                if debug:
                    print(f"  End: x{end_mult}")
            
            # Custom emphasis region
            if emphasis_end > 0 and emphasis_mult != 1.0:
                emp_start = max(0, min(emphasis_start, num_tokens - 1))
                emp_end = max(emp_start + 1, min(emphasis_end, num_tokens))
                active[:, emp_start:emp_end, :] *= emphasis_mult
                if debug:
                    print(f"  Emphasis [{emp_start}:{emp_end}]: x{emphasis_mult}")
            
            # Write back
            result = cond.clone()
            result[:, :active_end, :] = active
            
            if debug:
                diff = (result - cond).abs()
                print(f"Output change: mean={diff.mean().item():.6f}, max={diff.max().item():.6f}")
            
            output.append((result.to("cpu", dtype=original_dtype), meta))
            
            del cond, active, result
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        return (output,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "Flux2KleinEnhancer": Flux2KleinEnhancer,
    "Flux2KleinDetailController": Flux2KleinDetailController,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinEnhancer": "FLUX.2 Klein Enhancer",
    "Flux2KleinDetailController": "FLUX.2 Klein Detail Controller",
}
