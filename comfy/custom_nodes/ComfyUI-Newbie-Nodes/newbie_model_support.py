
import torch
import comfy.sd
from comfy.model_patcher import ModelPatcher

print("--- NewBie Native Support Initializing ---")

# =================================================================================
# ==                      STEP 1: MODEL CREATION FUNCTION                        ==
# =================================================================================
# This function correctly instantiates your NextDiT model by dynamically
# inferring parameters from the state dictionary itself, eliminating the
# need for any external config files.

def create_newbie_model(state_dict):
    """
    Creates the NextDiT model instance with parameters inferred from the model file.
    """
    try:
        from models.model import NextDiT_models
    except ImportError as e:
        raise Exception("Could not import NextDiT model definition. Make sure your project is in sys.path.", e)

    # --- Infer critical parameters from the state_dict --- 
    # This is the key to making it work without a config file.
    
    # Infer the main dimension from the x_embedder's output features
    inferred_dim = state_dict["x_embedder.weight"].shape[0]
    
    # Infer the caption feature dimension from the caption embedder's input features
    inferred_cap_feat_dim = state_dict["cap_embedder.1.weight"].shape[1]

    print(f"[NewBie Support] Inferred Parameters: dim={inferred_dim}, cap_feat_dim={inferred_cap_feat_dim}")

    # --- Hardcode the rest of the architecture parameters ---
    # This is a stable and reliable approach for a specific model architecture.
    model_params = {
        "architecture": "NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP",
        "patch_size": 2,
        "in_channels": 16,
        "dim": inferred_dim, # Use inferred value
        "n_heads": 24,
        "n_kv_heads": 8,
        "n_layers": 36,
        "axes_dims": [32, 32, 32],
        "axes_lens": [1024, 512, 512],
        "qk_norm": True,
        "cap_feat_dim": inferred_cap_feat_dim, # Use inferred value
        "clip_text_dim": 1024,
        "clip_img_dim": 1024,
    }

    architecture = model_params.pop("architecture")
    model_builder = NextDiT_models[architecture]
    
    # Create the model instance with the correct parameters
    model = model_builder(**model_params)
    return model

# =================================================================================
# ==                         STEP 2: INTERFACE WRAPPER                           ==
# =================================================================================
# This class adapts your model's interface to what ComfyUI's sampler expects.
class NextDiTWrapper:
    def __init__(self, model):
        self.model = model
        self.model_type = "eps"

    def apply_model(self, x, t, cond, uncond, cond_scale, **kwargs):
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        cond_feats = cond[0][0]
        uncond_feats = uncond[0][0]
        cap_feats = torch.cat([cond_feats['cap_feats'], uncond_feats['cap_feats']])
        cap_mask = torch.cat([cond_feats['cap_mask'], uncond_feats['cap_mask']])
        clip_kwargs = {}
        if 'clip_text_pooled' in cond_feats and cond_feats['clip_text_pooled'] is not None:
            clip_kwargs['clip_text_pooled'] = torch.cat([cond_feats['clip_text_pooled'], uncond_feats['clip_text_pooled']])
        output = self.model.forward_with_cfg(x_in, t_in, cap_feats, cap_mask, cond_scale, **clip_kwargs)
        return output[:x.shape[0]]

# =================================================================================
# ==                       STEP 3: CONVERSION FUNCTION                           ==
# =================================================================================
# This function orchestrates the entire conversion process.

def convert_newbie_checkpoint(state_dict):
    print("[NewBie Support] Converting NewBie (NextDiT) checkpoint.")
    # 1. Create the model with the correct, inferred architecture
    model = create_newbie_model(state_dict)
    # 2. Load the weights into the correctly-sized model skeleton
    model.load_state_dict(state_dict, strict=True)
    # 3. Wrap the model to make it compatible with ComfyUI's sampler
    wrapped_model = NextDiTWrapper(model)
    # 4. Return the final, usable ModelPatcher object
    return ModelPatcher(wrapped_model)

# =================================================================================
# ==                        STEP 4: REGISTRATION                              ==
# =================================================================================
# This is the most important part. We are registering our model with ComfyUI's
# official model loader system.

# Define a detection function. It returns True if the state_dict is ours.
def is_newbie_checkpoint(state_dict):
    # A robust way to check is to look for unique, essential keys.
    required_keys = ["cap_embedder.1.weight", "x_embedder.weight", "final_layer.linear.weight"]
    return all(key in state_dict for key in required_keys)

# Create the entry for the official ComfyUI converter settings
NEWBIE_CONVERTER_SETTINGS = {
    "detection": is_newbie_checkpoint,
    "convert": convert_newbie_checkpoint,
}

# Add our converter to ComfyUI's list of converters
comfy.sd.MODEL_CONVERT_SETTINGS.insert(0, NEWBIE_CONVERTER_SETTINGS)

print("--- NewBie Native Support has been successfully installed. ---")
