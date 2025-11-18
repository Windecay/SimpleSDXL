import torch
import torch.nn as nn
from accelerate import init_empty_weights

#based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/quantizers/gguf/utils.py
def _replace_linear(model, compute_dtype, state_dict, prefix="", patches=None, scale_weights=None, compile_args=None):
   
    has_children = list(model.children())
    if not has_children:
        return
    
    allow_compile = False

    for name, module in model.named_children():
        if compile_args is not None:
            allow_compile = compile_args.get("allow_unmerged_lora_compile", False)
        module_prefix = prefix + name + "."
        module_prefix = module_prefix.replace("_orig_mod.", "")
        _replace_linear(module, compute_dtype, state_dict, module_prefix, patches, scale_weights, compile_args)

        if isinstance(module, nn.Linear) and "loras" not in module_prefix:
            in_features = state_dict[module_prefix + "weight"].shape[1]
            out_features = state_dict[module_prefix + "weight"].shape[0]
            if scale_weights is not None:
                scale_key = f"{module_prefix}scale_weight"

            with init_empty_weights():
                model._modules[name] = CustomLinear(
                    in_features,
                    out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                    scale_weight=scale_weights.get(scale_key) if scale_weights else None,
                    allow_compile=allow_compile
                )
            model._modules[name].source_cls = type(module)
            model._modules[name].requires_grad_(False)

    return model

def set_lora_params(module, patches, module_prefix="", device=torch.device("cpu")):
    remove_lora_from_module(module)
    # Recursively set lora_diffs and lora_strengths for all CustomLinear layers
    for name, child in module.named_children():
        params = list(child.parameters())
        if params:
            device = params[0].device
        else:
            device = torch.device("cpu")
        child_prefix = (f"{module_prefix}{name}.")
        set_lora_params(child, patches, child_prefix, device)
    if isinstance(module, CustomLinear):
        key = f"diffusion_model.{module_prefix}weight"
        patch = patches.get(key, [])
        #print(f"Processing LoRA patches for {key}: {len(patch)} patches found")
        if len(patch) == 0:
            key = key.replace("_orig_mod.", "")
            patch = patches.get(key, [])
            #print(f"Processing LoRA patches for {key}: {len(patch)} patches found")
        if len(patch) != 0:
            lora_diffs = []
            for p in patch:
                lora_obj = p[1]
                if "head" in key:
                    continue  # For now skip LoRA for head layers
                elif hasattr(lora_obj, "weights"):
                    lora_diffs.append(lora_obj.weights)
                elif isinstance(lora_obj, tuple) and lora_obj[0] == "diff":
                    lora_diffs.append(lora_obj[1])
                else:
                    continue
            lora_strengths = [p[0] for p in patch]
            module.set_lora_diffs(lora_diffs, device=device)
            module.lora_strengths = lora_strengths
            module.step = 0  # Initialize step for LoRA scheduling


class CustomLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        compute_dtype=None,
        device=None,
        scale_weight=None,
        allow_compile=False
    ) -> None:
        super().__init__(in_features, out_features, bias, device)
        self.compute_dtype = compute_dtype
        self.lora_diffs = []
        self.step = 0
        self.scale_weight = scale_weight
        self.lora_strengths = []
        self.allow_compile = allow_compile

        if not allow_compile:
            self._get_weight_with_lora = torch.compiler.disable()(self._get_weight_with_lora)
    
    def set_lora_diffs(self, lora_diffs, device=torch.device("cpu")):
        self.lora_diffs = []
        for i, diff in enumerate(lora_diffs):
            if len(diff) > 1:
                self.register_buffer(f"lora_diff_{i}_0", diff[0].to(device, self.compute_dtype))
                self.register_buffer(f"lora_diff_{i}_1", diff[1].to(device, self.compute_dtype))
                setattr(self, f"lora_diff_{i}_2", diff[2])
                self.lora_diffs.append((f"lora_diff_{i}_0", f"lora_diff_{i}_1", f"lora_diff_{i}_2"))
            else:
                self.register_buffer(f"lora_diff_{i}_0", diff[0].to(device, self.compute_dtype))
                self.lora_diffs.append(f"lora_diff_{i}_0")

    def _get_weight_with_lora(self, weight):
        """Apply LoRA outside compiled region"""
        if not hasattr(self, "lora_diff_0_0"):
            return weight
        
        for lora_diff_names, lora_strength in zip(self.lora_diffs, self.lora_strengths):
            if isinstance(lora_strength, list):
                lora_strength = lora_strength[self.step]
                if lora_strength == 0.0:
                    continue
            elif lora_strength == 0.0:
                continue
            if isinstance(lora_diff_names, tuple):
                lora_diff_0 = getattr(self, lora_diff_names[0])
                lora_diff_1 = getattr(self, lora_diff_names[1])
                lora_diff_2 = getattr(self, lora_diff_names[2])
                patch_diff = torch.mm(
                    lora_diff_0.flatten(start_dim=1),
                    lora_diff_1.flatten(start_dim=1)
                ).reshape(weight.shape) + 0
                alpha = lora_diff_2 / lora_diff_1.shape[0] if lora_diff_2 is not None else 1.0
                scale = lora_strength * alpha
                weight = weight.add(patch_diff, alpha=scale)
            else:
                lora_diff = getattr(self, lora_diff_names)
                weight = weight.add(lora_diff, alpha=lora_strength)
        return weight

    def forward(self, input):
        if self.bias is not None:
            bias = self.bias.to(input)
        else:
            bias = None
        weight = self.weight.to(input)

        if self.scale_weight is not None:
            if weight.numel() < input.numel():
                weight = weight * self.scale_weight
            else:
                input = input * self.scale_weight

        weight = self._get_weight_with_lora(weight)

        return torch.nn.functional.linear(input, weight, bias)
    
def remove_lora_from_module(module):
    for name, submodule in module.named_modules():
        if hasattr(submodule, "lora_diffs"):
            for i in range(len(submodule.lora_diffs)):
                if hasattr(submodule, f"lora_diff_{i}_0"):
                    delattr(submodule, f"lora_diff_{i}_0")
                if hasattr(submodule, f"lora_diff_{i}_1"):
                    delattr(submodule, f"lora_diff_{i}_1")
                if hasattr(submodule, f"lora_diff_{i}_2"):
                    delattr(submodule, f"lora_diff_{i}_2")
