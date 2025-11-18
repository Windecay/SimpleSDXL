import torch
import torch.nn as nn
import numpy as np
import gguf
from accelerate import init_empty_weights

from .gguf_utils import GGUFParameter, dequantize_gguf_tensor
from ..utils import log

def load_gguf(model_path):
    from gguf import GGUFReader
    reader = GGUFReader(model_path)
    parsed_parameters = {}
    for tensor in reader.tensors:
        # if the tensor is a torch supported dtype do not use GGUFParameter
        is_gguf_quant = tensor.tensor_type not in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]
        meta_tensor = torch.empty(tensor.data.shape, dtype=torch.from_numpy(np.empty(0, dtype=tensor.data.dtype)).dtype, device='meta')
        parsed_parameters[tensor.name] = GGUFParameter(meta_tensor, quant_type=tensor.tensor_type) if is_gguf_quant else meta_tensor
    return parsed_parameters, reader

#based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/quantizers/gguf/utils.py
def _replace_with_gguf_linear(model, compute_dtype, state_dict, prefix="", modules_to_not_convert=[], patches=None, compile_args=None):
    def _should_convert_to_gguf(state_dict, prefix):
        weight_key = prefix + "weight"
        return weight_key in state_dict and isinstance(state_dict[weight_key], GGUFParameter)

    has_children = list(model.children())
    if not has_children:
        return
    
    allow_compile = False

    for name, module in model.named_children():
        if compile_args is not None:
            allow_compile = compile_args.get("allow_unmerged_lora_compile", False)
        module_prefix = prefix + name + "."
        _replace_with_gguf_linear(module, compute_dtype, state_dict, module_prefix, modules_to_not_convert, patches, compile_args)

        if (
            isinstance(module, nn.Linear)
            and not isinstance(module, GGUFLinear) 
            and _should_convert_to_gguf(state_dict, module_prefix)
            and name not in modules_to_not_convert
        ):
            in_features = state_dict[module_prefix + "weight"].shape[1]
            out_features = state_dict[module_prefix + "weight"].shape[0]

            with init_empty_weights():
                model._modules[name] = GGUFLinear(
                    in_features,
                    out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                    allow_compile=allow_compile
                )
            
            model._modules[name].source_cls = type(module)
            model._modules[name].requires_grad_(False)
    return model

def set_lora_params_gguf(module, patches, module_prefix="", device=torch.device("cpu")):
    # Recursively set lora_diffs and lora_strengths for all GGUFLinear layers
    for name, child in module.named_children():
        params = list(child.parameters())
        if params:
            device = params[0].device
        else:
            device = torch.device("cpu")
        child_prefix = (f"{module_prefix}{name}.")
        set_lora_params_gguf(child, patches, child_prefix, device)
    if isinstance(module, GGUFLinear):
        key = f"diffusion_model.{module_prefix}weight"
        patch = patches.get(key, [])
        #print(f"Processing LoRA patches for {key}: {len(patch)} patches found")
        if len(patch) == 0:
            key = key.replace("_orig_mod.", "")
            patch = patches.get(key, [])
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
            module.lora_strengths = [p[0] for p in patch]
            module.set_lora_diffs(lora_diffs, device=device)
            module.step = 0  # Initialize step for LoRA scheduling


class GGUFLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        compute_dtype=None,
        device=None,
        allow_compile=False
    ) -> None:
        super().__init__(in_features, out_features, bias, device)
        self.compute_dtype = compute_dtype
        self.lora_diffs = []
        self.lora_strengths = []
        self.step = 0
        self.allow_compile = allow_compile

        if not allow_compile:
            self._get_weight_with_lora = torch.compiler.disable()(self._get_weight_with_lora)

    def forward(self, inputs):
        weight = dequantize_gguf_tensor(self.weight).to(self.compute_dtype)
        bias = self.bias.to(self.compute_dtype) if self.bias is not None else None

        weight = self._get_weight_with_lora(weight)#.to(self.compute_dtype)

        return torch.nn.functional.linear(inputs, weight, bias)
    
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