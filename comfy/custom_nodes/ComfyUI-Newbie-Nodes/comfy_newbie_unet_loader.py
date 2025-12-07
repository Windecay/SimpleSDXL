"""
NewBie UNet Loader Node for ComfyUI
专门用于加载NewBie扩散模型
"""

import torch
import torch.nn as nn
from typing import Tuple, Any, Dict, Optional
import os
import json
from safetensors.torch import load_file

try:
    # 导入ComfyUI相关模块
    import comfy.model_management as model_management
    import comfy.model_base
    import comfy.latent_formats
    import comfy.supported_models
    import comfy.utils
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    print("Warning: ComfyUI modules not available")

try:
    import sys
    
    # 临时禁用flash_attn导入，避免编译问题
    class MockFlashAttn:
        def __getattr__(self, name):
            def mock_func(*args, **kwargs):
                raise NotImplementedError("flash_attn not available")
            return mock_func
    
    # 如果flash_attn不可用，使用mock
    try:
        import flash_attn
    except ImportError:
        sys.modules['flash_attn'] = MockFlashAttn()
        sys.modules['flash_attn.bert_padding'] = MockFlashAttn()
    
    from .models.model import NextDiT_CLIP, NextDiT_2B_GQA_patch2_Adaln_Refiner_WHIT_CLIP, NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP
    NEWBIE_AVAILABLE = True
except ImportError:
    try:
        from models.model import NextDiT_CLIP, NextDiT_2B_GQA_patch2_Adaln_Refiner_WHIT_CLIP, NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP
        NEWBIE_AVAILABLE = True
    except ImportError:
        NEWBIE_AVAILABLE = False
        print("Warning: NewBie models not available")


class NewBieModelConfig:
    """模拟ComfyUI模型配置，基于Lumina Image2"""
    def __init__(self, model):
        # 基础UNet配置
        self.model_channels = getattr(model, 'dim', 2304)
        self.in_channels = getattr(model, 'in_channels', 16) 
        self.out_channels = getattr(model, 'in_channels', 16)  # NextDiT中out_channels = in_channels
        self.num_res_blocks = 2
        self.attention_resolutions = []
        self.channel_mult = [1, 2, 4]
        self.num_head_channels = getattr(model, 'n_heads', 24)
        self.transformer_depth = [1, 1, 1]
        self.context_dim = getattr(model, 'cap_feat_dim', 2560)
        self.use_linear_in_transformer = True
        
        # 采样配置
        self.beta_schedule = "linear"
        self.timesteps = 1000
        self.linear_start = 0.00085
        self.linear_end = 0.012
        
        # 其他配置
        self.parameterization = "eps"
        self.use_ema = False
        
        # 添加latent_format属性 - 使用FLUX格式因为NextDiT使用16通道和FLUX VAE
        if COMFY_AVAILABLE:
            try:
                import comfy.latent_formats
                # NextDiT使用16通道和FLUX VAE，所以使用FLUX latent format
                self.latent_format = comfy.latent_formats.Flux()
            except:
                # 如果无法导入FLUX格式，创建一个兼容的默认格式
                self.latent_format = self._create_default_latent_format()
        else:
            self.latent_format = self._create_default_latent_format()
        
        # 关键：添加ComfyUI期望的sampling_settings属性（基于Lumina）
        self.sampling_settings = {
            "shift": 6,  # Lumina默认shift值
            "sigma_min": 0.0292,
            "sigma_max": 14.6146,
            "sigma_data": 1.0,
            "beta_schedule": self.beta_schedule,
            "timesteps": self.timesteps,
            "linear_start": self.linear_start,
            "linear_end": self.linear_end,
            "parameterization": self.parameterization,
        }
    
    def _create_default_latent_format(self):
        """创建默认的latent format以防ComfyUI不可用"""
        class DefaultLatentFormat:
            def __init__(self):
                self.latent_rgb_factors = None
                self.latent_channels = 16  # NextDiT使用16通道
                self.scale_factor = 0.3611  # FLUX scale factor
                self.shift_factor = 0.1159  # FLUX shift factor
            
            def process_in(self, latent):
                return (latent - self.shift_factor) * self.scale_factor
            
            def process_out(self, latent):
                return (latent / self.scale_factor) + self.shift_factor
        
        return DefaultLatentFormat()


class NewBieModelPatcher:
    """NewBie模型包装器，兼容ComfyUI的ModelPatcher接口"""
    
    def __init__(self, model, load_device, offload_device, size=0, current_device=None, weight_inplace_update=False):
        self.model = model
        self.size = size
        self.current_device = current_device or load_device
        self.load_device = load_device
        self.offload_device = offload_device
        self.weight_inplace_update = weight_inplace_update
        self.model_options = {
            "transformer_options": {}  # ComfyUI 期望的transformer选项
        }
        self.patches = {}
        
        # 添加ComfyUI期望的object patches系统
        self.object_patches = {}
        self.object_patches_backup = {}
        
        # 添加模型状态管理
        self.is_injected = False
        self.callbacks = {}
        
        # 添加ComfyUI期望的hook_mode属性
        self.hook_mode = "none"  # 默认hook模式
        
        # 添加ComfyUI期望的wrapper系统
        self.wrappers = {}
        
        # 添加parent属性（用于模型管理）
        self.parent = None
        
        # 添加hook patches系统
        self.hook_patches = {}
        self.hook_patches_backup = None
        
        # 添加ComfyUI期望的model_config属性
        self.model_config = NewBieModelConfig(model)
        
        # 确保模型有适当的条件处理方法
        self._add_conditioning_methods()
        
    def clone(self):
        """克隆模型"""
        n = NewBieModelPatcher(self.model, self.load_device, self.offload_device, self.size, self.current_device, self.weight_inplace_update)
        n.patches = self.patches.copy()
        n.model_options = self.model_options.copy()
        n.object_patches = self.object_patches.copy()
        n.object_patches_backup = self.object_patches_backup.copy()
        n.is_injected = self.is_injected
        n.callbacks = self.callbacks.copy()
        n.hook_mode = self.hook_mode
        n.wrappers = self.wrappers.copy()
        n.parent = self.parent
        n.hook_patches = self.hook_patches.copy()
        n.hook_patches_backup = self.hook_patches_backup
        return n
    
    def is_clone(self, other):
        """检查是否为克隆"""
        if hasattr(other, 'model') and self.model is other.model:
            return True
        return False
        
    def memory_required(self, input_shape):
        """估算内存需求"""
        return self.size
        
    def set_model_sampler_cfg_function(self, sampler_cfg_function, disable_cfg1_optimization=False):
        """设置采样器配置函数"""
        if hasattr(self.model, 'set_model_sampler_cfg_function'):
            self.model.set_model_sampler_cfg_function(sampler_cfg_function, disable_cfg1_optimization)
    
    def set_model_unet_function_wrapper(self, unet_wrapper_function):
        """设置UNet包装函数"""
        if hasattr(self.model, 'set_model_unet_function_wrapper'):
            self.model.set_model_unet_function_wrapper(unet_wrapper_function)
    
    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        """添加权重patches"""
        for key, patch in patches.items():
            if key not in self.patches:
                self.patches[key] = []
            self.patches[key].append((strength_patch, patch, strength_model))
    
    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        """加载模型并应用patches"""
        target_device = device_to or self.load_device
        self.partially_load(target_device, lowvram_model_memory)
        if force_patch_weights or full_load:
            self.patch_model(target_device, lowvram_model_memory, load_weights=True, force_patch_weights=force_patch_weights)
        return self.model
    
    def inject_model(self):
        """注入模型到当前上下文"""
        self.is_injected = True
    
    def eject_model(self):
        """从当前上下文弹出模型"""
        self.is_injected = False
    
    def detach(self, unpatch_all=True):
        """分离并清理模型"""
        if unpatch_all:
            self.unpatch_model()
        self.eject_model()
    
    def set_model_compute_dtype(self, dtype):
        """设置模型计算数据类型"""
        if hasattr(self.model, 'to'):
            self.model = self.model.to(dtype=dtype)
        return self.model
    
    def restore_hook_patches(self):
        """恢复hook patches"""
        if self.hook_patches_backup is not None:
            self.hook_patches = self.hook_patches_backup
            self.hook_patches_backup = None
    
    def get_wrappers(self):
        """获取wrappers"""
        return self.wrappers
    
    def set_wrappers(self, wrappers):
        """设置wrappers"""
        self.wrappers = wrappers
    
    def register_all_hook_patches(self, hooks, target_dict, model_options=None, registered=None):
        """注册所有hook patches"""
        if registered is None:
            registered = set()
        
        # 备份当前的hook patches
        if self.hook_patches_backup is None:
            self.hook_patches_backup = self.hook_patches.copy()
        
        # 注册新的hook patches
        for hook_name, hook_patches in hooks.items():
            if hook_name not in registered:
                if hook_name not in self.hook_patches:
                    self.hook_patches[hook_name] = []
                self.hook_patches[hook_name].extend(hook_patches)
                registered.add(hook_name)
        
        return registered
    
    def model_dtype(self):
        """返回模型的数据类型"""
        # 获取模型的第一个参数的dtype
        for param in self.model.parameters():
            return param.dtype
        return torch.float32  # 默认值
    
    def get_nested_additional_models(self):
        """获取嵌套的额外模型"""
        return []
    
    def model_size(self):
        """返回模型大小"""
        return self.size
    
    def get_additional_models(self):
        """获取额外模型"""
        return []
    
    def current_loaded_device(self):
        """返回当前加载的设备"""
        return self.current_device
    
    def loaded_size(self):
        """返回已加载的模型大小"""
        return self.size
    
    def pre_run(self):
        """采样前的预处理"""
        # 应用所有patches
        self.patch_model()
        # 确保模型在正确设备上
        if hasattr(self.model, 'to'):
            self.model.to(self.current_device)
    
    def cleanup(self):
        """采样后的清理"""
        # 恢复patches
        self.unpatch_model()
        # 恢复hook patches
        self.restore_hook_patches()
    
    def model_patches_to(self, device):
        """将模型patches移动到设备"""
        # 将模型patches移动到指定设备
        for patch_key in self.patches:
            if hasattr(self.patches[patch_key], 'to'):
                self.patches[patch_key] = self.patches[patch_key].to(device)
    
    def get_model_object(self, name: str):
        """获取模型对象，支持object patches"""
        if name in self.object_patches:
            return self.object_patches[name]
        elif name in self.object_patches_backup:
            return self.object_patches_backup[name]
        else:
            # 如果comfy.utils可用，使用它；否则使用简单的getattr
            if COMFY_AVAILABLE:
                try:
                    import comfy.utils
                    return comfy.utils.get_attr(self.model, name)
                except:
                    pass
            
            # 简单的点符号属性访问实现
            attrs = name.split('.')
            obj = self.model
            for attr in attrs:
                obj = getattr(obj, attr)
            return obj
    
    def unpatch_model(self, device_to=None, unpatch_weights=True):
        """取消patch，恢复原始值"""
        # 恢复object patches
        if COMFY_AVAILABLE:
            try:
                import comfy.utils
                keys = list(self.object_patches_backup.keys())
                for k in keys:
                    comfy.utils.set_attr(self.model, k, self.object_patches_backup[k])
                self.object_patches_backup.clear()
            except:
                pass
        else:
            # 简单实现：只支持单级属性
            keys = list(self.object_patches_backup.keys())
            for k in keys:
                if '.' not in k and hasattr(self.model, k):
                    setattr(self.model, k, self.object_patches_backup[k])
            self.object_patches_backup.clear()
    
    def add_object_patch(self, name, obj):
        """添加对象patch"""
        self.object_patches[name] = obj
    
    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        """应用patches"""
        # 应用object patches
        for k in self.object_patches:
            if COMFY_AVAILABLE:
                try:
                    import comfy.utils
                    old = comfy.utils.set_attr(self.model, k, self.object_patches[k])
                    if k not in self.object_patches_backup:
                        self.object_patches_backup[k] = old
                except:
                    pass
            else:
                # 简单实现：只支持单级属性
                if '.' not in k and hasattr(self.model, k):
                    if k not in self.object_patches_backup:
                        self.object_patches_backup[k] = getattr(self.model, k)
                    setattr(self.model, k, self.object_patches[k])
    
    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        """部分加载模型"""
        if self.current_device != device_to:
            self.model.to(device_to)
            self.current_device = device_to
    
    def partially_unload(self, device_to, extra_memory=0):
        """部分卸载模型"""
        if device_to != self.current_device:
            self.model.to(device_to)
            self.current_device = device_to
    
    def _add_conditioning_methods(self):
        """添加条件处理方法以兼容ComfyUI"""
        import types
        
        def cond_stage_model_encode(text):
            """条件编码方法"""
            if isinstance(text, torch.Tensor):
                return {"model_conds": text}
            return {"model_conds": text}
        
        def encode_from_tokens(tokens, return_pooled=False):
            """从tokens编码"""
            # 简单实现：返回tokens本身
            if return_pooled:
                return tokens, None  # 返回tokens和空的pooled输出
            return tokens
        
        # 将方法绑定到模型
        if not hasattr(self.model, 'cond_stage_model_encode'):
            self.model.cond_stage_model_encode = cond_stage_model_encode
        if not hasattr(self.model, 'encode_from_tokens'):
            self.model.encode_from_tokens = encode_from_tokens
    
    def __call__(self, *args, **kwargs):
        """使模型可调用"""
        return self.model(*args, **kwargs)


class NewBieUNetLoader:
    """NewBie UNet模型加载器"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_path": ("STRING", {
                    "default": "",
                    "description": "Path to NewBie UNet safetensors file"
                }),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto"
                }),
                "dtype": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders"
    TITLE = "NewBie UNet Loader"

    def load_model_config(self, model_dir: str) -> Dict[str, Any]:
        """加载模型配置"""
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 默认配置
        return {
            "architecture": "NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP",
            "cap_feat_dim": 2560,
            "clip_text_dim": 1024,
            "clip_img_dim": 1024,
            "qk_norm": True,
        }

    def load_training_args(self, model_dir: str) -> Any:
        """加载训练参数"""
        args_path = os.path.join(model_dir, "model_args.pth")
        if os.path.exists(args_path):
            return torch.load(args_path, weights_only=False, map_location="cpu")
        
        # 创建默认参数
        return type('Args', (), {
            'model': 'NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP',
            'qk_norm': True,
            'cap_feat_dim': 2560,
        })()

    def create_newbie_model(self, config: Dict[str, Any], train_args: Any) -> nn.Module:
        """创建NewBie模型实例"""
        if not NEWBIE_AVAILABLE:
            raise ImportError("NewBie models not available")
        
        model_name = config.get("architecture", train_args.model)
        
        # 模型映射
        model_map = {
            "NextDiT_2B_GQA_patch2_Adaln_Refiner_WHIT_CLIP": NextDiT_2B_GQA_patch2_Adaln_Refiner_WHIT_CLIP,
            "NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP": NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP,
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model_cls = model_map[model_name]
        
        # 构建模型参数
        model_kwargs = {
            'in_channels': config.get('in_channels', 16),
            'qk_norm': config.get('qk_norm', True),
            'cap_feat_dim': config.get('cap_feat_dim', 2560),
            'clip_text_dim': config.get('clip_text_dim', 1024),
            'clip_img_dim': config.get('clip_img_dim', 1024),
        }
        
        print(f"Creating {model_name} with cap_feat_dim={model_kwargs['cap_feat_dim']}")
        
        return model_cls(**model_kwargs)

    def load_unet(
        self,
        unet_path: str,
        device: str = "auto",
        dtype: str = "auto"
    ) -> Tuple[Any,]:
        """
        加载NewBie UNet模型
        
        Returns:
            Tuple containing NewBie model wrapper
        """
        
        if not os.path.exists(unet_path):
            raise FileNotFoundError(f"UNet file not found: {unet_path}")
        
        print(f"Loading NewBie UNet from: {unet_path}")
        
        # 确定模型目录
        if os.path.isfile(unet_path):
            model_dir = os.path.dirname(unet_path)
            safetensors_path = unet_path
        else:
            model_dir = unet_path
            safetensors_path = os.path.join(model_dir, "newbie_dit.safetensors")
            if not os.path.exists(safetensors_path):
                safetensors_path = os.path.join(model_dir, "model.safetensors")
        
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(f"Safetensors file not found: {safetensors_path}")
        
        # 设置设备和数据类型
        if device == "auto":
            load_device = model_management.get_torch_device()
            offload_device = model_management.unet_offload_device()
        else:
            load_device = torch.device(device)
            offload_device = torch.device("cpu")
        
        if dtype == "auto":
            model_dtype = model_management.unet_dtype()
        else:
            dtype_map = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32
            }
            model_dtype = dtype_map[dtype]
        
        # 加载配置和训练参数
        config = self.load_model_config(model_dir)
        train_args = self.load_training_args(model_dir)
        
        # 创建模型
        model = self.create_newbie_model(config, train_args)
        model = model.to(dtype=model_dtype)
        
        # 为模型添加ComfyUI期望的model_config属性
        model.model_config = NewBieModelConfig(model)
        
        # 添加latent_format属性到模型本身
        model.latent_format = model.model_config.latent_format
        
        # 添加ComfyUI期望的extra_conds_shapes方法
        def extra_conds_shapes(self, **kwargs):
            """返回额外条件的形状信息"""
            return {}
        
        # 添加ComfyUI期望的memory_required方法
        def memory_required(self, input_shape, cond_shapes=None):
            """估算内存需求"""
            # 简单估算：模型大小 + 输入数据大小
            area = input_shape[0] * input_shape[2] * input_shape[3]
            return model.model_config.model_channels * area * 4  # 4 bytes per float32
        
        # 将方法绑定到模型
        import types
        model.extra_conds_shapes = types.MethodType(extra_conds_shapes, model)
        model.memory_required = types.MethodType(memory_required, model)
        
        
        # 加载权重
        print("Loading weights...")
        state_dict = load_file(safetensors_path, device="cpu")
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"⚠️ Missing keys: {len(missing_keys)}")
            for key in missing_keys[:5]:  # 只显示前5个
                print(f"  - {key}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
            for key in unexpected_keys[:5]:  # 只显示前5个
                print(f"  - {key}")
        
        # 移动到设备
        model = model.to(load_device)
        model.eval()
        
        print(f"✓ NewBie UNet loaded successfully")
        print(f"  - Device: {load_device}")
        print(f"  - Dtype: {model_dtype}")
        print(f"  - Cap feat dim: {config.get('cap_feat_dim', 'unknown')}")
        
        # 计算模型大小
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        
        # 创建ComfyUI兼容的模型包装器
        if COMFY_AVAILABLE:
            model_patcher = NewBieModelPatcher(
                model, 
                load_device, 
                offload_device, 
                size=int(model_size),
                current_device=load_device
            )
        else:
            model_patcher = model
        
        return (model_patcher,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "NewBieUNetLoader": NewBieUNetLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NewBieUNetLoader": "NewBie UNet Loader",
}