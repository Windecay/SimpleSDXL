#!/usr/bin/env python3
"""
固定版NewBie模型转换脚本 - 自动从checkpoint推断正确参数
"""

import argparse
import os
import json
import torch
from safetensors.torch import save_file
from pathlib import Path
import shutil
from typing import Dict, Any, Optional

# 导入NewBie模型
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
    
    from models.model import NextDiT_CLIP, NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP
except ImportError as e:
    print(f"无法导入模型: {e}")
    print("请确保在NewBie项目根目录运行此脚本")
    exit(1)


def infer_model_params_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """从checkpoint权重直接推断模型参数"""
    print(f"分析checkpoint权重: {checkpoint_path}")
    
    # 加载checkpoint
    if checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path, device='cpu')
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    
    # 从权重形状推断参数
    params = {}
    
    # 1. 从 x_embedder.weight 推断 patch_size 和 in_channels
    if 'x_embedder.weight' in state_dict:
        # x_embedder: [dim, patch_size^2 * in_channels]
        dim, patch_input_dim = state_dict['x_embedder.weight'].shape
        params['dim'] = dim
        
        # 假设patch_size=2, 计算in_channels
        patch_size = 2
        in_channels = patch_input_dim // (patch_size * patch_size)
        params['patch_size'] = patch_size
        params['in_channels'] = in_channels
        params['out_channels'] = in_channels
        
        print(f"推断: dim={dim}, patch_size={patch_size}, in_channels={in_channels}")
    
    # 2. 从 cap_embedder 推断 cap_feat_dim
    if 'cap_embedder.0.weight' in state_dict:
        cap_feat_dim = state_dict['cap_embedder.0.weight'].shape[0]
        params['cap_feat_dim'] = cap_feat_dim
        print(f"推断: cap_feat_dim={cap_feat_dim}")
    
    # 3. 从 final_layer.linear 验证输出维度
    if 'final_layer.linear.weight' in state_dict:
        output_dim, _ = state_dict['final_layer.linear.weight'].shape
        expected_output_dim = params['patch_size'] ** 2 * params['out_channels']
        print(f"验证: final_layer输出维度={output_dim}, 期望={expected_output_dim}")
        
        if output_dim != expected_output_dim:
            print(f"⚠️ 输出维度不匹配，重新推断...")
            # 重新计算patch_size或out_channels
            if output_dim == 64:  # 常见情况
                params['patch_size'] = 2
                params['out_channels'] = 16
                params['in_channels'] = 16
            elif output_dim == 16:
                params['patch_size'] = 2
                params['out_channels'] = 4
                params['in_channels'] = 4
    
    # 4. 推断其他架构参数 (基于权重数量和命名)
    # 计算transformer层数
    layer_count = 0
    for key in state_dict.keys():
        if key.startswith('layers.') and '.attention.qkv.weight' in key:
            layer_num = int(key.split('.')[1])
            layer_count = max(layer_count, layer_num + 1)
    
    if layer_count > 0:
        params['n_layers'] = layer_count
        print(f"推断: n_layers={layer_count}")
    
    # 从attention权重推断头数
    if 'layers.0.attention.qkv.weight' in state_dict:
        qkv_weight = state_dict['layers.0.attention.qkv.weight']
        total_dim, model_dim = qkv_weight.shape
        # total_dim = (n_heads + 2*n_kv_heads) * head_dim
        # 假设使用GQA，n_kv_heads=8
        n_kv_heads = 8
        head_dim = model_dim // 24  # 假设n_heads=24
        params['n_heads'] = 24
        params['n_kv_heads'] = n_kv_heads
        print(f"推断: n_heads=24, n_kv_heads={n_kv_heads}")
    
    # 设置默认的轴维度参数
    params.update({
        'axes_dims': [32, 32, 32],
        'axes_lens': [1024, 512, 512],
        'qk_norm': True,
        'clip_text_dim': 1024,
        'clip_img_dim': 1024,
    })
    
    return params


def convert_model_with_inferred_params(
    checkpoint_path: str,
    output_dir: str,
    target_dtype: str = "bf16",
    architecture: str = "NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP"
):
    """使用推断的参数转换模型"""
    
    # 设置数据类型映射
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }
    target_torch_dtype = dtype_map[target_dtype]
    
    print(f"开始转换NewBie模型...")
    print(f"输入: {checkpoint_path}")
    print(f"输出: {output_dir}")
    print(f"目标精度: {target_dtype}")
    print(f"架构: {architecture}")
    
    # 从checkpoint推断参数
    model_params = infer_model_params_from_checkpoint(checkpoint_path)
    
    # 选择模型类
    if architecture == "NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP":
        model_cls = NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP
    elif architecture == "NextDiT_2B_GQA_patch2_Adaln_Refiner_WHIT_CLIP":
        model_cls = NextDiT_2B_GQA_patch2_Adaln_Refiner_WHIT_CLIP
    else:
        raise ValueError(f"不支持的架构: {architecture}")
    
    # 创建模型
    print("创建模型实例...")
    model = model_cls(
        in_channels=model_params['in_channels'],
        cap_feat_dim=model_params['cap_feat_dim'],
        qk_norm=model_params['qk_norm'],
        clip_text_dim=model_params.get('clip_text_dim', 1024),
        clip_img_dim=model_params.get('clip_img_dim', 1024),
    )
    
    # 加载checkpoint
    print(f"加载checkpoint...")
    if checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path, device='cpu')
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    
    # 验证权重可以完美加载
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if not missing and not unexpected:
            print("✅ 权重完美匹配！")
        else:
            if missing:
                print(f"⚠️ 缺失的keys: {missing}")
            if unexpected:
                print(f"⚠️ 多余的keys: {unexpected}")
            
            # 如果有问题，尝试strict=True看详细错误
            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception as e:
                print(f"❌ 严格模式错误: {e}")
                raise
                
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        raise
    
    # 转换数据类型
    if target_torch_dtype != torch.float32:
        print(f"转换数据类型到 {target_dtype}...")
        converted_state_dict = {}
        for key, tensor in state_dict.items():
            if tensor.dtype == torch.float32:
                converted_state_dict[key] = tensor.to(target_torch_dtype)
            else:
                converted_state_dict[key] = tensor
        state_dict = converted_state_dict
    
    # 创建配置
    config = {
        "model_type": "newbie_dit",
        "architecture": architecture,
        "patch_size": model.patch_size,
        "in_channels": model.in_channels,
        "out_channels": model.out_channels,
        "dim": model.dim,
        "n_heads": model.n_heads,
        "axes_dims": model.axes_dims,
        "axes_lens": model.axes_lens,
        "enable_clip": getattr(model, 'enable_clip', True),
        "cap_feat_dim": model_params['cap_feat_dim'],
        "qk_norm": model_params['qk_norm'],
        "torch_dtype": target_dtype,
    }
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存safetensors权重文件
    model_name = "newbie_dit"
    safetensors_path = os.path.join(output_dir, f"{model_name}.safetensors")
    save_file(state_dict, safetensors_path)
    
    # 保存配置文件
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 创建模型索引文件
    model_index = {
        "metadata": {
            "total_size": sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
        },
        "weight_map": {key: f"{model_name}.safetensors" for key in state_dict.keys()}
    }
    
    index_path = os.path.join(output_dir, f"{model_name}.safetensors.index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(model_index, f, indent=2)
    
    # 打印统计信息
    total_params = sum(p.numel() for p in state_dict.values())
    total_size_mb = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values()) / (1024**2)
    
    print(f"\n✅ 转换完成!")
    print(f"总参数量: {total_params:,}")
    print(f"模型大小: {total_size_mb:.1f} MB")
    print(f"输出目录: {output_dir}")
    print(f"配置: {config}")


def main():
    parser = argparse.ArgumentParser(description="Convert NewBie model with auto-inferred parameters")
    
    parser.add_argument(
        "input_checkpoint",
        type=str,
        help="Input checkpoint file path"
    )
    
    parser.add_argument(
        "output_dir", 
        type=str,
        help="Output directory for converted model"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Target data type (default: bf16)"
    )
    
    parser.add_argument(
        "--architecture",
        type=str,
        default="NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP",
        choices=["NextDiT_2B_GQA_patch2_Adaln_Refiner_WHIT_CLIP", "NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP"],
        help="Model architecture"
    )
    
    args = parser.parse_args()
    
    try:
        convert_model_with_inferred_params(
            checkpoint_path=args.input_checkpoint,
            output_dir=args.output_dir,
            target_dtype=args.dtype,
            architecture=args.architecture
        )
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()