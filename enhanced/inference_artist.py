import csv
import json
from pathlib import Path
from typing import Dict, Optional
import os
import numpy as np
import itertools
import torch
import torch.nn.functional as F
import modules.config as C
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import create_model
from timm.models import register_model
from timm.models import build_model_with_cfg

from timm.models.vision_transformer import trunc_normal_
from timm.layers import SqueezeExcite

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.autograd import Function
import triton
import triton.language as tl
from torch.amp import custom_fwd, custom_bwd
import math
from modules.model_loader import load_file_from_url
def _grid(numel: int, bs: int) -> tuple:
    return (triton.cdiv(numel, bs),)

@triton.jit
def _idx(i, n: int, c: int, h: int, w: int):
    ni = i // (c * h * w)
    ci = (i // (h * w)) % c
    hi = (i // w) % h
    wi = i % w
    m = i < (n * c * h * w)
    return ni, ci, hi, wi, m

@triton.jit
def ska_fwd(
    x_ptr, w_ptr, o_ptr,
    n, ic, h, w, ks, pad, wc,
    BS: tl.constexpr,
    CT: tl.constexpr, AT: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
    val = tl.zeros((BS,), dtype=AT)

    for kh in range(ks):
        hin = hi - pad + kh
        hb = (hin >= 0) & (hin < h)
        for kw in range(ks):
            win = wi - pad + kw
            b = hb & (win >= 0) & (win < w)

            x_off = ((ni * ic + ci) * h + hin) * w + win
            w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi

            x_val = tl.load(x_ptr + x_off, mask=m & b, other=0.0).to(CT)
            w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
            val += tl.where(b & m, x_val * w_val, 0.0).to(AT)

    tl.store(o_ptr + offs, val.to(CT), mask=m)

@triton.jit
def ska_bwd_x(
    go_ptr, w_ptr, gi_ptr,
    n, ic, h, w, ks, pad, wc,
    BS: tl.constexpr,
    CT: tl.constexpr, AT: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
    val = tl.zeros((BS,), dtype=AT)

    for kh in range(ks):
        ho = hi + pad - kh
        hb = (ho >= 0) & (ho < h)
        for kw in range(ks):
            wo = wi + pad - kw
            b = hb & (wo >= 0) & (wo < w)

            go_off = ((ni * ic + ci) * h + ho) * w + wo
            w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + ho * w + wo

            go_val = tl.load(go_ptr + go_off, mask=m & b, other=0.0).to(CT)
            w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
            val += tl.where(b & m, go_val * w_val, 0.0).to(AT)

    tl.store(gi_ptr + offs, val.to(CT), mask=m)

@triton.jit
def ska_bwd_w(
    go_ptr, x_ptr, gw_ptr,
    n, wc, h, w, ic, ks, pad,
    BS: tl.constexpr,
    CT: tl.constexpr, AT: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, ci, hi, wi, m = _idx(offs, n, wc, h, w)

    for kh in range(ks):
        hin = hi - pad + kh
        hb = (hin >= 0) & (hin < h)
        for kw in range(ks):
            win = wi - pad + kw
            b = hb & (win >= 0) & (win < w)
            w_off = ((ni * wc + ci) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi

            val = tl.zeros((BS,), dtype=AT)
            steps = (ic - ci + wc - 1) // wc
            for s in range(tl.max(steps, axis=0)):
                cc = ci + s * wc
                cm = (cc < ic) & m & b

                x_off = ((ni * ic + cc) * h + hin) * w + win
                go_off = ((ni * ic + cc) * h + hi) * w + wi

                x_val = tl.load(x_ptr + x_off, mask=cm, other=0.0).to(CT)
                go_val = tl.load(go_ptr + go_off, mask=cm, other=0.0).to(CT)
                val += tl.where(cm, x_val * go_val, 0.0).to(AT)

            tl.store(gw_ptr + w_off, val.to(CT), mask=m)

class SkaFn(Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        ks = int(math.sqrt(w.shape[2]))
        pad = (ks - 1) // 2
        ctx.ks, ctx.pad = ks, pad
        n, ic, h, width = x.shape
        wc = w.shape[1]
        o = torch.empty(n, ic, h, width, device=x.device, dtype=x.dtype)
        numel = o.numel()

        x = x.contiguous()
        w = w.contiguous()

        grid = lambda meta: _grid(numel, meta["BS"])

        ct = tl.float16 if x.dtype == torch.float16 else (tl.float32 if x.dtype == torch.float32 else tl.float64)
        at = tl.float32 if x.dtype == torch.float16 else ct

        ska_fwd[grid](x, w, o, n, ic, h, width, ks, pad, wc, BS=1024, CT=ct, AT=at)

        ctx.save_for_backward(x, w)
        ctx.ct, ctx.at = ct, at
        return o

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, go: torch.Tensor) -> tuple:
        ks, pad = ctx.ks, ctx.pad
        x, w = ctx.saved_tensors
        n, ic, h, width = x.shape
        wc = w.shape[1]

        go = go.contiguous()
        gx = gw = None
        ct, at = ctx.ct, ctx.at

        if ctx.needs_input_grad[0]:
            gx = torch.empty_like(x)
            numel = gx.numel()
            ska_bwd_x[lambda meta: _grid(numel, meta["BS"])](go, w, gx, n, ic, h, width, ks, pad, wc, BS=1024, CT=ct, AT=at)

        if ctx.needs_input_grad[1]:
            gw = torch.empty_like(w)
            numel = gw.numel() // w.shape[2]
            ska_bwd_w[lambda meta: _grid(numel, meta["BS"])](go, x, gw, n, wc, h, width, ic, ks, pad, BS=1024, CT=ct, AT=at)

        return gx, gw, None, None

class SKA(torch.nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return SkaFn.apply(x, w) # type: ignore

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Conv2d_BN(dim, h, ks=1)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0))
        self.dw = Conv2d_BN(nh_kd, nh_kd, 3, 1, 1, groups=nh_kd)
        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, H, W).split([self.nh_kd, self.nh_kd, self.dh], dim=1)
        q = self.dw(q)
        q, k, v = q.view(B, self.num_heads, -1, N), k.view(B, self.num_heads, -1, N), v.view(B, self.num_heads, -1, N)
        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)
        x = self.proj(x)
        return x

class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
    
    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv

import torch.nn as nn

class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        
        self.sks = sks
        self.groups = groups
        self.dim = dim
        
    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w

class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x

class Block(torch.nn.Module):    
    def __init__(self,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 stage=-1, depth=-1):
        super().__init__()
            
        if depth % 2 == 0:
            self.mixer = RepVGGDW(ed)
            self.se = SqueezeExcite(ed, 0.25)
        else:
            self.se = torch.nn.Identity()
            if stage == 3:
                self.mixer = Residual(Attention(ed, kd, nh, ar, resolution=resolution))
            else:
                self.mixer = LSConv(ed)

        self.ffn = Residual(FFN(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn(self.se(self.mixer(x)))

class LSNet(torch.nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[64, 128, 192, 256],
                 key_dim=[16, 16, 16, 16],
                 depth=[1, 2, 3, 4],
                 num_heads=[4, 4, 4, 4],
                 distillation=False,
                 **kwargs):
        super().__init__()

        default_cfg = kwargs.pop('default_cfg', None)
        pretrained_cfg = kwargs.pop('pretrained_cfg', None)
        pretrained_cfg_overlay = kwargs.pop('pretrained_cfg_overlay', None)

        if default_cfg is not None:
            self.default_cfg = default_cfg
        if pretrained_cfg is not None:
            self.pretrained_cfg = pretrained_cfg
        if pretrained_cfg_overlay is not None:
            self.pretrained_cfg_overlay = pretrained_cfg_overlay

        if kwargs:
            self.extra_init_kwargs = kwargs

        resolution = img_size
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 4, 3, 2, 1), torch.nn.ReLU(),
                                Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1), torch.nn.ReLU(),
                                Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1)
                           )

        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = nn.Sequential()
        self.blocks2 = nn.Sequential()
        self.blocks3 = nn.Sequential()
        self.blocks4 = nn.Sequential()
        blocks = [self.blocks1, self.blocks2, self.blocks3, self.blocks4]
        
        for i, (ed, kd, dpth, nh, ar) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio)):
            for d in range(dpth):
                blocks[i].append(Block(ed, kd, nh, ar, resolution, stage=i, depth=d))
            
            if i != len(depth) - 1:
                blk = blocks[i+1]
                resolution_ = (resolution - 1) // 2 + 1
                blk.append(Conv2d_BN(embed_dim[i], embed_dim[i], ks=3, stride=2, pad=1, groups=embed_dim[i]))
                blk.append(Conv2d_BN(embed_dim[i], embed_dim[i+1], ks=1, stride=1, pad=0))
                resolution = resolution_

        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
            
        self.num_classes = num_classes
        self.num_features = embed_dim[-1]

    @torch.jit.ignore # type: ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x


class LSNetArtist(LSNet):
    def __init__(self, 
                 img_size=224,
                 patch_size=8,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[64, 128, 256, 384],
                 key_dim=[16, 16, 16, 16],
                 depth=[0, 2, 8, 10],
                 num_heads=[3, 3, 3, 4],
                 distillation=False,
                 feature_dim=None,  # 特征向量维度，默认为embed_dim[-1]
                 use_projection=True,  # 是否使用projection层
                 **kwargs):
        default_cfg = kwargs.pop('default_cfg', None)
        pretrained_cfg = kwargs.pop('pretrained_cfg', None)
        pretrained_cfg_overlay = kwargs.pop('pretrained_cfg_overlay', None)

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            key_dim=key_dim,
            depth=depth,
            num_heads=num_heads,
            distillation=distillation,
            default_cfg=default_cfg,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            **kwargs
        )
        
        self.feature_dim = feature_dim if feature_dim is not None else embed_dim[-1]
        self.use_projection = use_projection
        
        # 如果使用projection层，添加一个映射层来生成固定维度的特征
        if self.use_projection and self.feature_dim != embed_dim[-1]:
            self.projection = nn.Sequential(
                BN_Linear(embed_dim[-1], self.feature_dim),
                nn.ReLU(),
            )
        else:
            self.projection = nn.Identity()
        
        # 重新定义分类头（基于特征维度）
        if num_classes > 0:
            self.head = BN_Linear(self.feature_dim, num_classes)
            if distillation:
                self.head_dist = BN_Linear(self.feature_dim, num_classes)
    
    def forward_features(self, x):
        """
        提取特征，不经过分类头
        用于聚类或特征提取
        """
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.projection(x)
        return x
    
    def forward(self, x, return_features=False):
        """
        x: 输入图像
        return_features: 是否只返回特征向量（用于聚类）
                        False时返回分类logits（用于分类）
        
        如果return_features=True: 返回特征向量 (batch_size, feature_dim)
        如果return_features=False: 返回分类logits (batch_size, num_classes)
        """
        features = self.forward_features(x)
        
        if return_features:
            # 返回特征向量用于聚类
            return features
        
        # 返回分类结果
        if self.distillation:
            x = self.head(features), self.head_dist(features)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(features)
        
        return x

def _cfg_artist(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 
        'input_size': (3, 224, 224), 
        'pool_size': (4, 4),
        'crop_pct': .9, 
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 
        'std': (0.229, 0.224, 0.225),
        'first_conv': 'patch_embed.0.c', 
        'classifier': ('head.linear', 'head_dist.linear'),
        **kwargs
    }


default_cfgs_artist = dict(
    lsnet_t_artist = _cfg_artist(),
    lsnet_s_artist = _cfg_artist(),
    lsnet_b_artist = _cfg_artist(),
    lsnet_l_artist = _cfg_artist(),
    lsnet_xl_artist = _cfg_artist(),
)


def _create_lsnet_artist(variant, pretrained=False, **kwargs):
    cfg = default_cfgs_artist.get(variant, None)
    if cfg is not None:
        kwargs.setdefault('default_cfg', cfg)
        kwargs.setdefault('pretrained_cfg', cfg)
    model = build_model_with_cfg(
        LSNetArtist,
        variant,
        pretrained,
        **kwargs,
    )
    return model

@register_model
def lsnet_xl_artist(num_classes=1000, distillation=False, pretrained=False,
                    feature_dim=None, use_projection=True, **kwargs):
    model = _create_lsnet_artist(
        "lsnet_xl_artist",
        pretrained=pretrained,
        num_classes=num_classes, 
        distillation=distillation,
        img_size=224,
        patch_size=8,
        embed_dim=[192, 384, 576, 768],
        depth=[8, 12, 16, 20],
        num_heads=[6, 6, 6, 6],
        feature_dim=feature_dim,
        use_projection=use_projection,
        **kwargs
    )
    return model

def load_checkpoint_state(checkpoint_path: str):
    """加载 checkpoint 并返回模型权重"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            return checkpoint['model']
        if 'model_ema' in checkpoint:
            return checkpoint['model_ema']
    return checkpoint


def normalize_state_dict_keys(state_dict):
    """移除分布式训练前缀等冗余标记"""
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]
        else:
            new_key = key
        normalized[new_key] = value
    return normalized


def resolve_num_classes(num_classes_arg: Optional[int],
                        class_mapping: Optional[Dict[int, str]],
                        state_dict) -> int:
    """根据参数、CSV 或 checkpoint 推断类别数"""
    # 优先使用CSV中的类别数
    if class_mapping:
        csv_classes = len(class_mapping)
        if num_classes_arg is not None and num_classes_arg != csv_classes:
            print(f"[Warning] 提供的 num_classes={num_classes_arg} 与 CSV 中的类别数 {csv_classes} 不一致，已使用 CSV 的值。")
        return csv_classes

    # 如果没有CSV，使用参数
    if num_classes_arg is not None:
        return num_classes_arg

    # 最后尝试从权重中解析分类头大小
    for key, value in state_dict.items():
        if key.endswith('head.weight') or key.endswith('head.l.weight'):
            return value.shape[0]

    raise ValueError('无法推断 num_classes，请提供 CSV 映射文件或显式指定 num_classes 参数。')


def resolve_feature_dim(feature_dim_arg: Optional[int], state_dict) -> int:
    """根据参数或 checkpoint 推断特征维度"""
    if feature_dim_arg is not None:
        return feature_dim_arg

    # 尝试从权重中解析特征维度
    # 查找head.bn.weight的维度，这通常是特征维度
    for key, value in state_dict.items():
        if key.endswith('head.bn.weight'):
            return value.shape[0]

    # 如果找不到，尝试查找其他可能的特征维度指示器
    for key, value in state_dict.items():
        if 'head' in key and 'weight' in key and len(value.shape) >= 2:
            # 对于线性层，输入维度通常是特征维度
            return value.shape[1] if len(value.shape) > 1 else value.shape[0]

    # 默认值
    print("[Warning] 无法从checkpoint推断特征维度，使用默认值384")
    return 384


def load_model(args, state_dict):
    """加载模型"""
    print(f"Loading model: {args.model}")
    state_dict = normalize_state_dict_keys(state_dict)

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
    )

    model_state = model.state_dict()
    adapted_state = {}
    mismatched = {}

    for key, value in state_dict.items():
        if key in model_state:
            if model_state[key].shape != value.shape:
                mismatched[key] = (model_state[key].shape, value.shape)
                continue
        adapted_state[key] = value

    classifier_keys = [key for key in mismatched if 'head' in key or 'classifier' in key]
    other_mismatched = {key: shapes for key, shapes in mismatched.items() if key not in classifier_keys}

    if other_mismatched:
        details = '\n'.join([
            f"  - {key}: checkpoint {found} -> model {expected}"
            for key, (expected, found) in other_mismatched.items()
        ])
        raise RuntimeError(
            "以下权重尺寸与当前模型不兼容，且无法自动处理，请检查 checkpoint 或模型配置:\n" + details
        )

    require_strict_head = (args.mode in ['classify', 'both']) and not args.allow_head_reinit

    if classifier_keys and require_strict_head:
        details = '\n'.join([
            f"  - {key}: checkpoint {mismatched[key][1]} -> model {mismatched[key][0]}"
            for key in classifier_keys
        ])
        raise RuntimeError(
            "分类模式下检测到 checkpoint 分类头与当前 num_classes 不一致，已终止加载以避免随机初始化结果。\n"
            "请使用与训练数据一致的 checkpoint，或在确认需要重新初始化分类头时添加 --allow-head-reinit，"
            "或者切换到 --mode cluster 仅提取特征。\n" + details
        )

    if classifier_keys:
        print("[Warning] 分类头权重尺寸与当前 num_classes 不一致，将重新初始化以下权重：")
        for key in classifier_keys:
            expected, found = mismatched[key]
            print(f"  - {key}: checkpoint {found} -> model {expected}")
        # 冲突键已在上文过滤掉，无需额外处理

    load_result = model.load_state_dict(adapted_state, strict=False)

    if load_result.missing_keys:
        print(f"[Info] Missing keys during load: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"[Info] Unexpected keys ignored: {load_result.unexpected_keys}")

    if classifier_keys and args.mode in ['classify', 'both']:
        if args.allow_head_reinit:
            print("[Warning] 分类模式在 --allow-head-reinit 下运行，分类头为随机初始化；为获得可靠结果请提供匹配的数据集 checkpoint。")
        else:
            print("[Info] 分类模式未开启或无需分类头，已忽略冲突的分类权重。")

    model.to(args.device)
    model.eval()

    print(f"Model loaded from {args.checkpoint}")
    return model


def load_class_mapping(class_csv_path: Optional[str]) -> Optional[Dict[int, str]]:
    """加载 CSV 类别映射，返回 class_id -> name 的字典"""
    if not class_csv_path:
        return None

    csv_path = Path(class_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Class mapping CSV not found: {csv_path}")

    with csv_path.open('r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'class_id' not in reader.fieldnames or 'class_name' not in reader.fieldnames:
            raise ValueError('CSV 必须包含 class_id 和 class_name 两列。')

        mapping: Dict[int, str] = {}
        for row in reader:
            class_id = int(row['class_id'])
            class_name = row['class_name']
            mapping[class_id] = class_name

    if not mapping:
        raise ValueError(f"CSV {csv_path} 中未找到任何类别映射。")

    return mapping


def preprocess_image(image_path, transform):
    """预处理单张图像"""
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image)
    return tensor.unsqueeze(0)

def get_artist_tags_string(image, args=None):
    """获取艺术家预测结果的标签字符串（逗号分隔格式）
    
    Args:
        image: PIL图像对象或numpy数组
        args: 参数对象，如果为None则使用默认参数
        
    Returns:
        str: 逗号分隔的艺术家名称字符串
    """
    try:
        model_dir = C.path_models_root
        if args is None:
            # 创建一个简单的参数对象
            class Args:
                def __init__(self):
                    # 获取默认模型路径
                    self.checkpoint = os.path.join(model_dir, 'lsnet', 'kaloscope', 'best_checkpoint.pth')
                    self.class_map = os.path.join(model_dir, 'lsnet', 'kaloscope', 'class_mapping.csv')
                    self.top_k = 5
                    self.mode = 'classification'
                    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    self.model = 'lsnet_xl_artist'
                    self.num_classes = None
                    self.feature_dim = None
                    self.allow_head_reinit = False
                    self.threshold = 0.0
            args = Args()
        else:
            print(f"[Artist Inference] 使用外部参数: model={args.model}, device={args.device}")
        
        # 检查模型路径是否存在，如果不存在则下载
        if not os.path.exists(args.checkpoint):
            print(f"[Artist Inference] 模型文件不存在，准备下载: {args.checkpoint}")

            checkpoint_dir = os.path.join(model_dir, 'lsnet', 'kaloscope')
            checkpoint_filename = 'best_checkpoint.pth'

            model_url = "https://www.modelscope.cn/models/Heathcliff02/Kaloscope/resolve/master/best_checkpoint.pth"
            try:
                load_file_from_url(
                    url=model_url,
                    model_dir=checkpoint_dir,
                    file_name=checkpoint_filename
                )
            except Exception as download_error:
                error_msg = f"下载模型失败: {str(download_error)}"
                print(f"[Artist Inference] 错误: {error_msg}")
                return error_msg

        if args.class_map and not os.path.exists(args.class_map):
            print(f"[Artist Inference] 类别映射文件不存在，准备下载: {args.class_map}")

            class_map_dir = os.path.join(model_dir, 'lsnet', 'kaloscope')
            class_map_filename = 'class_mapping.csv'
            
            class_map_url = "https://www.modelscope.cn/models/Heathcliff02/Kaloscope/resolve/master/class_mapping.csv"  # 需要替换为实际的类别映射文件URL
            try:
                load_file_from_url(
                    url=class_map_url,
                    model_dir=class_map_dir,
                    file_name=class_map_filename
                )
            except Exception as download_error:
                error_msg = f"下载类别映射文件失败: {str(download_error)}"
                print(f"[Artist Inference] 错误: {error_msg}")
                return error_msg

        state_dict = load_checkpoint_state(args.checkpoint)

        class_mapping = load_class_mapping(args.class_map) if args.class_map else None

        args.num_classes = resolve_num_classes(args.num_classes, class_mapping, state_dict)
        args.feature_dim = resolve_feature_dim(args.feature_dim, state_dict)

        model = load_model(args, state_dict)

        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = image.convert('RGB')

        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            image_tensor = image_tensor.to(args.device)
            logits = model(image_tensor, return_features=False)
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=min(args.top_k, probs.size(-1)), dim=-1)
            
            results = []
            for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
                if prob >= args.threshold:
                    class_id = int(idx)
                    class_name = class_mapping.get(class_id, f"Class {class_id}") if class_mapping else f"Class {class_id}"
                    results.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'probability': float(prob)
                    })

            if len(results) > args.top_k:
                results = results[:args.top_k]

        tags = [res['class_name'] for res in results]
        tag_string = ",".join(tags)
        return tag_string
        
    except Exception as e:
        print(f"[Artist Inference] 处理图像时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return f"错误: {str(e)}"