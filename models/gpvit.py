# https://github.com/ChenhongyiYang/GPViT/blob/main/mmcls/gpvit_dev/models/utils/attentions.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Sequence

from einops import rearrange

from mmcv.cnn import build_norm_layer, build_conv_layer, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, AdaptivePadding, build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks import DropPath
import torch.utils.checkpoint as cp

import collections.abc
import warnings
from itertools import repeat

from models.group_vit import PatchEmbed


def _ntuple(n):
    """A `to_tuple` function generator.
    It returns a function, this function will repeat the input to a tuple of
    length ``n`` if the input is not an Iterable object, otherwise, return the
    input directly.
    Args:
        n (int): The number of the target length.
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    """Resize pos_embed weights.
    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.
    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    dst_weight = F.interpolate(
        src_weight, size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)

    return torch.cat((extra_tokens, dst_weight), dim=1)


def img2windows(img, H_sp, W_sp):
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class LePEAttention(nn.Module):
    def __init__(self, dim, mode, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        assert mode in (0, 1)
        self.mode = mode
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        if self.mode == 0:
            H_sp, W_sp = H, self.split_size
        else:
            H_sp, W_sp = self.split_size, W
        x = img2windows(x, H_sp, W_sp)
        x = x.reshape(-1, H_sp * W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, hw_shape, func):
        B, N, C = x.shape
        H, W = hw_shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        if self.mode == 0:
            H_sp, W_sp = H, self.split_size
        else:
            H_sp, W_sp = self.split_size, W
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'
        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        x = x.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv, hw_shape):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]
        ### Img2Window
        H, W = hw_shape
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q, hw_shape)
        k = self.im2cswin(k, hw_shape)
        v, lepe = self.get_lepe(v, hw_shape, self.get_v)

        if self.mode == 0:
            H_sp, W_sp = H, self.split_size
        else:
            H_sp, W_sp = self.split_size, W

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, H_sp * W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, H_sp, W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


class LePEAttnSimpleDWBlock(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,  # For convenience, we use window size to denote split size
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path=0.,
                 attn_cfgs=dict(),
                 ffn_cfgs=dict(),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super().__init__(init_cfg)
        self.with_cp = with_cp
        self.dim = embed_dims
        self.num_heads = num_heads
        self.split_size = window_size
        self.ffn_ratio = ffn_ratio
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=True)

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.branch_num = 2
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(0.)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attns = nn.ModuleList([
            LePEAttention(
                embed_dims // 2, mode=i,
                split_size=self.split_size, num_heads=num_heads // 2, dim_out=embed_dims // 2,
                qk_scale=None, attn_drop=0., proj_drop=drop_rate)
            for i in range(self.branch_num)])

        _ffn_cfgs = {
            'embed_dims': embed_dims,
            'feedforward_channels': int(embed_dims * ffn_ratio),
            'num_fcs': 2,
            'ffn_drop': drop_rate,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'act_cfg': dict(type='GELU'),
            **ffn_cfgs
        }
        self.ffn = FFN(**_ffn_cfgs)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.dw = nn.Conv2d(embed_dims, embed_dims, kernel_size=(3, 3), padding=(1, 1), bias=False, groups=embed_dims)

    def forward(self, x, hw_shape):
        """
        x: B, H*W, C
        """

        def _inner_forward(x, hw_shape):
            H, W = hw_shape
            B, L, C = x.shape
            assert L == H * W, "flatten img_tokens has wrong size"
            img = self.norm1(x)
            qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3).contiguous()

            x1 = self.attns[0](qkv[:, :, :, :C // 2], hw_shape)
            x2 = self.attns[1](qkv[:, :, :, C // 2:], hw_shape)
            attened_x = torch.cat([x1, x2], dim=2)
            attened_x = self.proj(attened_x)
            x = x + self.drop_path(attened_x)

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            B, L, C = x.shape
            x = x.permute(0, 2, 1).contiguous().reshape(B, C, hw_shape[0], hw_shape[1])
            x = self.dw(x)
            x = x.reshape(B, C, L).permute(0, 2, 1).contiguous()
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, hw_shape)
        else:
            x = _inner_forward(x, hw_shape)
        return x


class MLPMixerLayer(nn.Module):
    def __init__(self,
                 num_patches,
                 embed_dims,
                 patch_expansion,
                 channel_expansion,
                 drop_path,
                 drop_out,
                 **kwargs):
        super(MLPMixerLayer, self).__init__()

        patch_mix_dims = int(patch_expansion * embed_dims)
        channel_mix_dims = int(channel_expansion * embed_dims)

        self.patch_mixer = nn.Sequential(
            nn.Linear(num_patches, patch_mix_dims),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(patch_mix_dims, num_patches),
            nn.Dropout(drop_out)
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(embed_dims, channel_mix_dims),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(channel_mix_dims, embed_dims),
            nn.Dropout(drop_out)
        )

        self.drop_path1 = build_dropout(dict(type='DropPath', drop_prob=drop_path))
        self.drop_path2 = build_dropout(dict(type='DropPath', drop_prob=drop_path))

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, x):
        x = x + self.drop_path1(self.patch_mixer(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path2(self.channel_mixer(self.norm2(x)))
        return x


class MLPMixer(BaseModule):
    def __init__(self,
                 num_patches,
                 embed_dims,
                 patch_expansion=0.5,
                 channel_expansion=4.0,
                 depth=1,
                 drop_path=0.,
                 drop_out=0.,
                 init_cfg=None,
                 **kwargs):
        super(MLPMixer, self).__init__(init_cfg)
        layers = [
            MLPMixerLayer(num_patches, embed_dims, patch_expansion, channel_expansion, drop_path, drop_out)
            for _ in range(depth)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LightAttModule(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 q_project=True,
                 k_project=True,
                 v_project=True,
                 proj_after_att=True):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias) if q_project else None
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias) if k_project else None
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias) if v_project else None

        self.attn_drop = nn.Dropout(attn_drop)

        if proj_after_att:
            self.proj = nn.Sequential(nn.Linear(dim, out_dim), nn.Dropout(proj_drop))
        else:
            self.proj = None

    def forward(self, query, key, value, att_bias=None):
        bq, nq, cq = query.shape
        bk, nk, ck = key.shape
        bv, nv, cv = value.shape

        # [bq, nh, nq, cq//nh]
        if self.q_proj:
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=bq, n=nq,
                          c=cq // self.num_heads)
        else:
            q = rearrange(query, 'b n (h c)-> b h n c', h=self.num_heads, b=bq, n=nq, c=cq // self.num_heads)
        # [bk, nh, nk, ck//nh]
        if self.k_proj:
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=bk, n=nk, c=ck // self.num_heads)
        else:
            k = rearrange(key, 'b n (h c)-> b h n c', h=self.num_heads, b=bk, n=nk, c=ck // self.num_heads)
        # [bv, nh, nv, cv//nh]
        if self.v_proj:
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=bv, n=nv,
                          c=cv // self.num_heads)
        else:
            v = rearrange(value, 'b n (h c)-> b h n c', h=self.num_heads, b=bv, n=nv, c=cv // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if att_bias is not None:
            attn = attn + att_bias.unsqueeze(dim=1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (bq, self.num_heads, nq, nk)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=bq, n=nq, c=cv // self.num_heads)
        if self.proj:
            out = self.proj(out)
        return out


class FullAttnModule(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 q_project=True):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias) if q_project else None
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, att_bias=None):
        bq, nq, cq = query.shape
        bk, nk, ck = key.shape
        bv, nv, cv = value.shape

        # [bq, nh, nq, cq//nh]
        if self.q_proj:
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=bq, n=nq,
                          c=cq // self.num_heads)
        else:
            q = rearrange(query, 'b n (h c)-> b h n c', h=self.num_heads, b=bq, n=nq, c=cq // self.num_heads)
        # [bk, nh, nk, ck//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=bk, n=nk, c=ck // self.num_heads)
        # [bv, nh, nv, cv//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=bv, n=nv, c=cv // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if att_bias is not None:
            attn = attn + att_bias.unsqueeze(dim=1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (bq, self.num_heads, nq, nk)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=bq, n=nq, c=cv // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FullAttnCatBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 ffn_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 key_is_query=False,
                 value_is_key=False,
                 q_project=True,
                 with_cp=False,
                 **kwargs):
        super().__init__()
        self.with_cp = with_cp

        self.norm_query = build_norm_layer(norm_cfg, embed_dims)[1]

        if not key_is_query:
            self.norm_key = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm_key = None
        self.key_is_query = key_is_query

        if not value_is_key:
            self.norm_value = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm_value = None
        self.value_is_key = value_is_key

        self.attn = FullAttnModule(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            q_project=q_project)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        _ffn_cfgs = {
            'embed_dims': embed_dims,
            'feedforward_channels': int(embed_dims * ffn_ratio),
            'num_fcs': 2,
            'ffn_drop': drop,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'act_cfg': act_cfg,
        }
        self.ffn = FFN(**_ffn_cfgs)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.proj = nn.Linear(embed_dims * 2, embed_dims, bias=True)

    def forward(self, query, key, value, att_bias=None):
        def _inner_forward(query, key, value, att_bias):
            q = self.norm_query(query)
            k = q if self.key_is_query else self.norm_key(key)
            v = k if self.value_is_key else self.norm_value(value)

            x = torch.cat((query, self.drop_path(self.attn(q, k, v, att_bias=att_bias))), dim=-1)
            x = self.proj(x)
            x = self.ffn(self.norm2(x), identity=x)
            return x

        if self.with_cp:
            return cp.checkpoint(_inner_forward, query, key, value, att_bias)
        else:
            return _inner_forward(query, key, value, att_bias)


class LightGroupAttnBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 ffn_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 key_is_query=False,
                 value_is_key=False,
                 with_cp=False):
        super().__init__()

        self.with_cp = with_cp

        self.norm_query = build_norm_layer(norm_cfg, embed_dims)[1]

        if not key_is_query:
            self.norm_key = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm_key = None
        self.key_is_query = key_is_query

        if not value_is_key:
            self.norm_value = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm_value = None
        self.value_is_key = value_is_key

        self.attn = LightAttModule(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            q_project=False,
            k_project=True,
            v_project=False,
            proj_after_att=False)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, key, value, att_bias=None):
        def _inner_forward(query, key, value, att_bias):
            q = self.norm_query(query)
            k = q if self.key_is_query else self.norm_key(key)
            v = k if self.value_is_key else self.norm_value(value)
            x = self.drop_path(self.attn(q, k, v, att_bias=att_bias))
            return x

        if self.with_cp:
            return cp.checkpoint(_inner_forward, query, key, value, att_bias)
        else:
            return _inner_forward(query, key, value, att_bias)


class GPBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 depth,
                 num_group_heads,
                 num_ungroup_heads,
                 num_group_token,
                 ffn_ratio=4.,
                 qkv_bias=True,
                 group_qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 with_cp=False,
                 group_att_cfg=dict(),
                 fwd_att_cfg=dict(),
                 ungroup_att_cfg=dict(),
                 **kwargs):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_group_token = num_group_token
        self.with_cp = with_cp

        self.group_token = nn.Parameter(torch.zeros(1, num_group_token, embed_dims))
        trunc_normal_(self.group_token, std=.02)

        _group_att_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_group_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            qk_scale=group_qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=0.,
            key_is_query=False,
            value_is_key=True,
            with_cp=with_cp)
        _group_att_cfg.update(group_att_cfg)
        self.group_layer = LightGroupAttnBlock(**_group_att_cfg)

        _mixer_cfg = dict(
            num_patches=num_group_token,
            embed_dims=embed_dims,
            patch_expansion=0.5,
            channel_expansion=4.0,
            depth=depth,
            drop_path=drop_path)
        _mixer_cfg.update(fwd_att_cfg)
        self.mixer = MLPMixer(**_mixer_cfg)

        _ungroup_att_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_ungroup_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            qk_scale=None,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            key_is_query=False,
            value_is_key=True,
            with_cp=with_cp)
        _ungroup_att_cfg.update(ungroup_att_cfg)
        self.un_group_layer = FullAttnCatBlock(**_ungroup_att_cfg)

        self.dwconv = torch.nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=(3, 3), padding=(1, 1), bias=False, groups=embed_dims),
            nn.BatchNorm2d(num_features=embed_dims),
            nn.ReLU(True))

    def forward(self, x, hw_shape):
        """
        Args:
            x: image tokens, shape [B, L, C]
            hw_shape: tuple or list (H, W)
        Returns:
            proj_tokens: shape [B, L, C]
        """
        B, L, C = x.size()
        group_token = self.group_token.expand(x.size(0), -1, -1)
        gt = group_token

        gt = self.group_layer(query=gt, key=x, value=x)
        gt = self.mixer(gt)
        ungroup_tokens = self.un_group_layer(query=x, key=gt, value=gt)
        ungroup_tokens = ungroup_tokens.permute(0, 2, 1).contiguous().reshape(B, C, hw_shape[0], hw_shape[1])
        proj_tokens = self.dwconv(ungroup_tokens).view(B, C, -1).permute(0, 2, 1).contiguous().view(B, L, C)
        return proj_tokens


class GPViT(nn.Module):
    arch_zoo = {
        **dict.fromkeys(
            ['L1', 'L1'], {
                'embed_dims': 216,
                'patch_size': 8,
                'window_size': 2,
                'num_layers': 12,
                'num_heads': 12,
                'num_group_heads': 6,
                'num_group_forward_heads': 6,
                'num_ungroup_heads': 6,
                'ffn_ratio': 4.,
                'patch_embed': dict(type='ConvPatchEmbed', num_convs=0),
                'mlpmixer_depth': 1,
                'group_layers': {1: 64, 4: 32, 7: 32, 10: 16},
                'drop_path_rate': 0.2
            }),
        **dict.fromkeys(
            ['L2', 'L2'], {
                'embed_dims': 348,
                'patch_size': 8,
                'window_size': 2,
                'num_layers': 12,
                'num_heads': 12,
                'num_group_heads': 6,
                'num_group_forward_heads': 6,
                'num_ungroup_heads': 6,
                'ffn_ratio': 4.,
                'patch_embed': dict(type='ConvPatchEmbed', num_convs=1),
                'mlpmixer_depth': 1,
                'group_layers': {1: 64, 4: 32, 7: 32, 10: 16},
                'drop_path_rate': 0.2
            }),
        **dict.fromkeys(
            ['L3', 'L3'], {
                'embed_dims': 432,
                'patch_size': 8,
                'window_size': 2,
                'num_layers': 12,
                'num_heads': 12,
                'num_group_heads': 6,
                'num_group_forward_heads': 6,
                'num_ungroup_heads': 6,
                'ffn_ratio': 4.,
                'patch_embed': dict(type='ConvPatchEmbed', num_convs=1),
                'mlpmixer_depth': 1,
                'group_layers': {1: 64, 4: 32, 7: 32, 10: 16},
                'drop_path_rate': 0.3
            }),
        **dict.fromkeys(
            ['L4', 'L4'], {
                'embed_dims': 624,
                'patch_size': 8,
                'window_size': 2,
                'num_layers': 12,
                'num_heads': 12,
                'num_group_heads': 6,
                'num_group_forward_heads': 6,
                'num_ungroup_heads': 6,
                'ffn_ratio': 4.,
                'patch_embed': dict(type='ConvPatchEmbed', num_convs=2),
                'mlpmixer_depth': 1,
                'group_layers': {1: 64, 4: 32, 7: 32, 10: 16},
                'drop_path_rate': 0.3
            }),
    }

    def __init__(self,
                 arch='',
                 img_size=224,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None,
                 test_cfg=dict(vis_group=False),
                 convert_syncbn=False,
                 freeze_patch_embed=False,
                 **kwargs):
        super(GPViT, self).__init__()
        self.arch = arch

        if isinstance(arch, str):
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.test_cfg = test_cfg
        if kwargs.get('embed_dims', None) is not None:
            self.embed_dims = kwargs.get('embed_dims', None)
        else:
            self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.img_size = to_2tuple(img_size)
        self.convert_syncbn = convert_syncbn

        # set gradient checkpoint
        _att_with_cp = False
        if _att_with_cp is None:
            if not hasattr(self, "att_with_cp"):
                self.att_with_cp = self.arch_settings['with_cp']
        else:
            self.att_with_cp = _att_with_cp
        _group_with_cp = kwargs.pop('group_with_cp', None)
        if _group_with_cp is None:
            if not hasattr(self, "group_with_cp"):
                self.group_with_cp = self.att_with_cp
        else:
            self.group_with_cp = _group_with_cp

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            patch_size=self.arch_settings['patch_size'],
            stride=self.arch_settings['patch_size'],
        )
        _patch_cfg.update(patch_cfg)
        _patch_cfg.update(self.arch_settings['patch_embed'])
        self.patch_embed = PatchEmbed(
            img_size=_patch_cfg['input_size'],
            kernel_size=_patch_cfg['patch_size'],
            stride=_patch_cfg['stride'],
            padding=0,
            in_chans=_patch_cfg['in_channels'],
            embed_dim=_patch_cfg['embed_dims'],
        )
        self.freeze_patch_embed = freeze_patch_embed

        self.patch_size = self.arch_settings['patch_size']

        self.patch_resolution = (_patch_cfg['input_size'] // _patch_cfg['patch_size'], ) * 2
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        if drop_path_rate < 0:
            _drop_path_rate = self.arch_settings.get('drop_path_rate', None)
            if _drop_path_rate is None:
                raise ValueError
        else:
            _drop_path_rate = drop_path_rate

        dpr = np.linspace(0, _drop_path_rate, self.num_layers)
        self.drop_path_rate = _drop_path_rate

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers

        for i in range(self.num_layers):
            _arch_settings = copy.deepcopy(self.arch_settings)
            if i not in _arch_settings['group_layers'].keys():
                _layer_cfg = dict(
                    embed_dims=self.embed_dims,
                    num_heads=_arch_settings['num_heads'],
                    window_size=_arch_settings['window_size'],
                    ffn_ratio=_arch_settings['ffn_ratio'],
                    drop_rate=drop_rate,
                    drop_path=dpr[i],
                    norm_cfg=norm_cfg,
                    with_cp=self.att_with_cp)
                _layer_cfg.update(layer_cfgs[i])
                attn_layer = LePEAttnSimpleDWBlock(**_layer_cfg)
                self.layers.append(attn_layer)
            else:
                _layer_cfg = dict(
                    embed_dims=self.embed_dims,
                    depth=_arch_settings['mlpmixer_depth'],
                    num_group_heads=_arch_settings['num_group_heads'],
                    num_forward_heads=_arch_settings['num_group_forward_heads'],
                    num_ungroup_heads=_arch_settings['num_ungroup_heads'],
                    num_group_token=_arch_settings['group_layers'][i],
                    ffn_ratio=_arch_settings['ffn_ratio'],
                    drop_path=dpr[i],
                    with_cp=self.group_with_cp)
                group_layer = GPBlock(**_layer_cfg)
                self.layers.append(group_layer)
        self.final_norm = final_norm
        # assert final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        for i in out_indices:
            if i != self.num_layers - 1:
                if norm_cfg is not None:
                    norm_layer = build_norm_layer(norm_cfg, self.embed_dims)[1]
                else:
                    norm_layer = nn.Identity()
                self.add_module(f'norm{i}', norm_layer)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        super(GPViT, self).init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            trunc_normal_(self.pos_embed, std=0.02)
        self.set_freeze_patch_embed()

    def set_freeze_patch_embed(self):
        if self.freeze_patch_embed:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        pos_embed = resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=0)

        x = x + pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, hw_shape=patch_resolution)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x.reshape(B, *patch_resolution, C)
                if i != self.num_layers - 1:
                    norm_layer = getattr(self, f'norm{i}')
                    patch_token = norm_layer(patch_token)
                patch_token = patch_token.permute(0, 3, 1, 2)
                outs.append(patch_token)
        return tuple(outs)

if __name__ == '__main__':
    model = GPViT(arch='L1')
    print(model)
    img = torch.randn(1, 3, 224, 224)
    out = model(img)
    print(out[0].shape)