#!/usr/bin/env python3

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
# from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg, _update_default_kwargs
# from .registry import register_pip_model
# from fastervit.models.registry import register_pip_model
from pathlib import Path
import numpy as np


def _cfg(url='', **kwargs):
    return {'url': url,
            'num_classes': 1000,
            'input_size': (3, 224, 224),
            'pool_size': None,
            'crop_pct': 0.875,
            'interpolation': 'bicubic',
            'fixed_input_size': True,
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            **kwargs
            }


default_cfgs = {
    'faster_vit_0_224': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_0_224_1k.pth.tar',
                             crop_pct=0.875,
                             input_size=(3, 224, 224),
                             crop_mode='center'),
    'faster_vit_1_224': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_1_224_1k.pth.tar',
                             crop_pct=1.0,
                             input_size=(3, 224, 224),
                             crop_mode='center'),
    'faster_vit_2_224': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_2_224_1k.pth.tar',
                             crop_pct=1.0,
                             input_size=(3, 224, 224),
                             crop_mode='center'),
    'faster_vit_3_224': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_3_224_1k.pth.tar',
                             crop_pct=1.0,
                             input_size=(3, 224, 224),
                             crop_mode='center'),
    'faster_vit_4_224': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_224_1k.pth.tar',
                             crop_pct=1.0,
                             input_size=(3, 224, 224),
                             crop_mode='center'),
    'faster_vit_5_224': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_5_224_1k.pth.tar',
                             crop_pct=1.0,
                             input_size=(3, 224, 224),
                             crop_mode='center'),
    'faster_vit_6_224': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_6_224_1k.pth.tar',
                             crop_pct=1.0,
                             input_size=(3, 224, 224),
                             crop_mode='center'),
    'faster_vit_4_21k_224': _cfg(
        url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_21k_224_w14.pth.tar',
        crop_pct=0.95,
        input_size=(3, 224, 224),
        crop_mode='squash'),
    'faster_vit_4_21k_384': _cfg(
        url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_21k_384_w24.pth.tar',
        crop_pct=1.0,
        input_size=(3, 384, 384),
        crop_mode='squash'),
    'faster_vit_4_21k_512': _cfg(
        url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_21k_512_w32.pth.tar',
        crop_pct=1.0,
        input_size=(3, 512, 512),
        crop_mode='squash'),
    'faster_vit_4_21k_768': _cfg(
        url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_21k_768_w48.pth.tar',
        crop_pct=0.93,
        input_size=(3, 768, 768),
        crop_mode='squash'),
}


def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, B):
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x


def ct_dewindow(ct, W, H, window_size):
    bs = ct.shape[0]
    N = ct.shape[2]
    ct2 = ct.view(-1, W // window_size, H // window_size, window_size, window_size, N).permute(0, 5, 1, 3, 2, 4)
    ct2 = ct2.reshape(bs, N, W * H).transpose(1, 2)
    return ct2


def ct_window(ct, W, H, window_size):
    bs = ct.shape[0]
    N = ct.shape[2]
    ct = ct.view(bs, H // window_size, window_size, W // window_size, window_size, N)
    ct = ct.permute(0, 1, 3, 2, 4, 5)
    return ct


def _load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(model,
                     filename,
                     map_location='cpu',
                     strict=False,
                     logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class PosEmbMLPSwinv2D(nn.Module):
    def __init__(self,
                 window_size,
                 pretrained_window_size,
                 num_heads, seq_length,
                 ct_correct=False,
                 no_log=False):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)

        if not no_log:
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.grid_exists = False
        self.pos_emb = None
        self.deploy = False
        relative_bias = torch.zeros(1, num_heads, seq_length, seq_length)
        self.seq_length = seq_length
        self.register_buffer("relative_bias", relative_bias)
        self.ct_correct = ct_correct

    def switch_to_deploy(self):
        self.deploy = True

    def forward(self, input_tensor, local_window_size):
        if self.deploy:
            input_tensor += self.relative_bias
            return input_tensor
        else:
            self.grid_exists = False

        if not self.grid_exists:
            self.grid_exists = True

            relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
            n_global_feature = input_tensor.shape[2] - local_window_size
            if n_global_feature > 0 and self.ct_correct:

                step_for_ct = self.window_size[0] / (n_global_feature ** 0.5 + 1)
                seq_length = int(n_global_feature ** 0.5)
                indices = []
                for i in range(seq_length):
                    for j in range(seq_length):
                        ind = (i + 1) * step_for_ct * self.window_size[0] + (j + 1) * step_for_ct
                        indices.append(int(ind))

                top_part = relative_position_bias[:, indices, :]
                lefttop_part = relative_position_bias[:, indices, :][:, :, indices]
                left_part = relative_position_bias[:, :, indices]
            relative_position_bias = torch.nn.functional.pad(relative_position_bias, (n_global_feature,
                                                                                      0,
                                                                                      n_global_feature,
                                                                                      0)).contiguous()
            if n_global_feature > 0 and self.ct_correct:
                relative_position_bias = relative_position_bias * 0.0
                relative_position_bias[:, :n_global_feature, :n_global_feature] = lefttop_part
                relative_position_bias[:, :n_global_feature, n_global_feature:] = top_part
                relative_position_bias[:, n_global_feature:, :n_global_feature] = left_part

            self.pos_emb = relative_position_bias.unsqueeze(0)
            self.relative_bias = self.pos_emb

        input_tensor += self.pos_emb
        return input_tensor


class PosEmbMLPSwinv1D(nn.Module):
    def __init__(self,
                 dim,
                 rank=2,
                 seq_length=4,
                 conv=False):
        super().__init__()
        self.rank = rank
        if not conv:
            self.cpb_mlp = nn.Sequential(nn.Linear(self.rank, 512, bias=True),
                                         nn.ReLU(),
                                         nn.Linear(512, dim, bias=False))
        else:
            self.cpb_mlp = nn.Sequential(nn.Conv1d(self.rank, 512, 1, bias=True),
                                         nn.ReLU(),
                                         nn.Conv1d(512, dim, 1, bias=False))
        self.grid_exists = False
        self.pos_emb = None
        self.deploy = False
        relative_bias = torch.zeros(1, seq_length, dim)
        self.register_buffer("relative_bias", relative_bias)
        self.conv = conv

    def switch_to_deploy(self):
        self.deploy = True

    def forward(self, input_tensor):
        seq_length = input_tensor.shape[1] if not self.conv else input_tensor.shape[2]
        if self.deploy:
            return input_tensor + self.relative_bias
        else:
            self.grid_exists = False
        if not self.grid_exists:
            self.grid_exists = True
            if self.rank == 1:
                relative_coords_h = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
                relative_coords_h -= seq_length // 2
                relative_coords_h /= (seq_length // 2)
                relative_coords_table = relative_coords_h
                self.pos_emb = self.cpb_mlp(relative_coords_table.unsqueeze(0).unsqueeze(2))
                self.relative_bias = self.pos_emb
            else:
                seq_length = int(seq_length ** 0.5)
                relative_coords_h = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
                relative_coords_w = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
                relative_coords_table = torch.stack(
                    torch.meshgrid([relative_coords_h, relative_coords_w])).contiguous().unsqueeze(0)
                relative_coords_table -= seq_length // 2
                relative_coords_table /= (seq_length // 2)
                if not self.conv:
                    self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2).transpose(1, 2))
                else:
                    self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2))
                self.relative_bias = self.pos_emb
        input_tensor = input_tensor + self.pos_emb
        return input_tensor


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        """
        Args:
            in_features: input features dimension.
            hidden_features: hidden features dimension.
            out_features: output features dimension.
            act_layer: activation function.
            drop: dropout rate.
        """

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1, x_size[-1])
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.view(x_size)
        return x


class Downsample(nn.Module):
    """
    Down-sampling block based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.norm = LayerNorm2d(dim)
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(nn.Module):
    """
    Conv block based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()
        """
        Args:
            drop_path: drop path.
            layer_scale: layer scale coefficient.
            kernel_size: kernel size.
        """
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, global_feature=None):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x, global_feature


class WindowAttention(nn.Module):
    """
    Window attention based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 resolution=0,
                 seq_length=0):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
            resolution: feature resolution.
            seq_length: sequence length.
        """
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # attention positional bias
        self.pos_emb_funct = PosEmbMLPSwinv2D(window_size=[resolution, resolution],
                                              pretrained_window_size=[resolution, resolution],
                                              num_heads=num_heads,
                                              seq_length=seq_length)

        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.pos_emb_funct(attn, self.resolution ** 2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HAT(nn.Module):
    """
    Hierarchical attention (HAT) based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1.,
                 window_size=7,
                 last=False,
                 layer_scale=None,
                 ct_size=1,
                 do_propagation=False):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
            act_layer: activation function.
            norm_layer: normalization layer.
            sr_ratio: input to window size ratio.
            window_size: window size.
            last: last layer flag.
            layer_scale: layer scale coefficient.
            ct_size: spatial dimension of carrier token local window.
            do_propagation: enable carrier token propagation.
        """
        # positional encoding for windowed attention tokens
        self.pos_embed = PosEmbMLPSwinv1D(dim, rank=2, seq_length=window_size ** 2)
        self.norm1 = norm_layer(dim)
        # number of carrier tokens per every window
        cr_tokens_per_window = ct_size ** 2 if sr_ratio > 1 else 0
        # total number of carrier tokens
        cr_tokens_total = cr_tokens_per_window * sr_ratio * sr_ratio
        self.cr_window = ct_size
        self.attn = WindowAttention(dim,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop,
                                    resolution=window_size,
                                    seq_length=window_size ** 2 + cr_tokens_per_window)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.window_size = window_size

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma3 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma4 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # if do hierarchical attention, this part is for carrier tokens
            self.hat_norm1 = norm_layer(dim)
            self.hat_norm2 = norm_layer(dim)
            self.hat_attn = WindowAttention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, resolution=int(cr_tokens_total ** 0.5),
                seq_length=cr_tokens_total)

            self.hat_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.hat_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.hat_pos_embed = PosEmbMLPSwinv1D(dim, rank=2, seq_length=cr_tokens_total)
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
            self.upsampler = nn.Upsample(size=window_size, mode='nearest')

        # keep track for the last block to explicitly add carrier tokens to feature maps
        self.last = last
        self.do_propagation = do_propagation

    def forward(self, x, carrier_tokens):
        B, T, N = x.shape
        ct = carrier_tokens
        x = self.pos_embed(x)

        if self.sr_ratio > 1:
            # do hierarchical attention via carrier tokens
            # first do attention for carrier tokens
            Bg, Ng, Hg = ct.shape

            # ct are located quite differently
            ct = ct_dewindow(ct, self.cr_window * self.sr_ratio, self.cr_window * self.sr_ratio, self.cr_window)

            # positional bias for carrier tokens
            ct = self.hat_pos_embed(ct)

            # attention plus mlp
            ct = ct + self.hat_drop_path(self.gamma1 * self.hat_attn(self.hat_norm1(ct)))
            ct = ct + self.hat_drop_path(self.gamma2 * self.hat_mlp(self.hat_norm2(ct)))

            # ct are put back to windows
            ct = ct_window(ct, self.cr_window * self.sr_ratio, self.cr_window * self.sr_ratio, self.cr_window)

            ct = ct.reshape(x.shape[0], -1, N)
            # concatenate carrier_tokens to the windowed tokens
            x = torch.cat((ct, x), dim=1)

        # window attention together with carrier tokens
        x = x + self.drop_path(self.gamma3 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma4 * self.mlp(self.norm2(x)))

        if self.sr_ratio > 1:
            # for hierarchical attention we need to split carrier tokens and window tokens back
            ctr, x = x.split([x.shape[1] - self.window_size * self.window_size, self.window_size * self.window_size],
                             dim=1)
            ct = ctr.reshape(Bg, Ng, Hg)  # reshape carrier tokens.
            if self.last and self.do_propagation:
                # propagate carrier token information into the image
                ctr_image_space = ctr.transpose(1, 2).reshape(B, N, self.cr_window, self.cr_window)
                x = x + self.gamma1 * self.upsampler(ctr_image_space.to(dtype=torch.float32)).flatten(2).transpose(1,
                                                                                                                   2).to(
                    dtype=x.dtype)
        return x, ct


class TokenInitializer(nn.Module):
    """
    Carrier token Initializer based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 window_size,
                 ct_size=1):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            window_size: window size.
            ct_size: spatial dimension of carrier token local window
        """
        super().__init__()

        output_size = int(ct_size * input_resolution / window_size)
        stride_size = int(input_resolution / output_size)
        kernel_size = input_resolution - (output_size - 1) * stride_size
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        to_global_feature = nn.Sequential()
        to_global_feature.add_module("pos", self.pos_embed)
        to_global_feature.add_module("pool", nn.AvgPool2d(kernel_size=kernel_size, stride=stride_size))
        self.to_global_feature = to_global_feature
        self.window_size_ct = ct_size

    def forward(self, x):
        x = self.to_global_feature(x)
        B, C, H, W = x.shape
        ct = x.view(B, C, H // self.window_size_ct, self.window_size_ct, W // self.window_size_ct, self.window_size_ct)
        ct = ct.permute(0, 2, 4, 3, 5, 1).reshape(-1, H * W, C)
        return ct


# if __name__ == '__main__':
class FasterViT(nn.Module):
        """
        FasterViT based on: "Hatamizadeh et al.,
        FasterViT: Fast Vision Transformers with Hierarchical Attention
        """
        def __init__(self, dim=64, ct_size=2, window_size=[2, 2, 2, 2], resolution=224):
            """
            Args:
                dim: feature size dimension.
                in_dim: inner-plane feature size dimension.
                depths: layer depth.
                window_size: window size.
                ct_size: spatial dimension of carrier token local window.
                mlp_ratio: MLP ratio.
                num_heads: number of attention head.
                resolution: image resolution.
                drop_path_rate: drop path rate.
                in_chans: input channel dimension.
                num_classes: number of classes.
                qkv_bias: bool argument for query, key, value learnable bias.
                qk_scale: bool argument to scaling query, key.
                drop_rate: dropout rate.
                attn_drop_rate: attention dropout rate.
                layer_scale: layer scale coefficient.
                layer_scale_conv: conv layer scale coefficient.
                layer_norm_last: last stage layer norm flag.
                hat: hierarchical attention flag.
                do_propagation: enable carrier token propagation.
            """
            super().__init__()
            i = 2
            # window_size = window_size # [2, 2, 2, 2]
            # self.window_size = window_size[i]
            self.window_size = window_size
            ct_size = ct_size # 2
            dim = dim
            resolution = resolution # 要能整除window_size, 22

            depths = [2, 3, 6, 5]
            num_heads = [2, 4, 8, 16]
            mlp_ratio = 4
            drop_path_rate = 0.2
            model_path = "/tmp/faster_vit_0.pth.tar"
            hat = [False, False, True, False]

            depth = depths[i]
            num_heads = num_heads[i]
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
            drop_path = dpr[sum(depths[:i]):sum(depths[:i + 1])]
            input_resolution = resolution
            only_local = not hat[i]

            qkv_bias = True
            qk_scale = None
            drop = 0.
            attn_drop = 0.
            # drop_path=0.
            layer_scale = None
            do_propagation = False
            sr_ratio = input_resolution // self.window_size if not only_local else 1
            self.hat = HAT(dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    sr_ratio=sr_ratio,
                    window_size=self.window_size,
                    last=(i == depth - 1),
                    layer_scale=layer_scale,
                    ct_size=ct_size,
                    do_propagation=do_propagation,
                    )
            self.global_tokenizer = TokenInitializer(dim,
                                                input_resolution,
                                                self.window_size,
                                                ct_size=ct_size)

        def forward(self, x):
            ct = self.global_tokenizer(x)
            B, C, H, W = x.shape
            input = window_partition(x, self.window_size)
            # output, ct_ = self.hat(input, ct)
            output, ct_ = self.hat(input, ct)
            output, ct_ = self.hat(output, ct_)
            output, ct_ = self.hat(output, ct_)
            # output, ct_ = self.hat(output, ct_)
            output = window_reverse(output, self.window_size, H, W, B)
            return output
