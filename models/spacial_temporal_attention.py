# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from models.module import TimestepEmbedder, FrozenCLIPEmbedder, PatchEmbedder
from timm.models.vision_transformer import Mlp, PatchEmbed
import random


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def space_time_alternate(x, attn_dim_size):
    # space -> time : (B * F, T, D) -> (B * T, F, D), attn_dim_size = frame_size
    # time -> space : (B * T, F, D) -> (B * F, T, D), attn_dim_size = patch_size
    x = x.unsqueeze(dim=1).reshape(-1, attn_dim_size, *x.shape[1:])
    x = x.permute(0, 2, 1, 3).flatten(0, 1)
    return x


def label_space_to_time(c, frame_size, patch_size):
    # (B * F, D) -> (B * T, D)
    c = c.unsqueeze(dim=1).reshape(-1, frame_size, c.size(-1))[:, :1]
    c = c + torch.zeros(1, patch_size, 1).to(c.device)
    return c.flatten(0, 1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


#################################################################################
#                                 Core DiT Model                                #
#################################################################################
class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.1,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        """
            x: (B, L, D)
            y: (B, M, D)
        """
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(B, y.size(1), 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CondBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        # TODO: modify 768
        self.y_mlp = nn.Linear(768, hidden_size)
        self.y_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.y_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.x_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, c, y):
        """
            x: (B, L, D)
            y: (B, M, D)
        """
        y = self.y_norm(self.y_mlp(y))
        shift_y, scale_y, gate_y, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        x = x + gate_y.unsqueeze(1) * self.y_attn(modulate(self.x_norm(x), shift_y, scale_y), y)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class VideoDiTBlock(nn.Module):
    def __init__(self, hidden_size, frame_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.frame_size = frame_size
        self.spacial_block = CondBlock(hidden_size, num_heads, mlp_ratio, **block_kwargs)

        self.temporal_block = Block(hidden_size, num_heads, mlp_ratio, **block_kwargs)

    def forward(self, x, c, y):
        # x: (N, T, D), c: (N, D), where N = B * F
        patch_size = x.size(1)
        x = self.spacial_block(x, c, y)
        if self.frame_size != 1:
            x = space_time_alternate(x, self.frame_size)
            c_time = label_space_to_time(c, self.frame_size, patch_size)
            x = self.temporal_block(x, c_time)
            x = space_time_alternate(x, patch_size)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class STAttentionVideoDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone for video.
    """
    def __init__(
        self,
        input_size=32,
        frame_size=4,           # fixed frame size for each video
        patch_size=2,
        frame_patch_size=1,     # patch embedding size for frame dimension
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cond_dropout_prob=0.1,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        # ADD, parameters for frame
        self.frame_size = frame_size
        self.frame_patch_size = frame_patch_size
        self.cond_dropout_prob = cond_dropout_prob

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size,
                                        bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = FrozenCLIPEmbedder()
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        # ADD, positional embedding for frame dimension
        self.frame_embed = nn.Parameter(torch.zeros(1, frame_size, hidden_size), requires_grad=False)
        self.frame_mlp = nn.Linear(hidden_size, hidden_size)

        self.blocks = nn.ModuleList([
            VideoDiTBlock(hidden_size, frame_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def dropout(self, y):
        for i in range(len(y)):
            if random.random() < self.cond_dropout_prob:
                y[i] = ""

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # ADD, zero-out frame positional encoding
        frame_embed = get_1d_sincos_pos_embed(self.frame_embed.shape[-1], self.frame_size)
        self.frame_embed.data.copy_(torch.from_numpy(frame_embed).float().unsqueeze(0))
        nn.init.constant_(self.frame_mlp.weight, 0)
        nn.init.constant_(self.frame_mlp.bias, 0)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.spacial_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.spacial_block.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(block.temporal_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.temporal_block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (B * F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B, ) tensor of diffusion timesteps
        y: (B, ) list of language
        """
        x = self.x_embedder(x) + self.pos_embed
        # frame positional encoding
        x = x.unsqueeze(dim=1).reshape(-1, self.frame_size, *x.shape[1:])  # (B, F, T, D)
        x = (x + self.frame_mlp(self.frame_embed).unsqueeze(dim=2)).flatten(0, 1)  # (N, T, D)

        t = self.t_embedder(t)                   # (N, D)
        self.dropout(y)
        y = self.y_embedder(y)                   # (N, M, D)

        for block in self.blocks:
            x = block(x, t, y)                      # (N, T, D)
        x = self.final_layer(x, t)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size):
    grid = np.arange(grid_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return STAttentionVideoDiT(depth=28, frame_size=16, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return STAttentionVideoDiT(depth=24, frame_size=16, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_B_4(**kwargs):
    return STAttentionVideoDiT(depth=12, frame_size=8, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


if __name__ == '__main__':
    model = DiT_L_4().to('cuda')
    x = torch.randn(32, 4, 32, 32).cuda()
    y = ['a boy'] * 32
    t = torch.randint(0, 10, size=(32,)).cuda()
    model.forward(x, t, y)
