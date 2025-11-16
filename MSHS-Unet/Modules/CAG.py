import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionGate(nn.Module):
    def __init__(self, dim_x, dim_g, num_heads=8, qkv_bias=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_x // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim_g, dim_x, bias=qkv_bias)
        self.k_proj = nn.Linear(dim_x, dim_x, bias=qkv_bias)
        self.v_proj = nn.Linear(dim_x, dim_x, bias=qkv_bias)

        self.proj = nn.Linear(dim_x, dim_x)

        # Normalization layers
        self.norm_x = norm_layer(dim_x)
        self.norm_g = norm_layer(dim_g)

    def forward(self, x, g):
        B, L_x, C_x = x.shape
        B, L_g, C_g = g.shape
        assert L_x == L_g, f"Token lengths of encoder feature ({L_x}) and decoder feature ({L_g}) must be the same."

        x_norm = self.norm_x(x)
        g_norm = self.norm_g(g)

        q = self.q_proj(g_norm).reshape(B, L_g, self.num_heads, C_x // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(x_norm).reshape(B, L_x, self.num_heads, C_x // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(x_norm).reshape(B, L_x, self.num_heads, C_x // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        gated_x = (attn @ v).transpose(1, 2).reshape(B, L_g, C_x)

        gated_x = self.proj(gated_x)

        return x + gated_x