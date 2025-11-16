import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


from CEM import ConvEnhancerModule


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock_MSWA(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=[3, 7, 11], shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_sizes = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        for i, ws in enumerate(self.window_sizes):
            if min(self.input_resolution) <= ws:
                self.window_sizes[i] = min(self.input_resolution)
        if self.shift_size > 0:
            smallest_ws = min(self.window_sizes) if self.window_sizes else 1
            if smallest_ws <= self.shift_size:
                self.shift_size = smallest_ws // 2

        self.norm1 = norm_layer(dim)

        self.num_scales = len(self.window_sizes)
        assert self.num_scales > 0, "window_sizes list cannot be empty."
        assert num_heads % self.num_scales == 0, f"num_heads {num_heads} must be divisible by num_scales {self.num_scales}"
        self.heads_per_scale = num_heads // self.num_scales

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        self.relative_position_bias_tables = nn.ParameterList()
        self.relative_position_indices = {}
        for ws in self.window_sizes:
            ws_tuple = to_2tuple(ws)
            table = nn.Parameter(torch.zeros((2 * ws_tuple[0] - 1) * (2 * ws_tuple[1] - 1), self.heads_per_scale))
            trunc_normal_(table, std=.02)
            self.relative_position_bias_tables.append(table)

            coords_h = torch.arange(ws_tuple[0])
            coords_w = torch.arange(ws_tuple[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += ws_tuple[0] - 1
            relative_coords[:, :, 1] += ws_tuple[1] - 1
            relative_coords[:, :, 0] *= 2 * ws_tuple[1] - 1
            index = relative_coords.sum(-1)
            self.register_buffer(f"rpi_ws{ws}", index)
            self.relative_position_indices[ws] = f"rpi_ws{ws}"

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.fusion_proj = nn.Linear(dim * self.num_scales, dim)
        self.proj_drop = nn.Dropout(drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.conv_enhancer = ConvEnhancerModule(dim=dim, kernel_size=3, norm_layer=norm_layer)

        if self.shift_size > 0:
            self.attn_masks = self.create_attn_masks()
        else:
            self.attn_masks = None

    def create_attn_masks(self):
        H, W = self.input_resolution
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        masks = {}
        for ws in self.window_sizes:
            pad_b = (ws - H % ws) % ws
            pad_r = (W - W % ws) % ws
            padded_mask = F.pad(img_mask, (0, 0, 0, pad_r, 0, pad_b))
            mask_windows = window_partition(padded_mask, ws).view(-1, ws * ws)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            masks[ws] = attn_mask
        return masks

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut1 = x

        normed_x = self.norm1(x)
        qkv = self.qkv(normed_x).reshape(B, L, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.shift_size > 0:
            q = torch.roll(q.view(B, H, W, C), shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)).view(B, L, C)
            k = torch.roll(k.view(B, H, W, C), shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)).view(B, L, C)
            v = torch.roll(v.view(B, H, W, C), shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)).view(B, L, C)

        scale_outputs = []
        head_dim = C // self.num_heads

        for i, ws in enumerate(self.window_sizes):
            q_i, k_i, v_i = q, k, v

            pad_b = (ws - H % ws) % ws
            pad_r = (ws - W % ws) % ws
            q_i = F.pad(q_i.view(B, H, W, C), (0, 0, 0, pad_r, 0, pad_b))
            k_i = F.pad(k_i.view(B, H, W, C), (0, 0, 0, pad_r, 0, pad_b))
            v_i = F.pad(v_i.view(B, H, W, C), (0, 0, 0, pad_r, 0, pad_b))

            H_pad, W_pad = H + pad_b, W + pad_r

            q_windows = window_partition(q_i, ws).view(-1, ws * ws, self.heads_per_scale, head_dim).permute(0, 2, 1, 3)
            k_windows = window_partition(k_i, ws).view(-1, ws * ws, self.heads_per_scale, head_dim).permute(0, 2, 1, 3)
            v_windows = window_partition(v_i, ws).view(-1, ws * ws, self.heads_per_scale, head_dim).permute(0, 2, 1, 3)

            attn = (q_windows @ k_windows.transpose(-2, -1)) * self.scale

            bias_table = self.relative_position_bias_tables[i]
            relative_position_index = getattr(self, self.relative_position_indices[ws])
            relative_position_bias = bias_table[relative_position_index.view(-1)].view(
                ws * ws, ws * ws, -1).permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

            if self.attn_masks:
                attn = attn + self.attn_masks[ws].to(x.device).unsqueeze(1)

            attn = self.softmax(attn)
            attn = self.attn_drop(attn)

            attn_windows = (attn @ v_windows).transpose(1, 2).reshape(-1, ws * ws, self.heads_per_scale * head_dim)

            scale_x = window_reverse(attn_windows, ws, H_pad, W_pad)
            scale_x = scale_x[:, :H, :W, :].contiguous().view(B, L, C)
            scale_outputs.append(scale_x)

        x_attn = torch.cat(scale_outputs, dim=-1)
        x_attn = self.fusion_proj(x_attn)
        x_attn = self.proj_drop(x_attn)

        if self.shift_size > 0:
            x_attn = torch.roll(x_attn.view(B, H, W, C), shifts=(self.shift_size, self.shift_size), dims=(1, 2)).view(B,
                                                                                                                      L,
                                                                                                                      C)

        x = shortcut1 + self.drop_path(x_attn)

        normed_x_for_ffn = self.norm2(x)
        x_mlp = self.mlp(normed_x_for_ffn)

        x_enhanced = self.conv_enhancer(x_mlp, H, W)

        x = x + self.drop_path(x_enhanced)

        return x