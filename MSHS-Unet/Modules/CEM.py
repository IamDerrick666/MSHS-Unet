import torch
import torch.nn as nn


class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcite, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, reduced_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_transposed = x.transpose(1, 2)
        channel_weights = self.se(x_transposed)
        return x * channel_weights.transpose(1, 2)


class ConvEnhancerModule(nn.Module):
    def __init__(self, dim, kernel_size=3, reduction_ratio=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        padding = kernel_size // 2

        self.dw_conv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
            bias=False
        )

        self.norm = norm_layer(dim)
        self.act = nn.GELU()

        self.se_attention = SqueezeExcite(in_channels=dim, reduced_dim=dim // reduction_ratio)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        assert C == self.dim, "Input channel dimension has wrong size"

        shortcut = x

        x_image = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x_conv = self.dw_conv(x_image)

        x_conv_tokens = x_conv.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x_normed = self.norm(x_conv_tokens)

        x_attended = self.se_attention(x_normed)

        return shortcut + x_attended