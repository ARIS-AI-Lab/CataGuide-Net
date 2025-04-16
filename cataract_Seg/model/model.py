import torch
import torch.nn as nn
import torch.nn.functional as F


# Dense Block定义
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DoubleConv(in_channels + i * growth_rate, growth_rate))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)


# CSPNet Block定义
class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, split_ratio=0.5):
        super(CSPBlock, self).__init__()
        split_channels = int(out_channels * split_ratio)
        self.conv1 = DoubleConv(in_channels, split_channels)
        self.conv2 = DoubleConv(in_channels, split_channels)
        self.conv3 = DoubleConv(split_channels * 2, out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return out


# DoubleConv定义
class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# 混合卷积的定义
class MixedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5], stride=1, padding=1):
        super(MixedConv, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // len(kernel_sizes), kernel_size=ks, stride=stride, padding=ks // 2)
            for ks in kernel_sizes
        ])

    def forward(self, x):
        return torch.cat([conv(x) for conv in self.convs], dim=1)


# SEBlock (自注意力机制)
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class BAM(nn.Module):
    def __init__(self, in_channels, reduction=16, dilation=4):
        super(BAM, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=3, padding=dilation,
                      dilation=dilation),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)
        attention = self.sigmoid(ca + sa)
        return x * attention


# SMUnet 网络架构定义
class SMUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SMUnet, self).__init__()

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), CSPBlock(64, 128), BAM(128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DenseBlock(128, growth_rate=32, num_layers=4), BAM(256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), CSPBlock(256, 512), BAM(512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DenseBlock(512, growth_rate=64, num_layers=4), BAM(768))

        # Upsampling path (corresponding to 4 downsampling layers)
        self.up1 = nn.ConvTranspose2d(768, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        # Output layer with multiple labels
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Downsample path
        x1 = self.inc(x)  # 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256
        x4 = self.down3(x3)  # 512
        x5 = self.down4(x4)  # 768

        # Upsample path
        x = self.up1(x5)
        x = self.conv_up1(torch.cat([x, x4], dim=1))
        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.conv_up3(torch.cat([x, x2], dim=1))
        x = self.up4(x)
        x = self.conv_up4(torch.cat([x, x1], dim=1))

        # Multiple labels output with softmax activation
        x = self.outc(x)
        return torch.softmax(x, dim=1)


# Example usage
if __name__ == '__main__':
    test_tensor = torch.randn(1, 3, 640, 640)
    model = SMUnet(in_channels=3, out_channels=16)
    out = model(test_tensor)
    print(out.shape)
