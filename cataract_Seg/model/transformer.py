import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MobileUnet(nn.Module):
    def __init__(self, num_classes=16):
        super(MobileUnet, self).__init__()

        # Encoder
        self.encoder1 = DepthwiseSeparableConv(3, 32, kernel_size=7, padding=3)  # Larger receptive field
        self.encoder2 = DepthwiseSeparableConv(32, 64, kernel_size=5, padding=2)  # Larger receptive field
        self.encoder3 = DepthwiseSeparableConv(64, 128)
        self.encoder4 = DepthwiseSeparableConv(128, 256)
        self.encoder5 = DepthwiseSeparableConv(256, 512)

        # Bottleneck
        self.bottleneck = DepthwiseSeparableConv(512, 1024)

        # Decoder
        self.decoder5 = DepthwiseSeparableConv(1024 + 512, 512)
        self.decoder4 = DepthwiseSeparableConv(512 + 256, 256)
        self.decoder3 = DepthwiseSeparableConv(256 + 128, 128)
        self.decoder2 = DepthwiseSeparableConv(128 + 64, 64)
        self.decoder1 = DepthwiseSeparableConv(64 + 32, 32)

        # Final Convolution
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        # Max Pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Upsample
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool(e3)
        e4 = self.encoder4(p3)
        p4 = self.pool(e4)
        e5 = self.encoder5(p4)
        p5 = self.pool(e5)

        # Bottleneck
        b = self.bottleneck(p5)

        # Decoder
        d5 = self.up(b)
        d5 = torch.cat([d5, e5], dim=1)
        d5 = self.decoder5(d5)

        d4 = self.up(d5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.up(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.up(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.up(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)

        # Final Convolution
        out = self.final_conv(d1)

        return out


if __name__ == "__main__":
    model = MobileUnet(num_classes=16)
    x = torch.randn((1, 3, 512, 512))
    y = model(x)
    print(y.shape)  # Should output torch.Size([1, 16, 512, 512])
