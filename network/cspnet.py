import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：Depthwise + Pointwise"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    """轻量级卷积块，使用深度可分离卷积"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, stride=stride)

    def forward(self, x):
        return self.conv(x)


class CSPBlock(nn.Module):
    """轻量级 CSPNet 核心块，使用深度可分离卷积"""
    def __init__(self, in_channels, out_channels, num_blocks, stride=1):
        super(CSPBlock, self).__init__()
        reduced_channels = out_channels // 2  # 通道减半
        self.left_path = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )

        self.right_path = nn.Sequential(
            *[BasicBlock(reduced_channels, reduced_channels, stride=1) for _ in range(num_blocks)]
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        left = self.left_path(x)
        right = self.right_path(left)
        out = torch.cat([left, right], dim=1)
        return self.fusion(out)


class CSPNet(nn.Module):
    """轻量级 CSPNet 实现，使用深度可分离卷积"""
    def __init__(self, in_channels=3, num_classes=1000):
        super(CSPNet, self).__init__()
        self.stem = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 16, stride=2)
        )

        self.csp_layers = nn.Sequential(
            CSPBlock(16, 32, num_blocks=1),  # 第一层 CSP
            CSPBlock(32, 64, num_blocks=1, stride=2),  # 第二层 CSP
            CSPBlock(64, 128, num_blocks=1, stride=2),  # 第三层 CSP
            # CSPBlock(128, 256, num_blocks=2, stride=2),  # 第四层 CSP
            CSPBlock(128, 512, num_blocks=2, stride=2),  # 第五层 CSP
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.csp_layers(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 测试网络
if __name__ == "__main__":
    model = CSPNet(in_channels=3, num_classes=128)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print("Output shape:", output.shape)
    print(f"Number of parameters: {count_parameters(model)}")
