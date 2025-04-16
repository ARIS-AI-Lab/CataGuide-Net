import torch.nn as nn
import math
from torchinfo import summary
import torch.nn.functional as F
from torchvision import models
import torch

from config import params


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, 1, kernel_size=1)  # 输出1个通道的空间注意力图
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv1(x)  # 3x3卷积提取特征
        attention = self.conv2(attention)  # 生成单通道注意力图
        attention = self.sigmoid(attention)  # 将注意力范围映射到 [0, 1]
        return x * attention  # 根据注意力图对输入特征进行加权


class SpatialAttentionUpsample(nn.Module):
    def __init__(self, channels, scale_factor=2, mode='bilinear', align_corners=True):
        super(SpatialAttentionUpsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)
        self.spatial_attention = SpatialAttention(channels)

    def forward(self, x):
        # 上采样操作
        x_upsampled = self.upsample(x)

        # 空间注意力
        out = self.spatial_attention(x_upsampled)
        return out


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def conv_up(inp, oup):
    return nn.Sequential(
        conv_dw(inp, oup, 1),
        nn.ConvTranspose2d(oup, oup, kernel_size=4,
                           stride=2, padding=1, output_padding=0),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def conv_up_complex(inp, oup):
    """增加上采样模块的复杂度"""
    return nn.Sequential(
        conv_dw(inp, oup, 1),  # 深度可分离卷积
        nn.ConvTranspose2d(oup, oup, kernel_size=4, stride=2, padding=1, output_padding=0),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
        nn.Conv2d(oup, oup, kernel_size=3, padding=1),  # 增加额外卷积层
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
        nn.Conv2d(oup, oup, kernel_size=3, padding=1),  # 增加额外卷积层
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
        SEBlock(oup),  # 加入SE通道注意力模块
    )


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)  # 全局池化
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y  # 加权输出


class LWANet(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super(LWANet, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.layer1 = mobilenet.features[0]
        self.layer2 = mobilenet.features[1]
        self.layer3 = nn.Sequential(
            mobilenet.features[2],
            mobilenet.features[3], )
        self.layer4 = nn.Sequential(
            mobilenet.features[4],
            mobilenet.features[5],
            mobilenet.features[6], )
        self.layer5 = nn.Sequential(
            mobilenet.features[7],
            mobilenet.features[8],
            mobilenet.features[9],
            mobilenet.features[10], )
        self.layer6 = nn.Sequential(
            mobilenet.features[11],
            mobilenet.features[12],
            mobilenet.features[13], )
        self.layer7 = nn.Sequential(
            mobilenet.features[14],
            mobilenet.features[15],
            mobilenet.features[16], )
        self.layer8 = nn.Sequential(
            mobilenet.features[17],
        )
        self.additional_branch = nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=1),  # 1x1卷积，将通道从320降到128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),  # Depthwise卷积
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 64, kernel_size=1),  # Pointwise卷积，将通道从128降到64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),  # Depthwise卷积
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 48, kernel_size=1),  # Pointwise卷积，将通道从64降到48
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48),  # Depthwise卷积
            nn.BatchNorm2d(48),
            nn.Dropout2d(0.1),
            nn.Conv2d(48, 32, kernel_size=1),  # Pointwise卷积，将通道从48降到32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),  # Depthwise卷积
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 32, kernel_size=1),  # Pointwise卷积，将通道从32降到24
            nn.ReLU(inplace=True)
        )

        self.landmark_coords_branch = nn.Sequential(
            nn.Conv2d(32, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, params['model_param']['max_kpts'] * 2, kernel_size=1),  # 每个关键点有 (x, y) 坐标
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Tanh(),
        )

        # 关键点类别预测分支
        self.landmark_class_branch = nn.Sequential(
            nn.Flatten(),  # 将卷积输出展平成一维
            nn.Linear(32 * 20 * 20, 128),  # 假设卷积输出为 32 x h x w，需要根据实际 h 和 w 调整
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, params['model_param']['max_kpts'] * 16)  # 最终输出的大小应与目标匹配
        )
        self.relu = nn.LeakyReLU()

        # self.landmark_class_branch = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, params['model_param']['max_kpts'] * 16, kernel_size=1),  # 每个关键点有类别预测
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     # nn.ReLU(inplace=True),
        # )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)  # x / 2
        x1 = self.layer3(x)  # 24, x / 4
        x2 = self.layer4(x1)  # 32, x / 8
        l5 = self.layer5(x2)  # 64, x / 16
        x3 = self.layer6(l5)  # 96, x / 16
        l7 = self.layer7(x3)  # 160, x / 32
        x4 = self.layer8(l7)  # 320, x / 32
        # print(x4.shape)

        pre_landmark_out = self.additional_branch(x4)
        # print(pre_landmark_out.shape)

        landmark_coords = self.landmark_coords_branch(pre_landmark_out)  # (batch, num_landmarks * 2, H, W)
        # print(landmark_coords.shape)
        landmark_coords = landmark_coords.view(landmark_coords.size(0), -1, 2)  # (batch, num_landmarks, 2)
        # landmark_coords = self.relu(landmark_coords)

        # 关键点类别预测
        landmark_classes = self.landmark_class_branch(
            pre_landmark_out)
        landmark_classes = landmark_classes.view(landmark_classes.size(0), params['model_param']['max_kpts'],
                                                 -1)  # (batch, num_landmarks, landmark_classes)
        return landmark_coords, landmark_classes


class AFB(nn.Module):
    def __init__(self, mid_ch, out_ch):
        super(AFB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid())

    def forward(self, input_high, input_low):
        mid_high = self.global_pooling(input_high)
        weight_high = self.conv1(mid_high)
        mid_low = self.global_pooling(input_low)
        weight_low = self.conv2(mid_low)

        return input_high.mul(weight_high) + input_low.mul(weight_low)


if __name__ == "__main__":
    model = LWANet(num_classes=16)
    input_tensor = torch.randn(3, 3, 640, 640)
    # seg_output, landmark_output, landmark_class_output, edge_output = model(input_tensor)
    landmark_output, landmark_class_output  = model(input_tensor)
    # summary(model, input_size=(1, 3, 640, 640), device='cpu')
    # print("Segmentation output shape:", seg_output.shape)  # Expected: (3, 16, 1/4, 1/4)
    print("Landmark output shape:", landmark_output.shape)  # Expected: (3, 4, 2)
    print("Landmark class output shape:", landmark_class_output.shape)  # Expected: (3, 4, 15)
    # print("Landmark edge output shape:", edge_output.shape)  # Expected: (3, 4, 15)