import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import segmentation_models_pytorch as smp


class RecurrentBlock(nn.Module):
    def __init__(self, out_channels, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),  # padding=1 保持尺寸不变
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            else:
                x1 = self.conv(x + x1)
        return x1

class R2Block(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super(R2Block, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)  # padding=1 保持尺寸不变
        self.rcnn = nn.Sequential(
            RecurrentBlock(out_channels, t=t),
            RecurrentBlock(out_channels, t=t)
        )

    def forward(self, x):
        x = self.conv_in(x)
        return self.rcnn(x)

class R2Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, t=2):
        super(R2Unet, self).__init__()

        # 使用预训练的ResNet34作为编码器
        resnet = models.resnet34(pretrained=True)

        # 修改ResNet34的第一层，保持输入尺寸不变
        self.encoder0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),  # stride从2改为1
            resnet.bn1,
            resnet.relu
        )
        # 不再使用maxpool，避免尺寸减半

        self.encoder1 = resnet.layer1  # 64 通道
        self.encoder2 = resnet.layer2  # 128 通道
        self.encoder3 = resnet.layer3  # 256 通道
        self.encoder4 = resnet.layer4  # 512 通道

        # Bridge (Bottleneck)
        self.bridge = R2Block(512, 512, t=t)

        # 解码器部分
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = R2Block(512, 256, t=t)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = R2Block(256, 128, t=t)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = R2Block(128, 64, t=t)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.decoder1 = R2Block(128, 64, t=t)

        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器
        e0 = self.encoder0(x)  # 尺寸不变

        e1 = self.encoder1(e0)  # 尺寸不变
        e2 = self.encoder2(e1)  # 尺寸减半
        e3 = self.encoder3(e2)  # 尺寸再减半
        e4 = self.encoder4(e3)  # 尺寸再减半

        # Bridge
        b = self.bridge(e4)

        # 解码器
        d4 = self.up4(b)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)
        # print(d2.shape)
        d1 = self.up1(d2)
        # 由于我们没有使用 maxpool，e0 的尺寸与 d1 相同
        # print(d1.shape)
        # print(e0.shape)
        d1 = torch.cat([d1, e0], dim=1)
        d1 = self.decoder1(d1)

        # 输出层
        out = self.conv_final(d1)
        return out

# 测试模型
if __name__ == "__main__":
    model = R2Unet(in_channels=3, out_channels=16, t=2)
    pretrained_weights = torch.load(r'C:\Users\Charl\PycharmProjects\cataract_Seg\checkpoint_test\epoch_1000.pth', map_location=torch.device('cpu'))

    # 加载预训练权重
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    x = torch.randn((1, 3, 512, 512))
    y = model(x)
    print("预训练权重加载完成！")
    print("输入尺寸：", x.shape)
    print("输出尺寸：", y.shape)
