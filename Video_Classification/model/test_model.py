import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchinfo import summary


class ResNet50TemporalModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNet50TemporalModel, self).__init__()

        # ResNet50作为特征提取器
        self.resnet = resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Identity()  # 去掉ResNet50的全连接层，保留特征

        # 时序模型：LSTM
        self.rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)

        # 分类头
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        输入: x, 形状 (N, T, H, W, C)
        输出: 分类结果 (N, Classes)
        """
        N, T, H, W, C = x.shape

        # 调整输入形状适配ResNet (N*T, C, H, W)
        x = x.view(-1, H, W, C).permute(0, 3, 1, 2)  # 转换为 (N*T, C, H, W)
        features = self.resnet(x)  # ResNet50提取特征 (N*T, 2048)

        # 恢复时间维度 (N, T, 2048)
        features = features.view(N, T, -1)

        # LSTM处理时序特征 (N, T, H_out)
        _, (hidden, _) = self.rnn(features)  # 只取最终的隐藏状态

        # 取最后一层隐藏状态作为分类输入 (N, H_out)
        x = hidden[-1]

        # 分类
        x = self.fc(x)  # 输出 (N, Classes)
        return x


if __name__ == '__main__':
    model = ResNet50TemporalModel(num_classes=10, pretrained=True)  # 分类任务，10个类别
    input_tensor = torch.randn(8, 32, 224, 224, 3)  # 输入 (N=8, T=16, H=224, W=224, C=3)
    output = model(input_tensor)  # 输出 (N=8, Classes=10)
    print(output.shape)
    summary(model, input_size=(3, 32, 224, 224, 3), device='cuda')
