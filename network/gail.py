import torch
import torch.nn as nn
from einops import rearrange, repeat

class TemporalShiftModule(nn.Module):
    def __init__(self, net, n_segment=8, n_div=8):
        super(TemporalShiftModule, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.n_div = n_div

    def forward(self, x):
        # Input shape: (b * t, c, h, w)
        bt, c, h, w = x.size()
        b = bt // self.n_segment
        x = x.view(b, self.n_segment, c, h, w)

        # Shift a portion of the channels along the temporal dimension
        fold = c // self.n_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # Shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # Shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # No shift

        # Reshape back to (b * t, c, h, w)
        out = out.view(bt, c, h, w)

        # Pass through the original network
        return self.net(out)

class LightweightTSMGenerator(nn.Module):
    def __init__(self, image_size=224, num_classes=4, dim=64, n_segment=8):
        super(LightweightTSMGenerator, self).__init__()

        # 空间特征提取器（轻量化 CNN）
        self.spatial_extractor = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1),  # (3, 224, 224) -> (dim, 224, 224)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (dim, 112, 112)
            TemporalShiftModule(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)  # (dim, 56, 56)
            ), n_segment=n_segment),
            TemporalShiftModule(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))  # (dim, 1, 1)
            ), n_segment=n_segment)
        )

        # 额外的输入处理层
        self.extra_input_fc = nn.Sequential(
            nn.Linear(10, dim),  # 将额外输入映射到与空间特征相同的维度
            nn.ReLU()
        )

        # 输出层
        self.fc_out = nn.Linear(16, num_classes)

        self.fc_out2 = nn.Linear(16, 20)
        # self.temporal_lstm = nn.LSTM(dim, dim, num_layers=1, batch_first=True, bidirectional=True)
        self.stage_embedding = nn.Embedding(10, 16)

        # TSM 参数
        self.n_segment = n_segment

    def forward(self, x, extra_input, seq_len=256):
        # Input shapes:
        # x: (b, 256, 3, 224, 224) - 图像序列
        # extra_input: (b, 10) - 额外输入
        extra_input = extra_input.view(extra_input.size(0), -1).to(torch.float32)
        # print(extra_input.size())

        batch_size, _, c, h, w = x.size()

        # 提取每帧的空间特征
        x = rearrange(x, 'b t c h w -> (b t) c h w')  # (b * 256, 3, 224, 224)
        spatial_features = self.spatial_extractor(x)  # (b * 256, dim, 1, 1)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)  # (b, 256, dim)

        extra_features = self.stage_embedding(extra_input.argmax(dim=-1))  # (b, dim)
        # print(extra_features.shape)
        # extra_features = self.extra_input_fc(extra_input)  # (b, dim)
        extra_features = repeat(extra_features, 'b d -> b t d', t=seq_len)  # (b, 256, dim)
        gate = torch.sigmoid(extra_features)
        # 特征融合

        fused_features = spatial_features * gate + extra_features
        # fused_features = spatial_features + extra_features  # (b, 256, dim)
        # fused_features, _ = self.temporal_lstm(fused_features)  # (b, seq_len, 2*dim)

        # 输出层
        # print(fused_features.shape)
        outputs = self.fc_out(fused_features)  # (b, 256, num_classes)

        output_label = self.fc_out2(fused_features)
        output_label = output_label.view(batch_size, seq_len, 2, 10)

        return outputs, output_label # output -> (b, seq_len, 4), output_label1 -> (b, seq_len, 2, 10)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        # 共享的特征提取层
        self.label_input_layers = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),  # 1D卷积，保持时间步数
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten()  # 展平以匹配全连接层输入
        )

        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 256),  # 减少神经元数量
            nn.LeakyReLU(0.2),          # 使用 LeakyReLU
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )

        # 判别器输出（真实/生成的概率）
        self.logit = nn.Sequential(
            nn.Linear(64, 32),          # 减少神经元数量
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),           # 输出一个标量
            # nn.Sigmoid()              # 如果需要概率，可以取消注释
        )

        # 辅助任务输出（阶段预测）
        self.level_pred = nn.Sequential(
            nn.Linear(64, 32),          # 减少神经元数量
            nn.LeakyReLU(0.2),
            nn.Linear(32, 10),           # 输出 2 个类别
        )

    def forward(self, actions, stage, labels):
        # 将输入展平并拼接
        actions = actions.view(actions.size(0), -1)
        x = torch.cat([actions, stage], dim=1).to(torch.float32)

        new_features = self.label_input_layers(labels)

        x = torch.cat([x, new_features], dim=1)

        # 共享特征提取
        x = self.shared_layers(x)

        # 判别器输出
        logit_output = self.logit(x)

        # 辅助任务输出
        level_output = self.level_pred(x)

        return logit_output, level_output



# 测试生成器
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    generator = LightweightTSMGenerator().to(device)
    # discriminator = Discriminator(1034+64*20).to(device)
    input_tensor = torch.randn(1, 256, 3, 224, 224).to(device)  # Batch size = 8
    # input_tensor = torch.randn(1, 256, 4).to(device)  # Batch size = 8
    stage = torch.randn(1, 10).to(device)
    label = torch.randn(1, 256, 2, 10).to(device)
    output, output_level = generator(input_tensor, stage)
    # output, output_level = discriminator(input_tensor, stage, label)

    print(count_parameters(generator))
    # print(count_parameters(discriminator))
    # print("Output shape:", output.shape)  # Expected: (8, 256, 4)
    print("Output shape:", output.shape, "level_shape", output_level.shape)  # Expected: (8, 256, 4)