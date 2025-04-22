import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class TemporalShiftModule(nn.Module):
    def __init__(self, net, n_segment=10, n_div=8):
        """
        :param net: 要包装的子网络
        :param n_segment: 视频的帧数（这里设置为 10，对应输入时的帧数）
        :param n_div: 控制平移的通道数比例
        """
        super(TemporalShiftModule, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.n_div = n_div

    def forward(self, x):
        # 输入形状: (b * t, c, h, w)
        bt, c, h, w = x.size()
        b = bt // self.n_segment
        # 重塑为 (b, t, c, h, w)
        x = x.view(b, self.n_segment, c, h, w)

        fold = c // self.n_div  # 需要平移的通道数
        out = torch.zeros_like(x)
        # 向前平移：将后面帧的部分特征复制到前面
        out[:, :-1, :fold] = x[:, 1:, :fold]
        # 向后平移：将前面帧的部分特征复制到后面
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        # 剩余通道保持不变
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        # 恢复形状 (b*t, c, h, w)
        out = out.view(bt, c, h, w)
        return self.net(out)


class DiffusionLightweightTSMGenerator(nn.Module):
    def __init__(self, image_size=224, num_classes=4, dim=64, n_segment=10, timesteps=256):
        """
        参数说明：
          image_size: 图像尺寸（假设正方形）
          num_classes: 分类数
          dim: 空间特征提取器初始通道数
          n_segment: 输入视频序列的帧数（这里与实际输入帧数一致，如 10）
          timesteps: 扩散模型的总时间步数
        """
        super(DiffusionLightweightTSMGenerator, self).__init__()
        self.n_segment = n_segment  # 用于 TSM 的输入帧数
        self.timesteps = timesteps

        # 空间特征提取器（轻量化 CNN + TSM 模块）
        self.spatial_extractor = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1),  # (3,224,224) -> (dim,224,224)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (dim,112,112)
            TemporalShiftModule(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)  # (dim,56,56)
            ), n_segment=n_segment),
            TemporalShiftModule(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))  # (dim,1,1)
            ), n_segment=n_segment)
        )

        # 将空间特征从 dim 降至 16，与后续额外输入和时间嵌入匹配
        self.spatial_fc = nn.Linear(dim, 16)

        # 额外输入处理层（阶段嵌入）：将 (b, 10) 映射到 (b, 16)
        self.stage_embedding = nn.Embedding(10, 16)

        # 扩散模型的时间步嵌入层：将标量 t 映射到 16 维向量
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, 16)
        )

        # 输出层（保持与原模型一致）
        self.fc_out = nn.Linear(16, num_classes)
        self.fc_out2 = nn.Linear(16, 20)


    def forward(self, x, extra_input, target_seq_len=256, t=None):
        """
        输入:
          x: (b, input_seq, 3, 224, 224) - 图像序列，input_seq 为实际输入帧数（例如 10）
          extra_input: (b, 10) - 额外输入（例如阶段信息）
        输出:
          outputs: (b, target_seq_len, num_classes)
          output_label: (b, target_seq_len, 2, 10)
        说明:
          1. 内部会将实际输入的时间维度上升采样到 target_seq_len（默认为256）。
          2. 若 t 未传入，则在 [0, timesteps-1] 范围内随机采样。
        """
        batch_size, input_seq, c, h, w = x.size()
        # 保证 extra_input 为 float 类型
        extra_input = extra_input.view(extra_input.size(0), -1).to(torch.float32)

        # 若 t 未提供，则随机采样，形状 (b, 1)
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size, 1), device=x.device).float()
        # 计算时间嵌入，并扩展到目标时间步数
        t_emb = self.time_embed(t)  # (b, 16)
        t_emb = t_emb.unsqueeze(1).expand(batch_size, target_seq_len, 16)  # (b, target_seq_len, 16)

        # 提取每帧的空间特征：
        # 将输入转换为 (b*input_seq, 3, 224, 224)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        spatial_features = self.spatial_extractor(x)  # (b*input_seq, dim, 1, 1)
        spatial_features = spatial_features.view(batch_size, input_seq, -1)  # (b, input_seq, dim)
        spatial_features = self.spatial_fc(spatial_features)  # (b, input_seq, 16)

        # 处理额外输入：通过 argmax 得到索引，再通过嵌入映射到 16 维
        extra_features = self.stage_embedding(extra_input.argmax(dim=-1))  # (b, 16)
        extra_features = repeat(extra_features, 'b d -> b t d', t=input_seq)  # (b, input_seq, 16)
        gate = torch.sigmoid(extra_features)

        # 特征融合：空间特征乘以门控，再加上额外输入
        fused_features = spatial_features * gate + extra_features  # (b, input_seq, 16)

        # 由于实际输入帧数较少，将时序特征上采样到 target_seq_len
        # 转置后在时间维度上进行线性插值，然后转置回来
        fused_features = fused_features.transpose(1, 2)  # (b, 16, input_seq)
        fused_features_upsampled = F.interpolate(fused_features, size=target_seq_len, mode='linear',
                                                 align_corners=False)
        fused_features_upsampled = fused_features_upsampled.transpose(1, 2)  # (b, target_seq_len, 16)

        # 将上采样后的特征与时间步嵌入相加
        fused_features_final = fused_features_upsampled + t_emb  # (b, target_seq_len, 16)


        # 输出层分别作用于每个时间步
        outputs = self.fc_out(fused_features_final)  # (b, target_seq_len, num_classes)
        output_label = self.fc_out2(fused_features_final)  # (b, target_seq_len, 20)
        output_label = output_label.view(batch_size, target_seq_len, 2, 10)  # (b, target_seq_len, 2, 10)

        return outputs, output_label


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 构造模型时指定 n_segment=10，以匹配输入帧数（输入为 10 帧）
    generator = DiffusionLightweightTSMGenerator(n_segment=10).to(device)
    # 输入 shape 为 (1, 10, 3, 224, 224)
    input_tensor = torch.randn(1, 10, 3, 224, 224).to(device)
    # extra_input 的形状依然为 (1, 10)；这里代表额外信息（例如阶段信息）
    stage = torch.randn(1, 10).to(device)
    # forward 时 target_seq_len 指定为 256，保证输出形状与原来一致
    output, output_level = generator(input_tensor, stage, target_seq_len=256)

    print("Parameter count:", count_parameters(generator))
    print("Output shape:", output.shape)  # 期望 (1, 256, num_classes)
    print("Output level shape:", output_level.shape)  # 期望 (1, 256, 2, 10)
