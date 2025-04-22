import torch
import torch.nn as nn


class TrajectoryLoss(nn.Module):
    def __init__(self, weight_tra=1.0, weight_level=1.0):
        """
        自定义轨迹损失函数。
        包括：
        1. 判别器输出的二分类损失 (Binary Cross-Entropy Loss)
        2. 轨迹阶段的多分类损失 (Cross-Entropy Loss)

        参数:
            weight_tra: 判别器损失的权重
            weight_level: 阶段分类损失的权重
        """
        super(TrajectoryLoss, self).__init__()
        self.weight_tra = weight_tra
        self.weight_level = weight_level
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')  # 用于判别器的输出
        self.bce_loss_se = nn.BCEWithLogitsLoss(reduction='mean')  # 用于阶段分类输出
        self.batch_norm = nn.BatchNorm1d(1)

    def forward(self, output_tra, level, target_tra, target_level, eps=1e-8):
        """
        参数:
            output_tra: 判别器输出，形状为 (B, 1)（未经过 Sigmoid）
            level:      阶段分类输出，形状为 (B, num_classes)
            target_tra: 判别器的真实标签 (B, 1) (0 或 1)
            target_level: 阶段分类的真实标签 (B,) (离散类别标签)

        返回:
            loss: 加权组合的总损失
        """
        # output_tra = self.batch_norm(output_tra)
        # output_tra = output_tra.clamp(-10, 10)
        # target_tra = target_tra.clamp(0, 1)
        # print(target_level)

        # 判别器的 Binary Cross-Entropy Loss
        loss_tra = self.bce_loss(output_tra, target_tra)

        # 阶段分类的 Cross-Entropy Loss
        loss_level = self.bce_loss_se(level, target_level.unsqueeze(-1).float()) + eps

        # 总损失：加权组合
        total_loss = self.weight_tra * loss_tra + self.weight_level * loss_level + eps
        # print(total_loss)
        return total_loss

def total_variation_loss(x):
    """
    计算时间维度上的总变分损失
    参数：
      x: Tensor, 形状为 [B, T, C, H, W]
    返回：
      tv_loss: 标量总变分损失
    """
    # 计算相邻帧之间的差异
    tv_loss = torch.mean(torch.abs(x[:, 1:] - x[:, :-1]))
    return tv_loss