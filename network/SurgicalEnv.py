import torch
import torch.nn as nn
import torch.nn.functional as F
class SurgicalEnvironment:
    """
    环境:
      state = (视频特征, 当前 tips_kpts, 手术阶段)
      action = (dx1, dy1, dx2, dy2)
    """

    def __init__(self, video, tips_kpts, surgery_stage, label, device):
        """
        初始化环境。

        参数:
        - video: 视频数据，形状为 [seq_length, C, H, W]
        - tips_kpts: tips 的关键点位置，形状为 [seq_length, 4]
        - surgery_stage: 手术阶段，标量或列表，长度为 seq_length
        - feature_extractor: 特征提取模型 (如 CNN, ResNet 等)
        """
        seq_length = video.shape[0]
        self.video = video.unsqueeze(0)  # 增加 batch_size 维度，变为 [1, seq_length, C, H, W]
        self.tips_kpts = tips_kpts.unsqueeze(0)  # [1, seq_length, 4]
        self.surgery_stage = surgery_stage.unsqueeze(0)  # 增加 batch_size 维度
        self.labels = label.unsqueeze(0)  # [1, seq_length]

        self.batch_size = 1  # 视频只有一个样本
        self.T = seq_length  # 时间帧数
        self.current_step = torch.zeros(self.batch_size, dtype=torch.long)  # 当前步数
        self.done = torch.zeros(self.batch_size, dtype=torch.bool)  # 是否完成
        self.device = device

    def reset(self):
        """
        重置环境，返回初始状态。

        返回:
        - 初始状态，包括第 0 帧的特征、tips_kpts 和手术阶段信息
        """
        self.current_step = torch.zeros(self.batch_size, dtype=torch.long)
        self.done = torch.zeros(self.batch_size, dtype=torch.bool)
        return self._get_state()

    def _get_state(self):
        """
        获取当前状态。

        返回:
        - 当前状态: (视频帧特征, 当前 tips_kpts, 手术阶段, 当前标签)
        """
        frame_indices = self.current_step  # [batch_size]
        batch_indices = torch.arange(self.batch_size)  # [batch_size]

        # 当前帧图像，形状为 [batch_size, C, H, W]
        frame_imgs = self.video[batch_indices, frame_indices]


        # 当前帧的 tips_kpts，形状为 [batch_size, 4]
        current_tips = self.tips_kpts[batch_indices, frame_indices]

        current_label = self.labels[batch_indices, frame_indices]

        # 手术阶段，形状为 [batch_size]
        stages = self.surgery_stage

        next_frame_indices = torch.clamp(frame_indices + 1, max=self.video.shape[1] - 1)
        next_tips = self.tips_kpts[batch_indices, next_frame_indices]  # [batch_size, 4]
        next_label = self.labels[batch_indices, next_frame_indices]  # [batch_size]

        return frame_imgs, current_tips, stages, current_label, next_tips, next_label


    def calculate_reward(self, predicted_tips, predicted_label, true_next_tips, true_next_label):
        """
        计算奖励函数。

        参数:
        - predicted_tips: 模型预测的 tips_kpts，形状为 [batch_size, 4]
        - predicted_label: 模型预测的标签，形状为 [batch_size]

        返回:
        - reward: 计算出的奖励值
        """

        # 位置误差（L2 距离）
        # print(f'{predicted_tips.shape}:::::{predicted_tips}')
        # print(f'{predicted_label.shape}:::::{true_next_label}')
        position_error = torch.norm(predicted_tips - true_next_tips, dim=1)  # [batch_size]

        # 标签误差（分类准确性）
        label_accuracy = F.cross_entropy(true_next_label, predicted_label)  # [batch_size]

        # 奖励设计：位置误差的负值 + 标签准确性
        reward = -position_error + label_accuracy

        return reward.mean().item()  # 返回标量奖励

    def step(self, action, predicted_label, true_next_tips, true_next_label):
        batch_indices = torch.arange(self.batch_size, device=self.device)
        frame_indices = self.current_step

        # 更新 tips_kpts
        self.tips_kpts[batch_indices, frame_indices] += action

        # 更新步数
        self.current_step += 1

        # 更新完成标志
        self.done = self.current_step >= self.T

        # 获取下一状态或返回 None
        if not torch.all(self.done):
            next_state = self._get_state()
        else:
            next_state = None

        # 计算奖励（示例）
        reward = self.calculate_reward(action, predicted_label, true_next_tips, true_next_label)

        return next_state, reward, torch.all(self.done)

