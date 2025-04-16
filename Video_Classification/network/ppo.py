import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from numpy.array_api import float32
# from main import get_writer
# from network.light_vit import mobilevit_xxs
from network.cspnet import CSPNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# writer = get_writer()
# PPO 算法类
class PPOActorCritic(nn.Module):
    def __init__(self, feature_dim, action_dim, label_dim):
        super(PPOActorCritic, self).__init__()
        self.feature_extractor = CSPNet(in_channels=3, num_classes=128).to(device)
        # self.feature_extractor = mobilevit_xxs(output_dim=128).to(device)
        self.actor = nn.Sequential(
            nn.Linear(feature_dim + 4 + 10 + 20, 64),  # 图像特征 + tips_kpts (4维) + 手术阶段 (1维)
            nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            nn.Linear(64, action_dim),  # 动作维度
            nn.Softmax()  # 归一化动作输出
        ).to(device)
        self.critic = nn.Sequential(
            nn.Linear(feature_dim + 4 + 10 + 20, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 状态值
        ).to(device)

        self.label_predictor = nn.Sequential(
            nn.Linear(feature_dim + 4 + 10 + 20, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, label_dim),  # 分类类别数
            nn.Softmax(dim=-1)  # 确保输出为概率分布
        ).to(device)

    def forward(self, img, tips_kpts, stage, tip_label):

        # img = img.to('cpu')
        img_features = self.feature_extractor(img.to(torch.float32))  # 提取图像特征
        # print(img.shape)
        # print(img_features.shape)
        # print(tips_kpts.shape)
        # print(stage.shape)
        # print(tip_label.flatten(start_dim=1).shape)
        # img_features = img_features.to(device)
        state_input = torch.cat([img_features, tips_kpts, stage,
                                 tip_label.flatten(start_dim=1)], dim=1).to(torch.float32)
        # print(state_input.shape)

        action_probs = self.actor(state_input)  # 动作预测
        state_value = self.critic(state_input)  # 状态值预测
        label_probs = self.label_predictor(state_input)
        batch_size = label_probs.size(0)
        label_probs = label_probs.reshape(batch_size, 2, -1)# 类别预测

        return action_probs, state_value, label_probs


class PPO:
    def __init__(self, state_dim, action_dim, label_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.policy = PPOActorCritic(state_dim, action_dim, label_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.policy_old = PPOActorCritic(state_dim, action_dim, label_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, img, tips_kpts, stage, tip_label, memory):
        with torch.no_grad():
            action_probs, state_value, label_probs = self.policy_old(img, tips_kpts, stage, tip_label)

        # 动作分布
        # action_dist = torch.distributions.Categorical(action_probs)


        action = action_probs

        predicted_label = label_probs

        # 存储到内存
        memory.states.append((img, tips_kpts, stage, tip_label))
        memory.actions.append(action)
        memory.labels.append(predicted_label)

        memory.values.append(state_value)

        return action, predicted_label
        # return action.item(), predicted_label.item()

    def update(self, memory, step):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 将 rewards 转换为 PyTorch 张量，并转移到设备
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)  # 标准化

        for _ in range(self.k_epochs):
            for (img, tips_kpts, stage,
                 tip_label), action, predicted_label, reward, value in zip(
                    memory.states, memory.actions, memory.labels, rewards,
                    memory.values):
                # 计算新的 action_probs, state_value, label_probs
                action_probs, state_value, label_probs = self.policy(img.to(device), tips_kpts.to(device),
                                                                     stage.to(device), tip_label.to(device))

                # 动作分布 (连续动作: x1, y1, x2, y2)
                action_diff = action_probs - action  # 计算动作的差异
                action_loss = 0.5 * torch.mean(action_diff.pow(2))  # 动作回归损失 (MSE)

                # 类别分布 (label: 1, 2, class_num)
                batch_size = label_probs.size(0)
                label_probs = label_probs.view(batch_size, -1)  # 展平类别维度
                label_target = predicted_label.view(batch_size, -1)  # 展平目标类别
                label_loss = torch.nn.functional.cross_entropy(label_probs, label_target.argmax(dim=1))

                # 值函数损失
                value_loss = 0.5 * self.MseLoss(state_value, reward)

                # 总损失
                loss = action_loss + value_loss + label_loss

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 更新旧策略的参数
            self.policy_old.load_state_dict(self.policy.state_dict())
            # memory.clear_memory()
        # exit(0)
        # 更新旧策略的参数
        # torch.cuda.empty_cache()
        # self.policy_old.load_state_dict(self.policy.state_dict())


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.labels = []  # 存储类别
        # self.logprobs = []
        # self.label_logprobs = []  # 存储类别的 log_probs
        self.values = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.labels = []
        # self.logprobs = []
        # self.label_logprobs = []
        self.values = []
        self.rewards = []
        self.is_terminals = []

