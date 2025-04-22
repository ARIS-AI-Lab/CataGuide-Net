import torch
import torch.nn as nn

class RewardNetwork(nn.Module):
    def __init__(self, input_dim):
        super(RewardNetwork, self).__init__()

        # 定义网络结构
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # 增加隐藏层大小
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # 添加 Dropout 防止过拟合
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # 输出奖励值
            nn.Sigmoid(),
        )

    def forward(self, action, tool_logits):
        tool_logits = tool_logits.view(1, -1)
        x = torch.cat([action, tool_logits], dim=-1).to(torch.float32)


        return self.net(x)
