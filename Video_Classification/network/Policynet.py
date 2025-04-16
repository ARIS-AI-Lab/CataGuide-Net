import torch
import torch.nn as nn
from network.light_vit import mobilevit_xxs

class PolicyNet(nn.Module):
    def __init__(self, num_tools=10, device='cpu'):
        super(PolicyNet, self).__init__()
        self.num_tools = num_tools
        self.fc1 = nn.Linear(142, 64)  # 输入维度从 6 改为 7（包括 stage 标号）
        self.fc2 = nn.Linear(64, 64)
        self.action_head = nn.Linear(64, 4)
        self.tool_head = nn.Linear(64, num_tools*2)
        self.feature_extractor = mobilevit_xxs()

    def forward(self, img, tip, stage):
        # 将 stage 标号加入到输入中
        img_features = self.feature_extractor(img)
        # print(img_features.shape)
        # print(tip.shape)
        # print(stage.shape)
        stage = stage.unsqueeze(0)
        x = torch.cat([img_features, tip, stage], dim=1).to(dtype=torch.float32)

        # print(x.shape)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        action = self.action_head(x)
        tool_logits = self.tool_head(x)
        tool_logits = tool_logits.view(1, 2, self.num_tools)
        # print(tool_logits.shape)
        return action, tool_logits
