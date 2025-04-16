import torch
import torch.nn as nn


# ---------------------------------------------------------
# 1) Actions 特征提取分支 (使用分组卷积)
#    输入: (batch_size, 256, 4)
# ---------------------------------------------------------
class ActionConvExtractor(nn.Module):
    def __init__(self,
                 in_channels=256,
                 mid_channels=128,
                 out_channels=64,
                 groups=16):
        """
        通过若干层分组卷积，减少参数量。
        最终会将输出 Flatten 成一维向量。
        """
        super(ActionConvExtractor, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,  # 分组卷积
            bias=False
        )
        self.act1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv1d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False
        )
        self.act2 = nn.LeakyReLU(0.2)

        self.flatten = nn.Flatten()

    def forward(self, x):
        # x: (batch_size, 256, 4)
        x = self.conv1(x)  # -> (batch_size, mid_channels, 4)
        x = self.act1(x)
        x = self.conv2(x)  # -> (batch_size, out_channels, 4)
        x = self.act2(x)
        x = self.flatten(x)  # -> (batch_size, out_channels * 4)
        return x


# ---------------------------------------------------------
# 2) Labels 特征提取分支 (使用分组卷积)
#    输入: (batch_size, 256, 2, 10)
# ---------------------------------------------------------
class LabelConvExtractor(nn.Module):
    def __init__(self,
                 in_channels=2,
                 mid_channels=8,
                 out_channels=16,
                 groups=1):
        """
        输入形状: (b, 256, 2, 10)
        """
        super(LabelConvExtractor, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False
        )
        self.act1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv1d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False
        )
        self.act2 = nn.LeakyReLU(0.2)

        self.flatten = nn.Flatten()

    def forward(self, x):
        # 输入形状: (b, 256, 2, 10)
        batch_size, seq_len, in_channels, feature_len = x.size()

        # 将输入转换为 (b * 256, 2, 10)
        x = x.view(batch_size * seq_len, in_channels, feature_len)

        # 通过卷积层
        x = self.conv1(x)  # -> (b * 256, mid_channels, 10)
        x = self.act1(x)
        x = self.conv2(x)  # -> (b * 256, out_channels, 10)
        x = self.act2(x)

        # 展平
        x = self.flatten(x)  # -> (b * 256, out_channels * 10)

        # 恢复形状为 (b, 256, out_channels * 10)
        x = x.view(batch_size, seq_len, -1)

        return x


# ---------------------------------------------------------
# 3) 判别器 (带有共享的 MLP, 真实/生成 & 辅助输出)
# ---------------------------------------------------------
class LightweightDiscriminator(nn.Module):
    """
    - actions_extractor: 用于提取 (b, 256, 4) 的卷积特征
    - labels_extractor:  用于提取 (b, 256, 2, 10) 的卷积特征
    - 最终将提取到的特征与 stage (b, 10) 拼接，然后经过共享 MLP
    - 输出 logit (1维) + 辅助分类 (num_classes)
    """

    def __init__(self,
                 stage_dim=10,  # stage输入的维度
                 num_classes=1  # 辅助分类类别数, 仅示例
                 ):
        super(LightweightDiscriminator, self).__init__()

        # 1) 动作特征提取器
        self.actions_extractor = ActionConvExtractor(
            in_channels=256,
            mid_channels=128,
            out_channels=64,
            groups=16
        )
        # 最终输出维度: 64 * 4 = 256

        # 2) 标签特征提取器
        self.labels_extractor = LabelConvExtractor(
            in_channels=2,
            mid_channels=8,
            out_channels=16,
            groups=1
        )
        # 最终输出维度: 16 * 10 = 160

        # 3) 共享 MLP: 用于拼接后 (actions_feat + stage + labels_feat)
        #    若 actions_feat_dim = 256, labels_feat_dim = 160,
        #    加上 stage_dim=10，则输入总维度为 426
        fc_input_dim = 256 + 160 + stage_dim

        self.shared_layers = nn.Sequential(
            nn.Linear(fc_input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2)
        )

        # 4) 判别器输出 (真实/生成)
        self.logit = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1)  # 输出标量
            # nn.Sigmoid()     # 如需概率输出可启用
        )

        # 5) 辅助分类输出 (多分类)
        self.level_pred = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, num_classes)
            # nn.Sigmoid()
        )

    def forward(self, actions, stage, labels):
        """
        actions: (batch_size, 256, 4)
        stage:   (batch_size, 10)
        labels:  (batch_size, 256, 2, 10)
        """
        actions = actions.to(torch.float32)
        stage = stage.to(torch.float32)
        labels = labels.to(torch.float32)

        # 1) 分组卷积提取 actions 特征
        actions_feat = self.actions_extractor(actions)
        # actions_feat.shape = (batch_size, 64*4) = (batch_size, 256)

        # 2) 分组卷积提取 labels 特征
        labels_feat = self.labels_extractor(labels)
        # labels_feat.shape = (batch_size, 256, 16*10) = (batch_size, 256, 160)

        # 3) 对 labels_feat 取均值，降维到 (batch_size, 160)
        labels_feat = torch.mean(labels_feat, dim=1)

        # 4) 拼接所有特征
        # stage.shape = (batch_size, 10)
        x = torch.cat([actions_feat, stage, labels_feat], dim=1)
        # x.shape = (batch_size, 256 + 10 + 160) = (batch_size, 426)

        # 5) 通过共享的 MLP
        x = self.shared_layers(x)
        # x.shape = (batch_size, 32)

        # 6) 得到真假判别输出 (logit)
        logit_output = self.logit(x)
        # logit_output.shape = (batch_size, 1)

        # 7) 得到辅助分类输出 (num_classes)
        level_output = self.level_pred(x)
        # level_output.shape = (batch_size, num_classes)

        return logit_output, level_output


# ---------------------------------------------------------
# 4) 测试网络
# ---------------------------------------------------------
if __name__ == "__main__":
    batch_size = 8

    # 构造随机输入
    actions = torch.randn(batch_size, 256, 4)  # (b, 256, 4)
    stage = torch.randn(batch_size, 10)  # (b, 10)
    labels = torch.randn(batch_size, 256, 2, 10)  # (b, 256, 2, 10)

    model = LightweightDiscriminator(
        stage_dim=10,
        num_classes=1  # 示例
    )

    logit_out, level_out = model(actions, stage, labels)
    print("logit_out.shape =", logit_out.shape)  # (8, 1)
    print("level_out.shape =", level_out.shape)  # (8, 1)