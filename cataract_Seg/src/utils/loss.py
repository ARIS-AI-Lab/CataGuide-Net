import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # 假设输入已经经过 Sigmoid 激活
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # print(inputs.size(), targets.size())
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, high_class_weight=0.1, background_weight=0,
                 ignore_index=0):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.high_class_weight = high_class_weight
        self.background_weight = background_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        pred = torch.exp(pred)
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = F.softmax(pred, dim=1)

        # Reshape pred and target
        # [N, C, W, H] -> [N, C, W*H] -> [N * W * H, C]
        pred = pred.view(pred.size(0), pred.size(1), -1).transpose(1, 2).contiguous().view(-1, pred.size(1))

        # [N, W, H] -> [N * W * H]
        target = target.view(-1).contiguous()

        # Mask out ignore_index
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        # Get number of classes from pred
        num_classes = pred.shape[1]

        # Convert target to one-hot format
        target = F.one_hot(torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes)
        # target = target * valid_mask

        # Calculate Dice loss
        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(pred[:, i] * target[:, i]) * 2 + self.smooth
                den = torch.sum(pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)) + self.smooth
                dice_loss = 1 - num / den
                if i == 0:
                    dice_loss *= self.background_weight

                elif i >= 12:
                    dice_loss *= self.high_class_weight
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss



class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, num_classes=10):
        super(CombinedLoss, self).__init__()
        self.nll_loss = nn.NLLLoss()
        self.dice = MultiClassDiceLoss(num_classes)
        self.alpha = alpha

    def forward(self, inputs, targets):

        log_softmax_inputs = torch.log(inputs + 1e-8)
        nll_loss = self.nll_loss(log_softmax_inputs, targets)
        dice_loss = self.dice(inputs, targets)

        return self.alpha * nll_loss + (1 - self.alpha) * dice_loss


class LandmarkLoss(nn.Module):
    def __init__(self, w=10, epsilon=2):
        super(LandmarkLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon

    def forward(self, predicted_landmarks, target_landmarks):
        # 联合有效性掩码，确保 x 和 y 同时有效
        # print(target_landmarks)
        validity_mask = (
                (target_landmarks[..., 0] > 0) & (target_landmarks[..., 0] < 1) &
                (target_landmarks[..., 1] > 0) & (target_landmarks[..., 1] < 1)
        ).float()  # 形状：[N]

        # 提取有效的预测和目标关键点
        valid_predicted = predicted_landmarks[validity_mask == 1]  # 形状：[M, 2]
        valid_target = target_landmarks[validity_mask == 1]  # 形状：[M, 2]

        # print(valid_predicted.shape)
        #
        # print(valid_target.shape)
        # print(valid_target)
        # print('*********'*10)
        # print(valid_predicted)
        # print('*********'*10)
        # print(validity_mask)
        # exit(0)
        # 计算每个有效坐标的误差
        errors = torch.abs(valid_predicted - valid_target)  # 计算每个坐标的误差 [N, 2]

        # WingLoss 核心计算
        w_tensor = torch.tensor(self.w, device=errors.device, dtype=errors.dtype)
        epsilon_tensor = torch.tensor(self.epsilon, device=errors.device, dtype=errors.dtype)
        c = w_tensor - w_tensor * torch.log1p(w_tensor / epsilon_tensor)

        wing_loss = torch.where(
            errors < w_tensor,
            w_tensor * torch.log1p(errors / epsilon_tensor),
            errors - c
        )

        # 计算最终损失值
        if wing_loss.numel() > 0:
            loss = wing_loss.mean()  # 计算有效关键点的平均损失
        else:
            loss = torch.tensor(0.0, requires_grad=True, device=errors.device)
        # exit(0)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)):
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = input
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class KLDivLossSelective(nn.Module):
    def __init__(self, valid_classes=range(1, 13)):  # 只保留1到12的类别
        super(KLDivLossSelective, self).__init__()
        self.valid_classes = valid_classes
        self.kl_loss = nn.KLDivLoss(reduction='none')  # 使用 `none` 以便逐类别计算

    def forward(self, pred, target):
        # 假设 pred 已经是 log-softmax 形式
        target = F.softmax(target, dim=-1)  # 对 target 应用 softmax

        # 创建类别掩码，仅保留 1 到 12 类别的标签
        class_mask = torch.zeros_like(target, dtype=torch.bool)
        class_mask.scatter_(-1, torch.tensor(self.valid_classes, device=target.device).view(1, 1, -1), True)

        # 创建无效关键点掩码
        valid_points_mask = target.sum(dim=-1) != 0  # 只保留非无效关键点的位置
        valid_points_mask = valid_points_mask.unsqueeze(-1).expand_as(target)  # 扩展维度匹配 target

        # 结合类别掩码和无效关键点掩码
        final_mask = class_mask & valid_points_mask

        # 计算未掩码的 KLDivLoss
        raw_loss = self.kl_loss(pred, target)

        # 仅保留有效类别和有效关键点的损失
        masked_loss = raw_loss * final_mask.float()

        # 避免除以零
        valid_count = final_mask.sum().clamp_min(1)
        loss = masked_loss.sum() / valid_count  # 计算有效类别的平均损失

        return loss



class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')  # 使用 'none' 以逐元素计算

    def forward(self, pred, target):
        # 检查 pred 和 target 的设备、数据类型和形状是否匹配
        assert pred.device == target.device, "pred and target must be on the same device"
        assert pred.shape == target.shape, f"Mismatch in shape: pred shape {pred.shape}, target shape {target.shape}"
        assert pred.dtype == target.dtype, "pred and target must have the same dtype"

        # 创建掩码，忽略标签为 -1 的关键点（假设 target 无效关键点的所有类别值都为 -1）
        mask = (target[..., 0] != -1).float().unsqueeze(-1)  # 在最后一维扩展以匹配 one-hot 格式

        # 将无效关键点位置设置为全零张量
        target = torch.where(target == -1, torch.zeros_like(target), target)

        # 计算 MSE 损失
        raw_loss = self.mse_loss(pred, target)

        # 应用掩码，仅计算有效关键点的损失
        masked_loss = raw_loss * mask
        loss = masked_loss.sum() / mask.sum().clamp_min(1)  # 避免除以零

        return loss