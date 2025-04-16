import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from check_model import freeze_model
from model.LWANet_l import LWANet
from model.sd_mobile import SegNet
from src.learning_schduler import CosineLRSchedulerWithWarmupAndHold
from tqdm import tqdm
from src.utils.loss import LandmarkLoss, MultiClassDiceLoss, FocalLoss, KLDivLossSelective, MaskedMSELoss
from src.utils.Lovaszloss import LovaszLoss
import os
from config import params
from preprocessing.DataLoader.cataract_dataloader import load_dataloader
from torch.utils.tensorboard import SummaryWriter

from model.transformer import MobileUnet
# from model.attention_Unet import AttU_Net

# 训练模型
def train_model():
    # 加载数据集
    num_epochs = params['num_epoch']
    warmup_epochs = params['model_param']['warmup_epochs']
    hold_epochs = params['model_param']['hold_epochs']
    initial_lr = params['model_param']['max_learning_rate']
    min_lr = params['model_param']['base_learning_rate']
    save_interval = params['model_param']['save_checkpoint_epochs']
    num_classes = len(params['model_param']['tool_and_eyes_class']) + 1
    pretrain_path = params['model_param']['pretrain_model_path']
    saved_checkpoints = []

    train_loader = load_dataloader()
    # 实例化模型，损失函数，优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    if not os.path.exists(params['model_param']['model_dir']):
        os.makedirs(params['model_param']['model_dir'])

    # model = SegNet(num_classes=16, num_landmarks=params['model_param']['max_kpts'])
    model = LWANet(num_classes=16)

    model.to(device)

    weights = torch.ones(16, device=device)
    weights[0] = 0.001  # 背景类的权重
    weights[1:12] = 3.0
    weights[1], weights[4] = 1.2, 1.2
    weights[2], weights[7] = 0.001, 0.001
    weights[12:] = 0.0001

    landmark_criterion = LandmarkLoss()
    criterion_class = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.SGD([
        {'params': model.landmark_coords_branch.parameters(), 'lr': min_lr, 'name': 'uv_lr'},  # 分支1学习率
        {'params': model.landmark_class_branch.parameters(), 'lr': min_lr, 'name': 'class_lr'},  # 分支2学习率
        {'params': [p for n, p in model.named_parameters()
                    if 'landmark_coords_branch' not in n and 'landmark_class_branch' not in n],
         'lr': min_lr, 'name': 'default_lr'}
    ], momentum=0.9, weight_decay=params['model_param']['weight_decay'])
    # optimizer = optim.Adam(model.parameters(), lr=min_lr, weight_decay=params['model_param']['weight_decay'])
    scheduler = CosineLRSchedulerWithWarmupAndHold(optimizer, num_epochs, warmup_epochs, hold_epochs,
                                                   initial_lrs=[2*initial_lr, initial_lr, initial_lr],
                                                   warmup_initial_lrs=[min_lr, min_lr, min_lr], min_lrs=[1e-6, 1e-6, 1e-6])
    writer = SummaryWriter(log_dir=params['model_param']['model_dir'])
    if not params['is_train_from_begin']:
        pretrained_dict = torch.load(pretrain_path)
        model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
        #                    k in model_dict and "layer" in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # model.load_state_dict(torch.load(pretrain_path))
        if params['model_param']['freeze_decoder']:
            model = freeze_model(model)
        print('model loaded')
    model.train()
    step = 0
    model_save_path_ab = params['model_param']['model_dir']
    for epoch in range(num_epochs):
        running_loss = 0.0
        loss_dice_total, loss_cross_total, loss_kpt_total, loss_class_total, landmark_loss_total = 0, 0, 0, 0, 0
        for images, mask, tips_landmark_class in tqdm(train_loader):
            images = images.to(device)
            # targets = mask.to(device)
            tips_landmark_class = tips_landmark_class.to(device)
            # targets_expand = targets.unsqueeze(1)
            # targets_sobel = apply_sobel_filter_multiclass_torch(targets_expand, num_classes=16)

            landmark_targets = tips_landmark_class[:, :, :2].to(device)
            class_targets = tips_landmark_class[:, :, 2].to(device)

            optimizer.zero_grad()
            if torch.all(class_targets == 0):
                print("Skipping batch due to all class_targets being zero.")
                continue
            landmark_pred, class_pred = model(images)  # Done with log_softmax

            loss_kpt = landmark_criterion(landmark_pred, landmark_targets)

            class_pred = class_pred.view(-1, num_classes)
            class_targets = class_targets.long().view(-1)
            # print(class_pred.shape, class_targets.shape)
            loss_class = criterion_class(class_pred, class_targets)
            if loss_class.item() == 0:
                loss_class = torch.tensor(1e-8, requires_grad=True, device=class_targets.device)


            # landmark_loss = torch.log(1 + loss_kpt * loss_class)
            landmark_loss = loss_kpt + loss_class


            writer.add_scalar('Loss_Landmark/loss_kpt', loss_kpt, step)
            writer.add_scalar('Loss_Landmark/loss_class', loss_class, step)
            writer.add_scalar('Loss_Landmark', landmark_loss, step)
            # '''
            step += 1

            loss = landmark_loss
            writer.add_scalar('Loss_total', loss, step)

            for i, param_group in enumerate(optimizer.param_groups):
                group_name = param_group['name']
                writer.add_scalar(f'Learning Rate/{group_name}', param_group['lr'], step)
            # writer.add_scalar('Learning Rate', current_lr, step)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)

            optimizer.step()
            # print(images.size())
            running_loss += loss.item() * params['batch_size']
            torch.cuda.empty_cache()
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        epoch_loss = running_loss / len(train_loader.dataset)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, LR:{current_lr:.8f},'
              )

        if (epoch + 1) % save_interval == 0:
            # Define the path and filename for saving the model
            model_save_path = os.path.join(model_save_path_ab, f'epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved at epoch {epoch + 1}')

            saved_checkpoints.append(model_save_path)

            if len(saved_checkpoints) > params['max_checkpoints']:
                oldest_checkpoint = saved_checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)
                    print(f"Deleted old checkpoint: {oldest_checkpoint}")

    writer.close()


# 运行训练
if __name__ == '__main__':
    train_model()
