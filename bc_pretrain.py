from sched import scheduler

from src.utils.loss import TrajectoryLoss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from tqdm import tqdm
from config import params
from src.learning_schduler import CosineLRSchedulerWithWarmupAndHold
from network.gail import LightweightTSMGenerator
from network.dis_net import LightweightDiscriminator
from network.diffusion import DiffusionLightweightTSMGenerator
from data.dataloader import load_dataloader
from network.reward_network import RewardNetwork


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_writer(log_dir="./checkpoints"):
    return SummaryWriter(log_dir=log_dir)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints_path = params['checkpoints_path']
    check_path(checkpoints_path)

    # log_dir = os.path.join(checkpoints_path, 'logs')
    writer = SummaryWriter(log_dir=checkpoints_path)
    warmup_epochs = params['model_param']['warmup_epochs']
    hold_epochs = params['model_param']['hold_epochs']
    initial_lr = params['model_param']['max_learning_rate']
    min_lr = params['model_param']['base_learning_rate']

    # dis = LightweightDiscriminator(stage_dim=10, num_classes=1).to(device)
    # gen = LightweightTSMGenerator().to(device)
    gen = DiffusionLightweightTSMGenerator().to(device)
    # optimizer_G = optim.Adam(gen.parameters(), lr=min_lr)
    optimizer_G = optim.SGD(gen.parameters(), lr=min_lr, weight_decay=params['model_param']['weight_decay'],
                                momentum=0.8)
    # optimizer_D = optim.Adam(dis.parameters(), lr=1e-4)
    criterion = nn.SmoothL1Loss(reduction='none')
    label_criterion = nn.CrossEntropyLoss(reduction='mean')
    total_epoch = params['num_epoch']

    train_dataloader = load_dataloader()
    if not params['is_train_from_begin']:
        pretrain_path = params['model_param']['pretrain_model_path']
        checkpoint = torch.load(pretrain_path)
        # gen_state_dict = checkpoint["generator_state_dict"]
        # gen_state_dict = {k: v for k, v in gen_state_dict.items() if "stage_embedding.weight" not in k}
        # gen.load_state_dict(gen_state_dict, strict=False)
        gen.load_state_dict(checkpoint["generator_state_dict"])
        # dis.load_state_dict(checkpoint["discriminator_state_dict"])
        # optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        # optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        epoch_num = checkpoint["epoch"]

        print(f"Model loaded successfully from {pretrain_path}, starting from epoch {epoch_num}")
    scheduler = CosineLRSchedulerWithWarmupAndHold(optimizer_G, total_epoch, warmup_epochs, hold_epochs,
                                                   initial_lr=initial_lr, warmup_initial_lr=min_lr, min_lr=1e-6)
    step = 0
    step_batch = 0
    # alpha = params['mix_alpha']
    # saved_checkpoints = [i for i in os.listdir(checkpoints_path) if i.endswith('.pth')]
    saved_checkpoints = []
    d_loss = 0

    for epoch in range(total_epoch):
        epoch_num = epoch + 1
        step_batch = 0
        for batch_idx, expert_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            step_batch += 1
            step += 1
            videos_batch = expert_data["video_data"].to(device)
            tips_kpts_batch = expert_data["kpt_uv"].to(device)
            label_batch = expert_data["label"].to(device)
            surgery_stage_batch = expert_data["surgery_stage"].to(device)

            #--------------------Discriminator--------------------------------
            mask = (tips_kpts_batch.abs().sum(dim=-1) > 0).float()

            tra_policy, policy_label = gen(videos_batch, surgery_stage_batch)

            loss_all_frame = criterion(tra_policy.to(torch.float32), tips_kpts_batch.to(torch.float32))
            loss_per_frame = loss_all_frame.mean(dim=-1)  # shape: (b, 256)
            loss_policy = (loss_per_frame * mask).sum() / (mask.sum() + 1e-8)

            loss_label = label_criterion(policy_label.permute(0, 3, 1, 2), torch.argmax(label_batch, dim=-1))

            loss = loss_policy + 0.4 * loss_label

            current_lr = optimizer_G.param_groups[0]['lr']
            writer.add_scalar('Loss/loss_policy', loss_policy, step)
            writer.add_scalar('Loss/loss_label', loss_label, step)


            writer.add_scalar('total_loss', loss, step)
            writer.add_scalar('Learning Rate', current_lr, step)

            # 更新判别器
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()


            if batch_idx % 50 == 0:
                print(f"Step_total {step}/Step {step_batch}/Epoch {epoch_num}: total_loss: {loss.item():.4f}, loss_policy: {loss_policy.item():.4f},"
                      f"loss_label: {loss_label.item():.4f}")


            if step % 1000 == 0:
                model_save_path = os.path.join(checkpoints_path, f"models_step_{step}.pth")
                torch.save({
                    "generator_state_dict": gen.state_dict(),
                    # "discriminator_state_dict": dis.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    # "optimizer_D_state_dict": optimizer_D.state_dict(),
                    "epoch": epoch_num
                }, model_save_path)
                saved_checkpoints.append(model_save_path)
                print(f"Models saved at step {step}: {model_save_path}")

            if len(saved_checkpoints) > params['max_checkpoints']:
                oldest_checkpoint = saved_checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)
                    print(f"Deleted old checkpoint: {oldest_checkpoint}")
        scheduler.step()

    writer.close()


if __name__ == "__main__":
    train()
