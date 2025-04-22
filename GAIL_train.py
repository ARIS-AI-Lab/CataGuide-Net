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
from network.gail import LightweightTSMGenerator
from network.dis_net import LightweightDiscriminator
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


    dis = LightweightDiscriminator(stage_dim=10, num_classes=1).to(device)
    gen = LightweightTSMGenerator().to(device)
    optimizer_G = optim.Adam(gen.parameters(), lr=1e-4)
    optimizer_D = optim.Adam(dis.parameters(), lr=1e-4)
    criterion = TrajectoryLoss(weight_tra=1.0, weight_level=0.8)
    criterion_g = nn.BCEWithLogitsLoss()
    total_epoch = params['num_epoch']
    train_dataloader = load_dataloader()
    if not params['is_train_from_begin']:
        pretrain_path = params['model_param']['pretrain_model_path']
        checkpoint = torch.load(pretrain_path)
        gen_state_dict = checkpoint["generator_state_dict"]
        gen_state_dict = {k: v for k, v in gen_state_dict.items() if "stage_embedding.weight" not in k}
        gen.load_state_dict(gen_state_dict, strict=False)
        # gen.load_state_dict(checkpoint["generator_state_dict"])
        dis.load_state_dict(checkpoint["discriminator_state_dict"])
        optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        epoch_num = checkpoint["epoch"]

        print(f"Model loaded successfully from {pretrain_path}, starting from epoch {epoch_num}")

    step = 0
    # step_batch = 0
    # alpha = params['mix_alpha']
    saved_checkpoints = []
    d_loss = 0

    for epoch in range(total_epoch):
        epoch_num = epoch + 1
        for batch_idx, expert_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            step += 1
            videos_batch = expert_data["video_data"].to(device)
            tips_kpts_batch = expert_data["kpt_uv"].to(device)
            label_batch = expert_data["label"].to(device)
            surgery_stage_batch = expert_data["surgery_stage"].to(device)
            surgery_level_batch = expert_data["surgery_level"].to(device)
            # print(videos_batch.shape)
            # print(videos_batch.shape)
            # print(tips_kpts_batch.shape)
            # print(surgery_stage_batch.shape)
            # print(label_batch.shape)
            #--------------------Discriminator--------------------------------
            tra_policy, policy_label = gen(videos_batch, surgery_stage_batch)

            real_labels = torch.full((videos_batch.size(0), 1), 0.9).to(device)  # 真实标签 1 -> 0.9
            fake_labels = torch.full((videos_batch.size(0), 1), 0.1).to(device)  # 假标签 0 -> 0.1

            fake_level_labels = torch.zeros_like(surgery_level_batch).to(device)
            high_fake_level_labels = torch.ones_like(surgery_level_batch).to(device)
            # print(label_batch.shape)
            # print(policy_label.shape)
            # print(tra_policy.shape)
            # print(tips_kpts_batch.shape)

            if step % 2:
                noise = torch.randn_like(tips_kpts_batch) * 0.05  # 添加 5% 的噪声
                real_output, real_level_output = dis(tips_kpts_batch + noise, surgery_stage_batch, label_batch)

                # print(real_output.shape)
                d_loss_real = criterion(real_output, real_level_output, real_labels, surgery_level_batch)
                writer.add_scalar('D_loss/d_loss_real', d_loss_real, step)

                fake_output, fake_level_output = dis(tra_policy.detach(), surgery_stage_batch.detach(), policy_label.detach())
                d_loss_fake = criterion(fake_output, fake_level_output, fake_labels, fake_level_labels)
                writer.add_scalar('D_loss/d_loss_real', d_loss_fake, step)

                # 判别器总损失
                d_loss = d_loss_real + d_loss_fake

                writer.add_scalar('D_loss', d_loss, step)

                # 更新判别器
                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

            #-------------------------Generator-----------------------
            fake_output, fake_level_output = dis(tra_policy, surgery_stage_batch, policy_label)
            lambda_reg = 0.01
            reg_loss = torch.nn.functional.mse_loss(tra_policy.to(torch.float32), tips_kpts_batch.to(torch.float32))
            g_loss = -torch.log(torch.sigmoid(fake_output) + 1e-10).mean() + lambda_reg * reg_loss


            # g_loss = criterion(fake_output, fake_level_output, real_labels, high_fake_level_labels)
            writer.add_scalar('G_loss', g_loss, step)
            # 更新生成器
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()
            if batch_idx % 50 == 0:
                print(f"Step {step}/Epoch {epoch_num}: D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        if epoch_num % 1 == 0:
            model_save_path = os.path.join(checkpoints_path,
                                            f"models_epoch_{epoch_num}.pth")
            torch.save({
                "generator_state_dict": gen.state_dict(),
                "discriminator_state_dict": dis.state_dict(),
                "optimizer_G_state_dict": optimizer_G.state_dict(),
                "optimizer_D_state_dict": optimizer_D.state_dict(),
                "epoch": epoch_num
            }, model_save_path)
            print(f"Models saved at epoch {epoch_num}: {model_save_path}")

        if len(saved_checkpoints) > params['max_checkpoints']:
            oldest_checkpoint = saved_checkpoints.pop(0)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)
                print(f"Deleted old checkpoint: {oldest_checkpoint}")

    writer.close()


if __name__ == "__main__":
    train()
