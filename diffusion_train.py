from src.utils.loss import TrajectoryLoss, total_variation_loss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from src.learning_schduler import CosineLRSchedulerWithWarmupAndHold
from tqdm import tqdm
from config import params
from network.diffusion import DiffusionLightweightTSMGenerator
from network.dis_net import LightweightDiscriminator
from data.dataloader import load_dataloader

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

    dis = LightweightDiscriminator(stage_dim=10, num_classes=1).to(device)
    tra_gen = DiffusionLightweightTSMGenerator(n_segment=10).to(device)


    optimizer_D = optim.SGD(dis.parameters(), lr=min_lr, weight_decay=params['model_param']['weight_decay'],
                                momentum=0.8)
    optimizer_G = optim.SGD(tra_gen.parameters(), lr=min_lr, weight_decay=params['model_param']['weight_decay'],
                                momentum=0.8)

    criterion = TrajectoryLoss(weight_tra=1.0, weight_level=0.8).to(device)

    total_epoch = params['num_epoch']
    train_dataloader = load_dataloader()
    if not params['is_train_from_begin']:
        pretrain_path = params['model_param']['pretrain_model_path']
        checkpoint = torch.load(pretrain_path)
        gen_state_dict = checkpoint["generator_state_dict"]
        optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        tra_gen.load_state_dict(gen_state_dict, strict=False)
        if params['is_train_from_begin']:
            dis.load_state_dict(checkpoint["discriminator_state_dict"])

            optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])

            epoch_num = checkpoint["epoch"]
            print(f"Model loaded successfully from {pretrain_path}, starting from epoch {epoch_num}")
        else:
            print(f"Generator Model loaded successfully from {pretrain_path}")


    step = 0
    saved_checkpoints = []

    scheduler_G = CosineLRSchedulerWithWarmupAndHold(optimizer_G, total_epoch, warmup_epochs, hold_epochs,
                                                   initial_lr=initial_lr, warmup_initial_lr=min_lr, min_lr=1e-6)
    scheduler_D = CosineLRSchedulerWithWarmupAndHold(optimizer_D, total_epoch, warmup_epochs+2, hold_epochs,
                                                   initial_lr=initial_lr/2, warmup_initial_lr=min_lr, min_lr=1e-6)

    for epoch in range(total_epoch):
        epoch_num = epoch + 1
        step_batch = 0
        for batch_idx, expert_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            step += 1
            videos_batch = expert_data["video_data"].to(device)
            tips_kpts_batch = expert_data["kpt_uv"].to(device)
            label_batch = expert_data["label"].to(device)
            surgery_stage_batch = expert_data["surgery_stage"].to(device)
            surgery_level_batch = expert_data["surgery_level"].to(device)

            real_labels = torch.full((videos_batch.size(0), 1), 0.9).to(device)  # 真实标签 1 -> 0.9
            fake_labels = torch.full((videos_batch.size(0), 1), 0.1).to(device)  # 假标签 0 -> 0.1
            fake_level_labels = torch.zeros_like(surgery_level_batch).to(device)
            high_fake_level_labels = torch.ones_like(surgery_level_batch).to(device)

            noise = torch.randn_like(tips_kpts_batch) * 0.05  # 添加 5% 的噪声


            # -------------------------Generator-----------------------
            # '''
            tra_policy, policy_label = tra_gen(videos_batch, surgery_stage_batch)
            tra_policy = torch.clamp(tra_policy, 0, 1)
            fake_output, fake_level_output = dis(tra_policy, surgery_stage_batch, policy_label)
            lambda_reg = 0.1
            lambda_kl = 0.1
            lambda_tv = 0.1
            eps = 1e-10

            gen_log_prob = F.log_softmax(tra_policy + eps, dim=-1)
            expert_prob = F.softmax(tips_kpts_batch + eps, dim=-1)
            kl_loss = F.kl_div(gen_log_prob, expert_prob, reduction='batchmean')
            reg_loss = torch.nn.functional.mse_loss(tra_policy.to(torch.float32), tips_kpts_batch.to(torch.float32))
            tv_loss = total_variation_loss(tra_policy)

            g_loss = (-torch.log(torch.sigmoid(fake_output) + 1e-10).mean() + lambda_reg * reg_loss
                      + lambda_kl * kl_loss + lambda_tv * tv_loss)
            '''
            print('________'*10)
            print(g_loss)
            print('simple loss', -torch.log(torch.sigmoid(fake_output) + 1e-10).mean())
            print('gen_log_prob', gen_log_prob.mean())
            print('expert_log_prob', expert_prob.mean())
            print('lambda_tv', tv_loss)
            print('lambda_kl', kl_loss)
            print('lambda_reg', reg_loss)
            print('________' * 10)
            '''
            g_criterion = criterion(fake_output, fake_level_output, real_labels, high_fake_level_labels)
            g_loss_total = g_loss + 0.5 * g_criterion
            writer.add_scalar('G_loss/dis', g_loss, step)
            writer.add_scalar('G_loss/criterion', g_criterion, step)
            writer.add_scalar('G_loss', g_loss_total, step)

            optimizer_G.zero_grad()
            g_loss_total.backward()
            optimizer_G.step()

            # --------------------Discriminator--------------------------------
            real_output, real_level_output = dis(tips_kpts_batch + noise, surgery_stage_batch, label_batch)

            # print(real_output.max().item())
            # print(real_output.min().item())
            # exit(0)
            d_loss_real = criterion(real_output, real_level_output, real_labels, surgery_level_batch)
            # exit(0)
            writer.add_scalar('D_loss/d_loss_real', d_loss_real, step)

            fake_output, fake_level_output = dis(tra_policy.detach(), surgery_stage_batch.detach(), policy_label.detach())
            d_loss_fake = criterion(fake_output, fake_level_output, fake_labels, fake_level_labels)
            writer.add_scalar('D_loss/d_loss_fake', d_loss_fake, step)

            # 判别器总损失
            d_loss = d_loss_real + d_loss_fake

            writer.add_scalar('D_loss', d_loss, step)

            # 更新判别器
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            current_lr_g = optimizer_G.param_groups[0]['lr']
            current_lr_d = optimizer_D.param_groups[0]['lr']

            writer.add_scalar('Learning Rate/g_lr', current_lr_g, step)
            writer.add_scalar('Learning Rate/d_lr', current_lr_d, step)


            # 更新生成器
            if batch_idx % 50 == 0:
                print(f"Step_total {step}/Step {step_batch}/Epoch {epoch_num}: D Loss: {d_loss.item():.4f},"
                      f" D_loss_real: { d_loss_real.item():.4f}, D_loss_fake: {d_loss_fake.item():.4f}"
                      f" G_loss_criterion: {g_criterion.item():.4f}, G_loss: {g_loss.item():.4f}")



            # Saving Processing
            if step % 1000 == 0:
                model_save_path = os.path.join(checkpoints_path,
                                                f"models_step_{step}.pth")
                torch.save({
                    "generator_state_dict": tra_gen.state_dict(),
                    "discriminator_state_dict": dis.state_dict(),
                    "optimizer_D_state_dict": optimizer_D.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "epoch": epoch_num
                }, model_save_path)
                saved_checkpoints.append(model_save_path)
                print(f"Models saved at epoch {epoch_num}: {model_save_path}")

            if len(saved_checkpoints) > params['max_checkpoints']:
                oldest_checkpoint = saved_checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)
                    print(f"Deleted old checkpoint: {oldest_checkpoint}")
        scheduler_D.step()
        scheduler_G.step()
    writer.close()


if __name__ == "__main__":
    train()
