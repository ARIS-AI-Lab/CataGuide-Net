from cProfile import label
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from tqdm import tqdm
from config import params
from network.SurgicalEnv import SurgicalEnvironment
from network.Policynet import PolicyNet
from network.ppo import PPO, Memory
from network.light_vit import mobilevit_xxs
from data.dataloader import load_dataloader
from network.reward_network import RewardNetwork



# GAIL判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.logit = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # nn.Sigmoid()  # 输出真实/生成的概率
        )

        self.level_pred = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, actions, stage):

        stage = stage.view(1, -1)
        actions = actions.view(1, -1)
        # print(actions.shape)
        # print(stage.shape)
        x = torch.cat([actions, stage], dim=1).to(torch.float32)


        x = self.shared_layers(x)
        logit_output = self.logit(x)
        level_output = self.level_pred(x)

        return logit_output, level_output

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_writer(log_dir="./checkpoints"):
    return SummaryWriter(log_dir=log_dir)

def train():
    # gc.collect()
    # torch.cuda.empty_cache()

    checkpoints_path = params['checkpoints_path']
    check_path(checkpoints_path)

    # log_dir = os.path.join(checkpoints_path, 'logs')
    writer = SummaryWriter(log_dir=checkpoints_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # exit(0)
    total_epoch = params['num_epoch']
    train_dataloader = load_dataloader()
    batch_size = params['batch_size']
    # reward_network = RewardNetwork(input_dim=24).to(device)
    # disc_net = Discriminator(input_dim=1034).to(device)
    # policy_net = PolicyNet().to(device)
    ppo = PPO(state_dim=128, action_dim=4, label_dim=20)
    # discriminator_optimizer = optim.Adam(disc_net.parameters(), lr=1e-4)
    # policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    # reward_optimizer = optim.Adam(reward_network.parameters(), lr=1e-4)
    # real_time_tool_func = nn.BCEWithLogitsLoss()
    # logit_loss_func = nn.BCEWithLogitsLoss()
    # level_loss_func = nn.BCEWithLogitsLoss()
    # distance_func = nn.MSELoss(reduction='mean')

    step = 0
    step_batch = 0
    alpha = params['mix_alpha']
    saved_checkpoints = []

    for epoch in range(total_epoch):
        epoch_num = epoch + 1
        for batch_idx, expert_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # step_batch += 1
            videos_batch = expert_data["video_data"].to(device)
            tips_kpts_batch = expert_data["kpt_uv"].to(device)
            label_batch = expert_data["label"].to(device)
            surgery_stage_batch = expert_data["surgery_stage"].to(device)
            # surgery_level_batch = expert_data["surgery_level"].to(device)
            # print(videos_batch.shape)
            memory = Memory()



            for b in range(batch_size):
                video = videos_batch[b]
                tips_kpts = tips_kpts_batch[b]

                surgery_stage = surgery_stage_batch[b]
                # surgery_stage = surgery_stage.unsqueeze(0)
                # print(surgery_stage.shape)
                # surgery_level = surgery_level_batch[b]
                labels = label_batch[b]
                # print(labels.shape)
                env = SurgicalEnvironment(video, tips_kpts, surgery_stage, labels,
                                          device=device)
                for episode in tqdm(range(3000)):
                    state = env.reset()
                    step_batch += 1
                    #
                    for _ in range(256):
                        step += 1
                        # (current_state, tip,
                        #  current_stage, current_label) = state

                        (current_state, tip,
                         current_stage, current_label, next_kpt, next_label) = state

                        action, predicted_label = ppo.select_action(current_state, tip, current_stage,
                                                                    current_label, memory)

                        next_state, reward, done = env.step(action, predicted_label, next_kpt, next_label)
                        # print(reward)
                        # exit(0)
                        writer.add_scalar('reward', reward, step)
                        # reward = -torch.norm(tips_kpts - expert_kpt, dim=1).mean()  # 计算奖励
                        memory.rewards.append(reward)
                        memory.is_terminals.append(done)
                        if done:
                            break
                        state = next_state
                        torch.cuda.empty_cache()

                    ppo.update(memory, step_batch)
                    memory.clear_memory()

                    # print(f"Reward is : {total_reward}")

                    if episode % 10 == 0:
                        model_save_path = os.path.join(checkpoints_path,
                                                       f"checkpoint_epoch_{epoch_num}_batch_{batch_idx}_step_{episode}.pth")
                        torch.save({
                            'policy_net': ppo.policy.state_dict(),
                            # 'policy_optimizer': policy_optimizer.state_dict(),
                            'epoch': epoch_num,
                            'step': step
                        }, model_save_path)
                        saved_checkpoints.append(model_save_path)

                    if len(saved_checkpoints) > params['max_checkpoints']:
                        oldest_checkpoint = saved_checkpoints.pop(0)
                        if os.path.exists(oldest_checkpoint):
                            os.remove(oldest_checkpoint)
                            print(f"Deleted old checkpoint: {oldest_checkpoint}")


    writer.close()






if __name__ == "__main__":
    train()
