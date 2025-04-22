import torch
import os
import numpy as np
import matplotlib.pyplot as plt

# 如果您使用的是相对导入，需要保证脚本运行环境和包结构与训练时一致
from network.Policynet import PolicyNet
from network.ppo import PPO, Memory
from network.SurgicalEnv import SurgicalEnvironment
from data.dataloader import load_dataloader  # 如果需要从DataLoader加载数据

def test_and_visualize_actions(checkpoint_path):
    """
    加载训练好的 PPO 模型，进行若干步推理，并可视化各步的 action。
    Args:
        checkpoint_path (str): 训练过程中保存的 .pth 文件路径
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 创建 PPO 实例并加载权重
    #    要确保与训练脚本保持一致的参数 state_dim / action_dim / label_dim
    ppo = PPO(state_dim=128 + 4 + 10, action_dim=4, label_dim=20)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} 不存在，请确认路径正确。")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 加载训练时保存的 policy 网络权重
    ppo.policy.load_state_dict(checkpoint['policy_net'])
    print(f"Loaded checkpoint from epoch={checkpoint.get('epoch', '?')}, step={checkpoint.get('step', '?')}")

    # 2. 准备测试环境 (示例：从 DataLoader 取一个 batch)
    test_dataloader = load_dataloader()  # 假设在load_dataloader里支持train=False获取测试集
    test_iter = iter(test_dataloader)
    test_data = next(test_iter)

    # 假设和训练脚本中数据结构一致
    video_data = test_data["video_data"].to(device)
    kpt_uv = test_data["kpt_uv"].to(device)
    surgery_stage = test_data["surgery_stage"].to(device)
    label_data = test_data["label"].to(device)

    # 创建环境(仅示例对单个样本操作，您可根据需求扩展)
    env = SurgicalEnvironment(
        video=video_data[0],
        tips_kpts=kpt_uv[0],
        surgery_stage=surgery_stage[0],
        label=label_data[0],
        device=device
    )

    # 3. 执行推理并记录 action
    # ------------------------------------------------
    state = env.reset()  # 环境重置，获得初始状态
    total_reward = 0
    step_count = 0

    # 用于存储各步 action，后续绘图
    actions_history = []
    memory = Memory()
    while True:
        step_count += 1
        # 解包当前状态 (请根据您的环境逻辑调整)
        current_state, tip, current_stage, current_label, next_kpt, next_label = state

        # 利用 PPO 策略网络选择动作
        # action: Tensor(shape=[action_dim]), predicted_label: (如果有用到多任务预测)
        action, predicted_label = ppo.select_action(
            current_state, tip, current_stage, current_label, memory
        )

        # 记录动作 (转为 CPU numpy，便于可视化)
        print(action)
        actions_history.append(action.detach().cpu().numpy())

        # 与环境交互
        next_state, reward, done = env.step(action, predicted_label, next_kpt, next_label)

        # 累加奖励         se reward

        # 若到达回合结束，则退出循环
        if done:
            print(f"Episode done at step={step_count}, total_reward={total_reward:.4f}")
            break

        # 更新 state
        state = next_state

    # 4. 可视化 Action 结果
    # ------------------------------------------------
    actions_history = np.array(actions_history)  # (T, action_dim)
    num_steps, action_dim = actions_history.shape
    print(f"actions_history shape = {actions_history.shape}")

    # 这里以子图的方式可视化每一维 action
    plt.figure(figsize=(8, 6))
    for dim in range(action_dim):
        plt.plot(range(num_steps), actions_history[:, dim], label=f"Action Dim {dim}")

    plt.xlabel('Step')
    plt.ylabel('Action Value')
    plt.title('Action Trajectories Over Time')
    plt.legend(loc='best')
    plt.tight_layout()

    # 如果是在本地交互式环境，可直接展示
    plt.show()

    # 如果是在后台服务器，可选择保存图片
    # plt.savefig('action_trajectories.png')
    print("Visualization complete.")

if __name__ == "__main__":
    # 假设在训练脚本中保存的某个 .pth 文件路径
    # 例如: "./checkpoints/checkpoint_epoch_1_batch_0_step_10.pth"
    checkpoint_file = "../checkpoints_train/checkpoint_epoch_1_batch_0_step_320.pth"
    test_and_visualize_actions(checkpoint_file)
