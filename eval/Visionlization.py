import torch
import os
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from network.gail import LightweightTSMGenerator
from network.diffusion import DiffusionLightweightTSMGenerator
from network.dis_net import LightweightDiscriminator
from data.dataloader import load_dataloader  # 确保此函数能够正确加载测试集
from src.utils.loss import TrajectoryLoss
from src.utils.utils import load_video_label_df
from config import params
import cv2

def sliding_window_sampling(video_path, window_size=256, step_size=256, target_size=(224, 224)):
    """
    滑动窗口采样视频帧并应用图像变换。
    :param video_path: 视频文件路径
    :param window_size: 每个窗口的帧数
    :param step_size: 窗口滑动的步长
    :param target_size: 每帧调整后的大小 (height, width)
    :return: List[torch.Tensor] 每个窗口的帧张量列表，形状为 [num_windows, window_size, C, H, W]
    """
    # 定义图像变换
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    sampled_windows = []  # 存储采样的窗口

    # 遍历所有滑动窗口
    for start_frame in tqdm(range(0, frame_count - window_size + 1, step_size)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 设置视频读取的起点
        window_frames = []

        for _ in range(window_size):
            ret, frame = cap.read()
            if not ret:
                break
            # 调整帧大小并转换为 RGB
            frame_resized = resize_and_pad(frame, target_size)
            # print(frame_resized.shape)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
            # print(frame_rgb.shape)
            # 应用图像变换
            # exit(0)
            transformed_frame = image_transform(frame_rgb)
            # print(transformed_frame.shape)
            # exit(0)
            window_frames.append(transformed_frame)
        # print(np.array(window_frames).shape)
        # 确保窗口包含完整帧数
        if len(window_frames) == window_size:
            sampled_windows.append(torch.stack(window_frames))  # 转为张量并存储
    print('Video Process Finished, Ready to Show')
    cap.release()
    return sampled_windows



def resize_and_pad(frame, target_size):
    """
    按比例缩放图像并填充到目标大小 (使用 NumPy 和 OpenCV)
    :param frame: 输入图像，形状为 (H, W, C)
    :param target_size: 目标大小 (target_height, target_width)
    :return: 填充后的图像，形状为 (C, target_height, target_width)
    """
    # 获取原始大小
    original_height, original_width = frame.shape[:2]
    target_height, target_width = target_size

    # 计算缩放比例
    scale = min(target_height / original_height, target_width / original_width)

    # 按比例缩放
    new_height, new_width = int(original_height * scale), int(original_width * scale)
    resized_frame = cv2.resize(frame, (new_width, new_height))  # (H, W, C)

    # 计算填充尺寸
    pad_h = target_height - new_height
    pad_w = target_width - new_width
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # 填充图像
    padded_frame = np.pad(
        resized_frame,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),  # 对 (H, W, C) 填充
        mode="constant",
        constant_values=0,  # 填充值为 0，可根据需要更改
    )

    return padded_frame


def denormalize(tensor, mean, std):
    """
    将归一化的张量或数组反归一化。
    :param tensor: 已归一化的图像张量 (C, H, W) 或 NumPy 数组
    :param mean: 每个通道的均值 (list 或 tuple)
    :param std: 每个通道的标准差 (list 或 tuple)
    :return: 反归一化后的张量或数组 (C, H, W)
    """
    if isinstance(tensor, np.ndarray):  # 如果输入是 NumPy 数组
        tensor = torch.tensor(tensor)

    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


def test(checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_path = r'C:\Users\Charl\Downloads\cataract-1k\videos\case_5309.mp4'
    output_path = r"C:\Users\Charl\PycharmProjects\Video_Classification\output_video_with_predictions.mp4"
    video_num = os.path.basename(video_path).split('_')[-1].strip('.mp4')
    video_stage_flag = load_video_label_df(video_id=int(video_num))
    load_video_label_df(video_id=int(video_num))
    # 加载模型
    # dis = LightweightDiscriminator(stage_dim=10, num_classes=1).to(device)
    gen = DiffusionLightweightTSMGenerator().to(device)
    dis = LightweightDiscriminator(stage_dim=10, num_classes=1).to(device)
    # 加载训练好的权重
    if os.path.exists(checkpoint):
        checkpoint = torch.load(checkpoint, map_location=device)
        # checkpoint = torch.load(r'C:\Users\Charl\Video_Classification\trajectory_RL\checkpoints_pretrain_bc\models_step_17000.pth')
        gen.load_state_dict(checkpoint["generator_state_dict"])
        dis.load_state_dict(checkpoint["discriminator_state_dict"])
        print(f"Model loaded successfully ")
    else:
        print(f"Checkpoint not found at {checkpoint}")
        return

    # 设置模型为评估模式
    gen.eval()
    # dis.eval()

    # 载入测试数据
    test_dataloader = load_dataloader()  # 需要修改 `load_dataloader` 以支持 test 模式

    criterion = TrajectoryLoss(weight_tra=1.0, weight_level=0.8)
    total_d_loss, total_g_loss = 0, 0
    total_samples = 0
    counts = 0
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义输出视频的编码器和文件名
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    sampled_windows = sliding_window_sampling(video_path)
    with torch.no_grad():
        for i in tqdm(range(len(sampled_windows))):
            # 当前窗口的帧

            sampled_windows_temp = np.array(sampled_windows[i])
            counts += 1
            videos_batch = sampled_windows_temp

            surgery_stage_batch = video_stage_flag

            # 生成轨迹
            tra_policy, policy_label = gen(videos_batch, surgery_stage_batch)

            # real_labels = torch.ones((videos_batch.size(0), 1)).to(device)
            # fake_labels = torch.zeros((videos_batch.size(0), 1)).to(device)
            # fake_level_labels = torch.zeros_like(surgery_level_batch).to(device)

            # 计算判别器损失
            # real_output, real_level_output = dis(tips_kpts_batch, surgery_stage_batch, label_batch)
            # d_loss_real = criterion(real_output, real_level_output, real_labels, surgery_level_batch)
            nor_tra_policy = torch.clamp(tra_policy, 0, 1)
            # print(nor_tra_policy.shape)
            # print(torch.max(nor_tra_policy, -1))
            # print(torch.min(nor_tra_policy, -1))
            loss = abs(nor_tra_policy - tips_kpts_batch).mean()
            total_g_loss += loss.item()
            # if counts == 500:
            #     break
            # print(torch.clamp(tra_policy, 0, 1))
            # print('_____________'*10)
            # print(tips_kpts_batch)
            # print(loss)
            # exit(0)

            # fake_output, fake_level_output = dis(tra_policy, surgery_stage_batch, policy_label)
            # d_loss_fake = criterion(fake_output, fake_level_output, fake_labels, fake_level_labels)

            # d_loss = d_loss_real + d_loss_fake
            # total_d_loss += d_loss.item()

            # 计算生成器损失
            # fake_output, fake_level_output = dis(tra_policy, surgery_stage_batch, policy_label)
            # g_loss = criterion(fake_output, fake_level_output, real_labels, surgery_level_batch)
            # total_g_loss += g_loss.item()

            # total_samples += videos_batch.size(0)

            # print(f"Batch {batch_idx + 1}/{len(test_dataloader)} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # avg_d_loss = total_d_loss / total_samples
    avg_g_loss = total_g_loss / counts
    # print(f"\nTest Results - Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}")
    print(f"\nTest Results - Avg G Loss: {avg_g_loss:.4f}")

if __name__ == "__main__":
    ckpts_folder = r'C:\Users\Charl\Video_Classification\trajectory_RL\checkpoints_train\models_step_32000.pth'
    test(ckpts_folder)