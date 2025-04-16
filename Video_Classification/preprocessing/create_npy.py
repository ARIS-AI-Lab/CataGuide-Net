from turtledemo.penrose import start

import av
import numpy as np
import os
import cv2
from config import params
from tqdm import tqdm

def decode_and_save_with_window(video_path, output_dir, target_size=(224, 224), num_samples=50, start_idx=0, window_size=256):
    """
    解码视频并保存为 .npy 文件，支持从指定帧开始并限制窗口范围内采样
    :param video_path: 输入视频路径
    :param output_dir: 输出文件夹路径
    :param target_size: 帧的目标大小 (height, width)
    :param num_samples: 保存的帧数
    :param start_idx: 起始帧索引
    :param window_size: 窗口大小（限制采样的帧范围）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)


    # 获取视频文件名作为 .npy 文件名
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    file_name = video_path.split('\\')[-2]
    # print(file_name)
    output_path = os.path.join(output_dir, f"{file_name + '_' + video_name + '_' + str(start_idx)}.npy")

    # 打开视频
    try:
        container = av.open(video_path)
        # print("Video opened successfully!")
    except av.AVError as e:
        print(f"Failed to open video: {e}")
    frames = []
    frame_idx = 0


    # 计算有效帧范围
    end_idx =  start_idx + window_size

    for frame in container.decode(video=0):
        # 跳过 start_idx 之前的帧
        if frame_idx < start_idx:
            frame_idx += 1
            continue

        # 超过 window_size 的帧停止处理
        if frame_idx >= end_idx:
            break

        # 转换帧为 RGB 格式
        img = frame.to_rgb().to_ndarray()  # (H, W, C)

        # 按目标大小调整帧大小
        img_resized = resize_and_pad(img, target_size)
        frames.append(img_resized)
        frame_idx += 1

    # 如果 frames 为空
    if not frames:
        raise ValueError(f"No frames available in the range start_idx={start_idx} to end_idx={end_idx} for {video_path}.")

    # 采样指定数量的帧
    if False:
        total_frames = len(frames)
        if total_frames >= num_samples:
            selected_indices = sorted(np.linspace(0, total_frames - 1, num_samples, dtype=int))
        else:
            selected_indices = sorted(np.random.choice(range(total_frames), size=num_samples, replace=True))
        sampled_frames = np.array([frames[i] for i in selected_indices])  # (num_samples, C, H, W)
    # sampled_frames = np.array([frames[i] for i in frames])  # (num_samples, C, H, W)

    # 保存为 .npy 文件
    sampled_frames = np.array(frames)
    # print(sampled_frames.shape)
    # exit(0)
    np.save(output_path, sampled_frames)
    # print(f"Saved {output_path}, shape: {sampled_frames.shape}")


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

def process_all_videos(input_dir, output_dir, target_size=(224, 224), num_samples=128, extensions=('.mp4', '.avi', '.mov')):
    """
    遍历输入目录下的所有视频，解码并保存为 .npy 文件
    :param input_dir: 输入视频目录
    :param output_dir: 输出 .npy 文件目录
    :param target_size: 帧的目标大小
    :param num_samples: 保存的帧数
    :param extensions: 需要处理的视频扩展名
    """
    indices_file = r'C:\Users\Charl\PycharmProjects\Video_Classification\data\test_128_256.txt'
    indices=[]

    with open(indices_file, 'r', encoding='utf-8') as f:
        for line in f:
            video_path, start_idx = line.strip().split('\t')
            start_idx = int(start_idx)
            indices.append((video_path, start_idx))

    # print(indices)
    # exit(0)
    for video_path, start_idx in tqdm(indices):
        try:
            # print('in')
            decode_and_save_with_window(video_path, output_dir, target_size, num_samples=128, start_idx=start_idx,
                                        window_size=256)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")


# 示例调用
if __name__ == "__main__":
    input_dir = params['input_pipeline_params']['dataset_path']  # 输入视频文件夹
    # output_dir = r"C:\Users\Charl\PycharmProjects\Video_Classification\npys_train"  # 输出 .npy 文件夹
    output_dir = r"D:\npys_test"  # 输出 .npy 文件夹

    process_all_videos(input_dir, output_dir)
