import numpy as np
import av
import random
import torch
import os
import glob
from torchvision.io import VideoReader
import pandas as pd
from config_v import params
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import torch.nn.functional as F_n
from preprocessing.video_check import load_video


class CustomImageDataset(Dataset):
    def __init__(self, transform=None, mode='train'):
        """
        :param transform: data augmentation
        :param mode: mode ('train', 'val', 'test')
        """
        self.transform = transform
        self.mode = mode
        self.data_dir = params['input_pipeline_params']['npy_dir'] + self.mode

        # 假设所有的 .npy 文件路径已经包含类别信息
        self.video_files = [os.path.join(self.data_dir, i) for i in os.listdir(self.data_dir) if i.endswith('.npy')]
        # print(len(self.video_files))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video_class = os.path.basename(video_path).split('_')[1]

        # 确保类别信息有效
        if video_class == '' or int(video_class) > 10:
            raise ValueError(f"Invalid video class: {video_class}")

        video_frames = self.load_video_frames(video_path)
        video_tensor = self.transform_frames_to_tensor(video_frames)

        label = torch.tensor(int(video_class) - 1)  # video class is from 1 to 10
        # label = F_n.one_hot(label, num_classes=10)
        # print(video_tensor.shape)
        # exit(0)

        return video_tensor, label

    def load_video_frames(self, video_path):
        # 加载 .npy 文件
        try:
            video_data = np.load(video_path)  # shape (num_frames, H, W, C)
        except FileNotFoundError:
            raise ValueError(f"File {video_path} not found")

        # 按时间顺序随机选取 128 帧
        total_frames = video_data.shape[0]
        if total_frames < params['num_samples']:

            blank_frame = np.zeros_like(video_data[0])  # create empty frame
            additional_frames = [blank_frame] * (params['num_samples'] - total_frames)
            video_data = np.concatenate([video_data, additional_frames], axis=0)
            total_frames = len(video_data)

        indices = sorted(np.random.choice(total_frames, params['num_samples'], replace=False))

        selected_frames = video_data[indices]
        drop_prob = 0.1
        for i in range(len(selected_frames)):
            if np.random.rand() < drop_prob:
                selected_frames[i] = np.zeros_like(selected_frames[i])  # 用空白帧替换

        return selected_frames

    def transform_frames_to_tensor(self, frames):
        video_tensor = torch.tensor(frames).float() / 255.0  # 转换为 (num_frames, H, W, C) 并归一化
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # 转为 (num_frames, C, H, W)
        if self.transform:
            video_tensor = torch.stack([self.transform(frame) for frame in video_tensor], dim=0)
        return video_tensor


def load_dataloader(mode='train'):
    image_transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

    ])

    # CustomImageData include flip and rotation
    dataset = CustomImageDataset(transform=image_transform, mode=mode)
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True, pin_memory=True)
    return dataloader


if __name__ == '__main__':
    dataloader = load_dataloader(mode='train')
    count = 0
    for batch_idx, (video, classes) in enumerate(dataloader):
        # images = sample['image']
        # labels = sample['landmarks']
        print(f"Batch {batch_idx}:")
        print(f"Video size: {video.size()}")
        print(f"classes size: {classes.size()}: {classes}")
        # print(f"tips_kpts size: {tips_kpts.size()}")
        # print(f"Landmarks: {labels}")
        # break
        count += 1
        if count == 3:
            break
        # break