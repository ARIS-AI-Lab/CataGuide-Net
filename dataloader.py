import os
import numpy as np
import pandas as pd
from tensorboard.compat.tensorflow_stub.dtypes import int64
from torch import dtype
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from config import params
from preprocess.correct_csv import correct_csv
from preprocess.Rename_file import load_df
import torch.nn.functional as F


class gt_dataset(Dataset):
    def __init__(self, transform=None):

        self.trajectory_path = params['pipline_params']['train_trajectory_path']
        self.directory = params['pipline_params']['npy_train_path']

        # self.trajectory_list_path = [os.path.join(self.trajectory_path, i) for i in os.listdir(self.trajectory_path)]
        self.file_list_path = [os.path.join(self.directory, i) for i in os.listdir(self.directory) if not i.endswith('.md')]
        self.num_sample = params['num_sample']
        self.transform = transform

    def __len__(self):
        """
        :return number of videos
        """
        return len(self.file_list_path)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本

        Args:
            idx (int): 索引值。

        Returns:
            sample (torch.Tensor): 处理后的样本。
        """
        file_path = self.file_list_path[idx]
        file_name = os.path.basename(file_path).strip('.npy')
        surgery_level = file_name.split('_')[-1].split('L')[-1]
        surgery_level = int(surgery_level) - 1
        # print(surgery_level)

        surgery_stage = file_name.split('_')[1]
        surgery_stage = int(surgery_stage) - 1

        surgery_trajectory_path = os.path.join(self.trajectory_path, file_name + '.csv')

        current_df = self.load_df(surgery_trajectory_path)
        kpt_right, kpt_left = current_df[0], current_df[1]
        kpt_r_uv, kpt_r_c = self.extract_kpt(kpt_right)
        kpt_l_uv, kpt_l_c = self.extract_kpt(kpt_left)
        # num_classes = params['num_classes']
        kpt_r_c = np.vectorize(params['class_mapping'].get)(kpt_r_c)
        kpt_l_c = np.vectorize(params['class_mapping'].get)(kpt_l_c)

        kpt_uv = np.hstack((kpt_r_uv, kpt_l_uv))
        label = np.column_stack((kpt_r_c, kpt_l_c))

        """
        shape : (256, C, H, W, (x1, y1, x2, y2), (c1, c2))
        :param 
        """

        # exit(0)
        # exit(0)
        try:
            video_data = np.load(file_path)  # npy's shape is (num_frames, H, W, C)
        except FileNotFoundError:
            raise ValueError(f"File {file_path} not found")

        num_frames = video_data.shape[0]

        if num_frames < 256:
            # 补齐 video_data
            pad_frames = np.zeros((256 - num_frames, *video_data.shape[1:]), dtype=video_data.dtype)
            video_data = np.concatenate([video_data, pad_frames], axis=0)

            # 补齐 kpt_uv
            pad_kpt_uv = np.zeros((256 - num_frames, kpt_uv.shape[1]), dtype=kpt_uv.dtype)
            kpt_uv = np.concatenate([kpt_uv, pad_kpt_uv], axis=0)

            # 补齐 label
            pad_label = np.zeros((256 - num_frames, label.shape[1]), dtype=label.dtype)
            label = np.concatenate([label, pad_label], axis=0)

        label_one_hot = np.eye(params['num_classes'])[label]
        label_one_hot_torch = torch.from_numpy(label_one_hot)
        kpt_uv_torch = torch.from_numpy(kpt_uv)
        # '''
        # sampled_video_data = self.sample_frames(video_data, self.num_sample)
        sampled_video_data = self.sample_key_frames(video_data, threshold=10.0, num_sample=self.num_sample)

        video_tensor = self.transform_frames_to_tensor(sampled_video_data)
        # '''
        # video_tensor = self.transform_frames_to_tensor(video_data)

        return {
            "video_data": video_tensor,
            "kpt_uv": kpt_uv_torch,
            "label": label_one_hot_torch,
            "surgery_level": torch.tensor(surgery_level, dtype=torch.int64),

            "surgery_stage": F.one_hot(torch.tensor(surgery_stage, dtype=torch.int64),
                                       num_classes=params['total_stage']),
        }
        # '''
        # exit(0)
        # return sample

    def sample_frames(self, video_data, num_sample):
        """
        对 video_data 进行均匀采样，同时确保第一帧和最后一帧被选中
        """
        num_frames = video_data.shape[0]  # 当前片段的总帧数

        # 1. 确保第一帧和最后一帧
        first_frame = video_data[0:1]  # 取第 1 帧
        last_frame = video_data[-1:]  # 取最后 1 帧

        if num_frames <= 2:  # 如果只有 1-2 帧，直接复制填充
            sampled_frames = np.concatenate([first_frame] * (num_sample - 1) + [last_frame], axis=0)
        else:
            # 2. 均匀采样 `num_sample - 2` 帧（去掉第一帧和最后一帧后均匀采样）
            indices = np.linspace(1, num_frames - 2, num=num_sample - 2, dtype=int)
            sampled_middle_frames = video_data[indices]

            # 3. 组合：第一帧 + 采样帧 + 最后一帧
            sampled_frames = np.concatenate([first_frame, sampled_middle_frames, last_frame], axis=0)

        return sampled_frames

    def sample_key_frames(self, video_data, threshold=10.0, num_sample=256):
        """
        采用帧差法采样关键帧：
          1. 计算相邻帧之间的均值绝对差（可以根据实际情况采用其他差异度量方式）
          2. 根据阈值选取变化较大的帧
          3. 为保证边界信息，始终保留第一帧和最后一帧
          4. 如果选出的帧数超过或不足 num_sample，则做相应的处理（例如均匀采样或填充）

        Args:
            video_data (numpy.ndarray): 原始视频数据，形状为 (num_frames, H, W, C)
            threshold (float): 差异度阈值，只有大于该值的帧才被视为关键帧
            num_sample (int): 目标采样帧数

        Returns:
            sampled_frames (numpy.ndarray): 采样后的关键帧，形状为 (num_sample, H, W, C)
        """
        num_frames = video_data.shape[0]

        # 计算相邻帧之间的均值绝对差（全图像像素均值差）
        # diff_scores 的 shape 为 (num_frames - 1,)
        diff_scores = np.mean(np.abs(video_data[1:].astype(np.float32) - video_data[:-1].astype(np.float32)),
                              axis=(1, 2, 3))

        # 找到变化超过阈值的帧索引（注意：diff_scores[i]表示帧 i 和帧 i+1 的差异）
        key_indices = np.where(diff_scores > threshold)[0] + 1  # 加 1 表示后一个帧

        # 始终保留第一帧和最后一帧
        key_indices = np.concatenate(([0], key_indices, [num_frames - 1]))

        # 去除重复并排序
        key_indices = np.unique(key_indices)
        # print(len(key_indices))

        # 如果选出的关键帧数目多于目标，则采用均匀采样选取 num_sample 个关键帧
        if len(key_indices) > num_sample:
            # 这里采用均匀采样关键帧索引
            indices = np.linspace(0, len(key_indices) - 1, num_sample, dtype=int)
            key_indices = key_indices[indices]
        elif len(key_indices) < num_sample:
            # 如果不足目标数量，可以采用线性插值或重复某些帧进行补充
            # 这里简单地使用均匀采样补充：在原始视频中均匀采样 num_sample 帧
            indices = np.linspace(0, num_frames - 1, num_sample, dtype=int)
            key_indices = indices

        # 根据 key_indices 采样关键帧
        sampled_frames = video_data[key_indices]
        return sampled_frames

    def transform_frames_to_tensor(self, frames):
        video_tensor = torch.tensor(frames).float() / 255.0  # 转换为 (num_frames, H, W, C) 并归一化
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # 转为 (num_frames, C, H, W)
        # video_resized = F.interpolate(video_tensor, size=(112, 112), mode='bilinear')

        if self.transform:
            # video_tensor = torch.stack([self.transform(frame) for frame in video_resized], dim=0)
            video_tensor = torch.stack([self.transform(frame) for frame in video_tensor], dim=0)
        return video_tensor


    def extract_kpt(self, df_col):
        split_columns = df_col.str.split(' ', expand=True)  # 按空格分割为多列
        x_points = split_columns[0].astype(float)  # 提取第一列并转换为浮点数
        y_points = split_columns[1].astype(float)
        classes = split_columns[2].astype(float).astype(int)

        kpts = pd.concat([x_points, y_points], axis=1)

        return kpts, classes

    def load_df(self, df_path):
        df = correct_csv(df_path)
        # print(df.head(10))
        # exit(0)
        return df


def load_dataloader():
    image_transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

    ])

    # 自定义的 CustomTransform 包含旋转和水平翻转的处理
    dataset = gt_dataset(transform=image_transform)
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True, pin_memory=True)
    return dataloader

# 示例用法


if __name__ == '__main__':
    dataloader = load_dataloader()
    count = 0
    for batch_idx, gt_data in enumerate(dataloader):
        # images = sample['image']
        # labels = sample['landmarks']
        print(f"Batch {batch_idx}:")
        print(f"Video size: {gt_data['video_data'].shape}")
        print(f"classes size: {gt_data['label'].shape}")
        print(f"tips_kpts size: {gt_data['kpt_uv'].shape}")
        print(f"suurgery_level size: {gt_data['surgery_level'].shape}")
        print(f"surgery_stage size: {gt_data['surgery_stage'].shape}")
        # print(f"Landmarks: {labels}")
        # break
        count += 1
        if count == 1:
            break
        # break  # 仅显示一个批次
