import os
import json
import numpy as np
from config import params
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from preprocessing.pre_check.k_check import find_tip_kpt
import torchvision.transforms as transforms
from src.utils.image_utils import resize_and_pad, CustomTransform
import torch.nn.functional as F


class CustomImageDataset(Dataset):
    def __init__(self, files_dir, transform=None):
        """

        :param annotations_file:  description, tags, size, objects
        :param img_dir: contains images matches json
        :param
        """
        self.annotations_file = os.path.join(files_dir, 'annotations')

        self.img_dir = [os.path.join(os.path.join(files_dir, 'images'), f)
                        for f in os.listdir(os.path.join(files_dir, 'images'))
                        if f.endswith('jpg') or f.endswith('png')]
        self.class_dict = params['model_param']['tool_and_eyes_class']
        self.transform = transform

    def __len__(self):
        # 返回数据集的样本数量
        return len(self.img_dir)

    def __getitem__(self, idx):
        """
        根据索引返回图像和标注
        :param idx: sample index
        :return: 预处理后的图像及关键点
        """
        image_path = self.img_dir[idx]
        anno_name = image_path.split('\\')[-1] + '.json'
        annotation_path = self.annotations_file + '/' + anno_name
        image = Image.open(image_path).convert("RGB")

        labels = {}
        with open(annotation_path, 'r') as f:
            content = json.load(f)

        for obj in content['objects']:
            classTitle = obj['classTitle']
            class_id = self.class_dict[classTitle]

            if not params['detect_eyes']:
                if ('pupil' not in classTitle.lower()) and ('cornea' not in classTitle.lower()):
                    kpts = np.array(obj['points']['exterior'])
                    class_dict = {class_id: np.array(kpts)}
                    labels.update(class_dict)
            else:
                kpts = np.array(obj['points']['exterior'])
                class_dict = {class_id: np.array(kpts)}
                labels.update(class_dict)

        image, labels = resize_and_pad(image, labels, target_size=params['input_pipeline_params']['image_size'])
        # print(labels.keys())

        mask = Image.new("L", image.size, 0)  # create an all zero matrix

        # unique_values = np.unique(mask)
        #
        # # 打印不同的像素值个数
        # print(f"有 {len(unique_values)} 种不同的像素值。")
        tool_classes = [class_id for class_id in labels.keys() if class_id < 11]
        pupil_cornea_classes = [class_id for class_id in labels.keys() if class_id >= 11]
        # Forceps_classes = [class_id for class_id in labels.keys() if class_id == 11]
        '''
        for class_id, value in labels.items():
            print(class_id)
            ImageDraw.Draw(mask).polygon([tuple(p) for p in value], outline=class_id, fill=class_id)
        '''
        tips_landmarks = []
        for class_id in pupil_cornea_classes:

            ImageDraw.Draw(mask).polygon([tuple(p) for p in labels[class_id]], outline=class_id, fill=class_id)

        # 再绘制手术工具的掩码
        for class_id in tool_classes:
            arr = find_tip_kpt(labels[class_id]) / (mask.width, mask.height)
            sorted_arr = arr[arr[:, 1].argsort()]
            min_y_points = sorted_arr[:2]
            center = np.array([0.5, 0.5])
            distances = np.linalg.norm(min_y_points - center, axis=1)
            closest_index = np.argmin(distances)
            tips = arr[closest_index]
            tips_be = tips.copy()
            tips = np.append(tips, [class_id])

            if class_id == 2 or class_id == 7:
                tips = -np.ones_like(tips_be)
                tips = np.append(tips, [class_id])


            # exit(0)
            tips_landmarks.append(tips)
            ImageDraw.Draw(mask).polygon([tuple(p) for p in labels[class_id]], outline=class_id, fill=class_id)
        # print(tips_landmarks)
        tips_landmarks = np.reshape(np.array(tips_landmarks), (-1, 3))

        tips_landmarks = pad_tips_landmark(tips_landmarks)

        # mask = mask.resize((mask.width // 4, mask.height // 4), Image.NEAREST)


        if self.transform:
            image, mask, tips_landmarks_uv = self.transform(image, mask, tips_landmarks[:, :2])
            labels = tips_landmarks[:, 2]
            # print(labels.shape)
            landmarks_class = torch.tensor(labels, dtype=torch.float32)
            # zero_mask = (tips_landmarks_uv == 0).all(dim=1)  # 得到 (n,) 的布尔张量，如果 (x, y) 全为 0，则为 True

            # 将 landmark_class 中对应 zero_mask 为 True 的位置设为 0
            # landmarks_class = torch.where(zero_mask, torch.tensor(0.0, device=landmarks_class.device), landmarks_class)
            # landmarks_class[torch.all(labels == 0, dim=1)] = 0
            # landmarks_class = torch.tensor(labels, dtype=torch.float32)
            # if torch.all(tips_landmarks_uv == 0):
            #     # 如果全为 0，生成全 0 标签
            #     landmarks_class = torch.zeros_like(landmarks_class)
            tips_landmarks_uv = torch.clamp(tips_landmarks_uv, min=-1, max=1)

            tips_landmarks = torch.cat([tips_landmarks_uv, landmarks_class.unsqueeze(1)], dim=1)

        # print(tips_landmarks)
        # print('****' * 20)
        return image, mask, tips_landmarks


def load_dataloader():
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

    ])

    # 自定义的 CustomTransform 包含旋转和水平翻转的处理
    custom_transform = CustomTransform(image_transform=image_transform, augmentation_prob=0.45, rotation_angle=60)
    # custom_transform = CustomTransform_test(image_transform=image_transform, rotation_angle=30, flip_prob=0.5)
    folder_path = params['input_pipeline_params']['dataset_path']

    custom_dataset = CustomImageDataset(files_dir=folder_path, transform=custom_transform,
                                        )

    dataloader = DataLoader(custom_dataset, batch_size=params['batch_size'], shuffle=True, pin_memory=True)
    return dataloader


def pad_tips_landmark(tips_landmark, max_tool_class=params['model_param']['max_kpts']):
    target_length = max_tool_class
    # 检查当前长度
    current_length = tips_landmark.shape[0]

    # 如果当前长度小于目标长度，进行填充
    if current_length < target_length:
        padding_length = target_length - current_length
        # 创建填充值，shape 为 (padding_length, 3)
        padding = np.array([[-1, -1, 0]] * padding_length)
        # 将原始数组和填充值拼接起来
        padded_landmark = np.vstack((tips_landmark, padding))
    elif current_length > target_length:
        # 如果当前长度超过目标长度，则截取前 target_length 个元素
        padded_landmark = tips_landmark[:target_length]
    else:
        # 如果当前长度正好等于目标长度，直接返回
        padded_landmark = tips_landmark

    return padded_landmark

# 示例用法


if __name__ == '__main__':
    dataloader = load_dataloader()

    for batch_idx, (image, mask, tips_kpts) in enumerate(dataloader):
        # images = sample['image']
        # labels = sample['landmarks']
        print(f"Batch {batch_idx}:")
        print(f"Image size: {image.size()}")
        print(f"Mask size: {mask.size()}")
        print(f"tips_kpts size: {tips_kpts.size()}: {tips_kpts}")
        # print(f"Landmarks: {labels}")
        break  # 仅显示一个批次