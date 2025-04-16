import numpy as np
import av
import random
import torch
import os
from src.utils import metrics
from torchvision.io import VideoReader
from tqdm import tqdm
from config_v import params
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from model.TimeSformer import TimeSformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class CustomImageDataset(Dataset):
    def __init__(self, transform=None, mode='train'):
        """
        :param transform: Data Augmentation
        :param mode: mode ('train', 'val', 'test')
        """
        self.transform = transform
        self.mode = mode
        self.data_dir = params['input_pipeline_params']['npy_test_dir'] + self.mode


        self.video_files = [os.path.join(self.data_dir, i) for i in os.listdir(self.data_dir)]
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
        # load .npy 文件
        try:
            video_data = np.load(video_path)  # 假设形状为 (num_frames, H, W, C)
        except FileNotFoundError:
            raise ValueError(f"File {video_path} not found")

        total_frames = video_data.shape[0]
        if total_frames < params['num_samples']:
            blank_frame = np.zeros_like(video_data[0])  # 创建空白帧
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
        video_tensor = torch.tensor(frames).float() / 255.0
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        if self.transform:
            video_tensor = torch.stack([self.transform(frame) for frame in video_tensor], dim=0)
        return video_tensor


def load_dataloader_eval(mode='test'):
    image_transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    # CustomTransform include flip and rotation
    dataset = CustomImageDataset(transform=image_transform, mode=mode)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)
    return dataloader



def load_model(model_path):
    model = TimeSformer(
        dim=128,
        image_size=224,
        patch_size=28,
        num_frames=64,
        num_classes=10,
        depth=6,
        heads=4,
        dim_head=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        # rotary_emb=False,
    ).to(device)
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()

    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
    #                    k in model_dict and "layer" in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('model loaded from {}'.format(os.path.basename(model_path)))

    return model


def evaluate(model, is_plot=False):
    dataloader = load_dataloader_eval(mode='test')
    acc_tr = 0.0
    batch_num = 0
    metric_fn = metrics.map_charades
    model.eval()
    acc_list = []
    all_true = []
    all_pred = []
    for idx_batch, (x, y_true) in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_num = idx_batch + 1

        x, y_true = x.to(device), y_true.to(device)
        with torch.no_grad():
            y_pred = model(x)
        # print(x.shape)
        # print(y_pred)
        # print(y_pred.shape)
        # print(torch.argmax(y_pred, axis=-1))
        # if idx_batch == 3:
        #     exit(0)
        y_true = y_true.long()
        y_true = y_true.cpu().numpy().astype(np.int32)
        y_pred = y_pred.cpu().detach().numpy()
        # y_pred_arg = np.argmax(y_pred, axis=-1)
        all_true.append(y_true)
        all_pred.append(y_pred)


        acc_b = metric_fn(y_true, y_pred)
        acc_list.append(acc_b)
        acc_tr += acc_b

    acc_b_tr = acc_tr / float(batch_num)
    # print("Classification Report:")
    # print(classification_report(all_true, all_pred, zero_division=0))

    print(f'The map of test dataset is {acc_b_tr * 100:.2f}%')
    if is_plot:
        import matplotlib.pyplot as plt
        acc_list = sorted(acc_list)
        plt.plot(acc_list)

        plt.show()



if __name__ == '__main__':
    # model_path = r'C:\Users\Charl\PycharmProjects\Video_Classification\checkpoint_train\epoch_60.pth'
    # model = load_model(model_path)
    # evaluate(model, is_plot=True)

    # '''
    model_path = r"C:\Users\Charl\PycharmProjects\Video_Classification\checkpoint_train"
    model_list = [os.path.join(model_path, i) for i in os.listdir(model_path) if i.endswith('.pth')]
    for model in model_list:
        model = load_model(model)
        evaluate(model, is_plot=False)
    # '''