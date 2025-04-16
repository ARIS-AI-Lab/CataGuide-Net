from itertools import count

import torch
import numpy as np
import random
from collections import Counter


class TestTimeClasses:
    def __init__(self, model, device='cpu', sample_times=4):
        self.model = model
        self.device = device
        self.sample_times = sample_times

    def augment(self, video):

        total_frames = video.shape[0]
        sampled_videos = []

        for _ in range(self.sample_times):
            if total_frames < 64:
                indices = np.linspace(0, total_frames - 1, 64, dtype=int)
            else:
                indices = sorted(np.random.choice(total_frames, 64, replace=False))
                # indices = np.random.choice(range(0, total_frames), size=64, replace=False)
                # start = random.randint(0, total_frames - 64)
                # indices = np.arange(start, start + 64)

            # 采样对应帧
            sampled_video = torch.tensor(video[indices], dtype=torch.float32).unsqueeze(0).to(self.device)
            # print(sampled_video.shape)
            # exit(0)
            sampled_videos.append(sampled_video)

        return sampled_videos

    def predict(self, video):
        video_augmented = self.augment(video)
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for i in range(self.sample_times):
                # print(video_augmented[i].shape)
                # exit(0)
                prediction = self.model(video_augmented[i].to(self.device))
                prediction = torch.argmax(prediction, dim=-1)
                predictions.append(prediction)
        # print(predictions)
        predictions = [t.item() for t in predictions]
        counter = Counter(predictions)
        most_common = counter.most_common(1)[0][0]

        return most_common
