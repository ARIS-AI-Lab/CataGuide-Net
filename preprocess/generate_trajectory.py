import sys
import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
sys.path.append(r'C:\Users\Charl\Video_Classification\Cataract_Pro')

from export import load_video

def main(npy_path):
    trajectory_folder = r"D:\trajectory_folder_test"
    if not os.path.exists(trajectory_folder):
        os.mkdir(trajectory_folder)
    npy_files = os.listdir(npy_path)
    for npy_file in tqdm(npy_files):
        npy_file_path = os.path.join(npy_path, npy_file)
        video_data = np.load(npy_file_path)
        df = load_video(video_data, target_size=(640, 640))
        # print(npy_file.strip('.npy'))
        file_name = npy_file.strip('.npy') + '.csv'
        # print(file_name)
        # print(os.path.join(trajectory_folder, npy_file))
        # exit(0)
        # df.to_parquet(os.path.join(trajectory_folder, file_name), index=False)
        df.to_csv(os.path.join(trajectory_folder, file_name), index=False, header=None)
        # print(df)



if __name__ == '__main__':
    # npy_folder = r'C:\Users\Charl\PycharmProjects\Video_Classification\npys_train'
    npy_folder = r"D:\npys_test"
    main(npy_folder)