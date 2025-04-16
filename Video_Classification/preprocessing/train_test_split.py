import os
from config import params
from sklearn.model_selection import train_test_split
import pandas as pd


def load_all_file(video_dataset_path):
    video_folders = [os.path.join(video_dataset_path, i) for i in os.listdir(video_dataset_path) if not i.endswith('.csv')]
    fake_folders_path = [os.path.join('/content/drive/MyDrive/Video_Classification_data', i) for i in os.listdir(video_dataset_path) if not i.endswith('.csv')]
    video_paths_list = []
    for i in range(len(video_folders)):
        videos_list = os.listdir(video_folders[i])
        for video in videos_list:
            video_path = os.path.join(fake_folders_path[i], video)
            # video_path = os.path.join(video_folder, video)
            video_paths_list.append(video_path)
    return video_paths_list


def split_test_train(dataset_list):
    train_list, test_list = train_test_split(dataset_list, test_size=0.2, random_state=0)
    test_pd = pd.DataFrame(test_list, columns=['test_video'])
    train_pd = pd.DataFrame(train_list, columns=['train_video'])
    total_pd = pd.concat([test_pd, train_pd], axis=1)
    # print(total_pd.head())
    return total_pd


if __name__ == '__main__':
    videos_folder_path = params['input_pipeline_params']['dataset_path']
    videos_paths = load_all_file(videos_folder_path)
    dataset_splited_idx = split_test_train(videos_paths)
    dataset_splited_idx.to_csv(params['input_pipeline_params']['dataset_instruction'], index=False)

    # print(len(train_list))
    # print(len(test_list))
    # print(test_list)