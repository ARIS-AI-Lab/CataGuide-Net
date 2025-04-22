import pandas as pd
import os
from tqdm import tqdm


def load_df(file_name):
    df = pd.read_csv(file_name, sep=';')
    return df

def match_npy(file_path, npy_folder_path):
    df = load_df(file_path)
    npy_files = os.listdir(npy_folder_path)
    for npy_file in tqdm(npy_files):
        old_file_name = os.path.join(npy_folder_path, npy_file)
        video_id = npy_file.split('_')[0]
        # print(df['VideoID']==video_id)
        # exit(0)
        current_skill_level = df[df['VideoID']==int(video_id)]['Experience'].values[0]
        new_name = npy_file.strip('.npy') + 'L' + str(current_skill_level) + '.npy'
        new_path = os.path.join(npy_folder_path, new_name)
        os.rename(old_file_name, new_path)
        # print(old_file_name)
        # print(new_path)
        # exit(0)



if __name__ == '__main__':
    file_path = r"D:\cataract-101\cataract-101\videos.csv"
    npy_path = r'C:\Users\Charl\PycharmProjects\Video_Classification\npys_train'
    # match_npy(file_path, npy_path)