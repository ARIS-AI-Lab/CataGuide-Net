import pandas as pd
import torchvision.io as io
import os
from config_v import params
from tqdm import tqdm

def save_indices(annotations_file, mode, window_size, step_size, indices_file):
    """
    Calculates and saves the starting index of the sliding window
    参数：
    - annotations_file: indicates the path of the CSV file that contains the video path.
    - mode: indicates the data mode, such as 'train' or 'test'.
    - window_size: indicates the size of the sliding window.
    - step_size: indicates the step size of the sliding window.
    -indices_file: saves the file path of indices.
    """
    annotations_df = pd.read_csv(annotations_file, header=0)
    annotations_df = annotations_df[mode + '_video']

    indices = []
    if not os.path.exists(indices_file):
        os.makedirs(indices_file)
    else:
        print('Folder is exists')

    indices_file_path = os.path.join(indices_file, (str(mode) + '_' + str(step_size) + '_' + str(window_size) + '.txt'))
    if not os.path.exists(indices_file_path):
        print(f'right path: {os.path.dirname(indices_file_path)}')
    else:
        print(f'Not such Folder: {os.path.dirname(indices_file_path)}')

    for row in tqdm(annotations_df):
        if pd.isna(row):
            print("Skipping NaN row")
            continue
        video_path = row.replace('\cataract-101', '')
        # print(video_path.replace('\cataract-101', ''))
        # exit(0)
        video_frames, _ = io.read_video_timestamps(video_path, pts_unit='pts')
        num_frames = len(video_frames)
        # 计算该视频的所有滑动窗口起始索引
        for start_idx in range(0, num_frames, step_size):
            # indices.append((video_path, start_idx))
            indices.append((video_path, start_idx))

    # Save indices to a file

    with open(indices_file_path, 'w', encoding='utf-8') as f:
        for video_path, start_idx in indices:
            f.write(f"{video_path}\t{start_idx}\n")

    print(f"indices are saved to {indices_file} for a total of {len(indices)} records.")


def check_file_level(csv_file_path, video_num):
    df = pd.read_csv(csv_file_path, sep=';')

    # Find the corresponding line based on the given video_id
    row = df.loc[df['VideoID'] == video_num]

    # If no corresponding row is found, the ROW will be an empty DataFrame, which can be determined
    if row.empty:
        return None
    else:
        # Get the value of the Experience column
        # If VideoID is unique, there will be only one row, so you can get a single value with values[0]
        experience = row['Experience'].values[0]
        return experience

if __name__ == '__main__':
    annotations_file = params['input_pipeline_params']['dataset_instruction']
    save_indices(annotations_file, mode='test', window_size=params['window_size'], step_size=params['step_size'],
                 indices_file=params['indices_file'])