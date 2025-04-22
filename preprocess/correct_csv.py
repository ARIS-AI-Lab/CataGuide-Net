import pandas as pd
import os
from config import params


def correct_csv(df_path):
    df = pd.read_csv(df_path, header=None)
    # print(df)
    last_class_right, last_class_left = 0, 0
    df_right, df_left = pd.DataFrame(), pd.DataFrame()
    for i in range(len(df)):
        line = df.iloc[i]
        right_c, left_c = extract_kpt(line[0]), extract_kpt(line[1])

        if right_c == last_class_left or left_c == last_class_right:
            df_right = pd.concat([df_right, pd.Series([line[1]])], axis=0)
            df_left = pd.concat([df_left, pd.Series([line[0]])], axis=0)
        else:
            df_right = pd.concat([df_right, pd.Series([line[0]])], axis=0)
            df_left = pd.concat([df_left, pd.Series([line[1]])], axis=0)
            last_class_right, last_class_left = right_c, left_c

        # print(right_c, left_c)

        # exit(0)
    df_after = pd.concat([df_right, df_left], axis=1)
    df_after.columns = [0, 1]
    # print(df_after.columns)
    df_after.to_csv(r'C:\Users\Charl\Video_Classification\trajectory_RL\test_2.csv', header=None, index=None)
    # print(df)
    # exit(0)
    return df_after

def extract_kpt(df_col):
    # print(type(df_col))
    classes = df_col.split(' ')[-1]
    classes = int(float(classes))
    # print(split_columns)
    # x_points = split_columns[0].astype(float)  # 提取第一列并转换为浮点数
    # y_points = split_columns[1].astype(float)
    # classes = split_columns[2].astype(float).astype(int)

    # kpts = pd.concat([x_points, y_points], axis=1)
    return classes
def scan_csvs(folder_path):

    csv_path_list = [os.path.join(folder_path, i) for i in os.listdir(folder_path)]
    for i in csv_path_list:
        correct_csv(i)
    # print(csv_path_list)

if __name__ == '__main__':
    folder_path = params['pipline_params']['train_trajectory_path']
    scan_csvs(folder_path)


