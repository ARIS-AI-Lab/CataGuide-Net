import pandas as pd
from config_v import params

def load_df(df_path=params['df_path']):
    df = pd.read_csv(df_path, header=None)
    return df.to_dict()[0]
    # print(df)
    # print(df.to_dict()[0][1].split(';')[-1])

def load_video_label_df(path=params['clip_label'], video_id=None):
    assert type(video_id) is int, f"Expected video_id need to be int, but got type {type(video_id).__name__}"

    df = pd.read_csv(path)
    # print(df)
    df_sampled =  df[df['VideoID'] == video_id]
    # df_dict = df_sampled.to_dict()
    # print(df_sampled)
    clip_dict = dict(zip(sorted(df_sampled['Phase']), sorted(df_sampled['FrameNo'])))
    # print(clip_dict)
    return clip_dict
    # return df[df['VideoID'] == video_id]


def find_key_by_value(input_value, data_dict):
    # 将字典的键值对按 value 排序
    # print(input_value)
    # print(data_dict)
    # sorted_items = sorted(data_dict.items(), key=lambda x: x[1])
    dis_list = []
    for value in data_dict.values():
        dis_list.append(int(value) - int(input_value))

        # if int(input_value) < int(value):
        #     print(key, value)
        #     # print(key-1)
        #     return key
        # else:
        #     return key
    # 如果输入值大于等于所有的 value，返回 None 或其他提示信息
    # return None
    result_dict = {key: value for key, value in enumerate(dis_list, start=1)}
    # print(result_dict)
    min_positive_key = max((key for key, value in result_dict.items() if value <= 0), key=lambda k: result_dict[k])
    return min_positive_key
    # print(min_positive_key)

if __name__ == '__main__':
    # file_path = r'..\data\phases.csv'
    file_path = r'..\data\merged.csv'
    # load_df(file_path)
    dict_data = load_video_label_df(video_id=280)
    # print(dict_data)
    key = find_key_by_value(9976, dict_data)
    print(key)