import pandas as pd
from config import params

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