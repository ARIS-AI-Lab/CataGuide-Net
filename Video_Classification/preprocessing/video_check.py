import torch
import torchvision.io as io

def load_video(video_path, start_pts=0, end_pts=None, pts_unit='pts'):
    """
    从指定路径加载视频，并返回视频帧数据和音频信息。

    参数:
        video_path (str): 视频文件的路径。
        start_pts (int): 开始时间的时间戳。
        end_pts (int): 结束时间的时间戳。
        pts_unit (str): 时间戳的单位，可以是 'pts'（视频的原始时间戳）或 'sec'（秒）。

    返回:
        video_frames (torch.Tensor): 视频帧，形状为 (num_frames, H, W, C)。
        audio_samples (torch.Tensor): 音频样本。
    """
    # 使用 torchvision.io.read_video 加载视频
    video_frames, audio_samples, info = io.read_video(video_path, start_pts=start_pts, end_pts=end_pts, pts_unit=pts_unit)
    video_pts, _ = io.read_video_timestamps(video_path, pts_unit='sec')
    return video_frames, audio_samples, video_pts

# 示例用法
if __name__ == '__main__':

    video_path = r"C:\Users\Charl\Downloads\cataract-101\Video_Classification\934\10_1265.mp4"  # 替换为你的视频文件路径
    video_frames, audio_samples = load_video(video_path)

    # 打印视频帧信息
    print("视频帧形状:", video_frames.shape)  # (num_frames, H, W, C)
    print("音频样本形状:", audio_samples.shape)  # (num_audio_samples, num_channels) 或 torch.Size([0]) 如果视频无音频
