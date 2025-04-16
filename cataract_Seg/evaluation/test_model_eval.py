import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
from model.model import SMUnet
# from model.lightweight_model import R2Unet
from src.utils.loss import CombinedLoss
from tqdm import tqdm
import os
from config import params
# from model.transformer import MobileUnet
from model.sd_mobile import SegNet
from preprocessing.DataLoader.cataract_dataloader import load_dataloader
import segmentation_models_pytorch as smp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
colors = [
    (255, 0, 0),    # 红色
    (0, 255, 0),    # 绿色
    (0, 0, 255),    # 蓝色
    (255, 255, 0),  # 黄色
    (255, 165, 0),  # 橙色
    (128, 0, 128),  # 紫色
    (0, 255, 255),  # 青色
    (255, 192, 203),# 粉红色
    (128, 128, 0),  # 橄榄色
    (0, 128, 128)   # 墨绿色
]


def resize_and_pad(image, target_size, pad_color=(0, 0, 0)):
    """
    Resizes and pads an image to the target size.

    Parameters:
        image (numpy.ndarray): The input image.
        target_size (tuple): The target size as (width, height).
        pad_color (tuple): The color to use for padding as (B, G, R).

    Returns:
        numpy.ndarray: The resized and padded image.
    """
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate the scaling factor to resize the image
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale = min(scale_w, scale_h)

    # Resize the image with the calculated scale
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Calculate padding to reach the target size
    pad_width = target_width - new_width
    pad_height = target_height - new_height

    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    # Pad the image to reach the target size
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)

    return padded_image


def prepare_input_tensor(image, target_size, pad_color=(0, 0, 0)):
    """
    Prepares an input tensor suitable for model input.

    Parameters:
        image (numpy.ndarray): The input image.
        target_size (tuple): The target size as (width, height).
        pad_color (tuple): The color to use for padding as (B, G, R).

    Returns:
        torch.Tensor: The image formatted as a model input tensor.
    """
    # Resize and pad the image
    padded_image = resize_and_pad(image, target_size, pad_color)

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

    mean = [0.2845, 0.2293, 0.1936]
    std = [0.2542, 0.2000, 0.1942]

    # 将mean和std转换为numpy数组
    mean = np.array(mean)
    std = np.array(std)

    # 标准化处理 (减去mean并除以std)


    # Normalize the image to [0, 1] range
    normalized_image = rgb_image / 255.0

    normalized_image = (normalized_image - mean) / std

    # Convert to (C, H, W) format
    input_tensor = np.transpose(normalized_image, (2, 0, 1)).astype(np.float32)

    # Convert to torch tensor
    input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)

    return input_tensor


def load_model(model_path):
    model = smp.Unet(
        encoder_name='resnet18',
        encoder_weights='imagenet',
        in_channels=3,
        classes=16).to(device)
    model.load_state_dict(torch.load(model_path))
    # model = SMUnet(in_channels=3, out_channels=16).to(device)
    # model = MobileUnet(num_classes=16).to(device)
    # model = SegNet(num_classes=16, num_landmarks=15).to(device)
    # model.load_state_dict(torch.load(model_path))

    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model



def overlay_image_with_onehot_mask_torch(image, mask, alpha=0.5):
    """
    将图像与one-hot编码的mask叠加，并为不同类别的区域着色 (使用 PyTorch 处理 mask)。

    参数：
    - image: 原始彩色图像, 形状 (1, 3, H, W) (torch.Tensor)
    - mask: one-hot编码的mask, 形状 [1, num_classes, H, W] (torch.Tensor)
    - alpha: 叠加时的透明度系数
    """
    # 获取图像和mask的高度和宽度
    img_height, img_width = image.shape[2], image.shape[3]
    mask_height, mask_width = mask.shape[2], mask.shape[3]

    # 检查图像和mask的尺寸是否匹配
    if (img_height != mask_height) or (img_width != mask_width):
        raise ValueError(f"Image size {(img_height, img_width)} and mask size {(mask_height, mask_width)} do not match")

    # 使用 torch.argmax 来获取 mask 中每个像素点所属类别的索引
    mask_class_index = torch.argmax(mask, dim=1).squeeze(0).cpu().numpy()  # [H, W]

    # 获取所有的唯一类别
    unique_labels = np.unique(mask_class_index)

    # 定义每个类别的颜色
    colors = plt.cm.get_cmap('jet', len(unique_labels))

    # 创建一个与image相同大小的彩色mask
    color_mask = np.zeros((img_height, img_width, 3), dtype=np.float32)

    # 定义工具和眼睛类别的映射
    tool_and_eyes_class = {'Capsulorhexis Cystotome': 1, 'Capsulorhexis Forceps': 2, 'Cornea': 3, 'Gauge': 4,
                           'Incision Knife': 5, 'Irrigation-Aspiration': 6, 'Katena Forceps': 7, 'Lens': 8,
                           'Lens Injector': 9, 'Phacoemulsification Tip': 10, 'Pupil': 11, 'Slit Knife': 12,
                           'Spatula': 13, 'cornea1': 14, 'pupil1': 15}

    # 遍历每个类别，给对应的区域着色
    for i, label in enumerate(unique_labels):
        if label == 0 or label == 3:
            continue  # 忽略背景或值为0以及值为3的区域
        color = colors(i)[:3]  # 获取 RGB 颜色
        color = np.array(color)  # 颜色值保持在[0,1]范围内
        color_mask[mask_class_index == label, 0] = color[0]  # 红色通道
        color_mask[mask_class_index == label, 1] = color[1]  # 绿色通道
        color_mask[mask_class_index == label, 2] = color[2]  # 蓝色通道

    if isinstance(image, torch.Tensor):
        image = image.squeeze(0).cpu().numpy()  # 去除批次维度，使形状为 (3, H, W)

    # 检查 image 的格式是否为 (H, W, 3)，如果不是需要转换
    if image.shape[0] == 3 and len(image.shape) == 3:
        image = np.transpose(image, (1, 2, 0))

    # 将图像和mask转换为浮点类型，以便进行加权操作
    image = image.astype(np.float32) / 255.0
    color_mask = color_mask.astype(np.float32)

    # 将原始图像与彩色mask叠加
    '''
    # overlayed_image = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    # 将类别名称显示在左上角
    for label in unique_labels:
        if label in tool_and_eyes_class.values():
            label_name = [name for name, value in tool_and_eyes_class.items() if value == label][0]
            cv2.putText(overlayed_image, label_name, (10, 30 * label), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 将叠加后的图像转换回[0, 255]的范围
    # overlayed_image = (overlayed_image * 255).astype(np.uint8)
    '''
    return color_mask




def test_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 逐帧处理视频
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame
        if not ret:
            break

        # 预处理输入帧
        # input_tensor = resize_and_pad(frame, (640, 640))
        input_tensor = prepare_input_tensor(frame, (512, 512))

        # print(input_tensor.shape)


        # 推断结果
        with torch.no_grad():
            # output, landmarks, class_type = model(input_tensor)
            output = model(input_tensor)

        # 后处理得到最终结果
        processed_frame = overlay_image_with_onehot_mask_torch(input_tensor, output, alpha=0.5)
        # print(processed_frame.shape, 'dealed')
        mask_mapped = (processed_frame * (255 // (16-1))).astype(np.uint8)

        # 将 mask 转换为伪彩色图像
        color_mask = cv2.applyColorMap(mask_mapped, cv2.COLORMAP_JET)

        # 将灰度图像与原始图像叠加
        # overlay_frame = overlay_segmentation(frame, processed_frame)

        # 实时显示
        cv2.imshow('ori', frame)
        cv2.imshow('img', color_mask)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = r"C:\Users\Charl\Downloads\cataract-1k\videos\case_5301.mp4"
    # video_path = r"C:\Users\Charl\Downloads\cataract-101\videos\case_269.mp4"
    # model_path = r"C:\Users\Charl\PycharmProjects\cataract_Seg\checkpoint\model_epoch_ft_80.pth"
    # model_path = r"C:\Users\Charl\PycharmProjects\cataract_Seg\checkpoint_test\epoch_1000.pth"
    # model_path = r"C:\Users\Charl\PycharmProjects\cataract_Seg\epoch_20.pth"
    model_path = r"C:\Users\Charl\PycharmProjects\cataract_Seg\checkpoint_recon\epoch_10.pth"
    # model_path = r"C:\Users\Charl\PycharmProjects\cataract_Seg\20.pth"

    model = load_model(model_path)

    test_video(video_path, model)