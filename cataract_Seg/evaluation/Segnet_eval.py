import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from scipy.ndimage import maximum_filter, label
from timm.data.auto_augment import color
from evaluation.TTA import TestTimeAugmentation
from model.LWANet import LWANet as LWANet_Image
from model.LWANet_l import LWANet as LWANet_Landmark
import os
from config import params
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
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
    # cv2.imshow('iii', padded_image)
    # cv2.waitKey(0)

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]

    # 将mean和std转换为numpy数组
    # mean = np.array(mean)
    # std = np.array(std)

    # 标准化处理 (减去mean并除以std)


    # Normalize the image to [0, 1] range
    normalized_image = rgb_image / 255.0

    normalized_image = (normalized_image - mean) / std

    # Convert to (C, H, W) format
    input_tensor = np.transpose(normalized_image, (2, 0, 1)).astype(np.float32)

    # Convert to torch tensor
    input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)

    return input_tensor


def load_model(model_path, with_kpt=True, detect_landmark=False):
    if detect_landmark:
        print('Landmark')
        model = LWANet_Landmark(num_classes=16).to(device)
    else:
        print('Segmentation')
        model = LWANet_Image(num_classes=16).to(device)
    pretrained_dict = torch.load(model_path)
    '''
    model_dict = model.state_dict()
    if not with_kpt:
        print('Deleted landmark')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "additional" not in k and "landmark" not in k}
    model_dict.update(pretrained_dict)
    '''
    model.load_state_dict(pretrained_dict)

    model.eval()
    return model


def overlay_image_with_onehot_mask_torch(image, mask, tool_and_eyes_class, alpha=0.5, min_area=900, max_iterations=10,
                                         draw_class_12=True, confidence_threshold=0.01):
    """
    将图像与 one-hot 编码的 mask 叠加，并为不同类别的区域着色，同时将小于 min_area 的像素区域归为最近的合规标签。

    参数：
    - image: 原始彩色图像, 形状 (1, 3, H, W) (torch.Tensor)
    - mask: one-hot 编码的 mask, 形状 [1, num_classes, H, W] (torch.Tensor)
    - tool_and_eyes_class: 类别名称和索引的字典，例如 {'Capsulorhexis Cystotome': 1, 'Pupil': 2, ...}
    - alpha: 叠加时的透明度系数 (默认 0.5)
    - min_area: 最小像素区域大小 (默认 100)
    - max_iterations: 最大迭代次数 (默认 10)

    返回：
    - overlayed_image: 叠加后的图像
    """
    # 将 mask 的大小调整为 (512, 512)
    # mask = F.interpolate(mask, (640, 640), mode='bilinear', align_corners=False)
    # mask = F.interpolate(mask, (640, 640), mode='nearest')
    # image = F.interpolate(image, (128, 128), mode='nearest')

    # 检查图像和 mask 的尺寸是否匹配
    img_height, img_width = image.shape[2], image.shape[3]
    mask_height, mask_width = mask.shape[2], mask.shape[3]
    if (img_height != mask_height) or (img_width != mask_width):
        raise ValueError(f"Image size {(img_height, img_width)} and mask size {(mask_height, mask_width)} do not match")

    # 获取每个像素点所属类别的索引
    mask = torch.exp(mask)
    # mask = nms_for_mask_torch(mask.float())
    # print(mask)
    # exit(0)
    mask_class_index = torch.argmax(mask, dim=1).squeeze(0).cpu().numpy()

    # mask_probabilities, mask_class_index = torch.max(mask, dim=1)
    # mask_class_index = mask_class_index.squeeze(0).cpu().numpy()
    # mask_probabilities = mask_probabilities.squeeze(0).cpu().numpy()

    # 应用置信度阈值过滤
    # mask_class_index[mask_probabilities < confidence_threshold] = 0

    unique_labels, counts = np.unique(mask_class_index, return_counts=True)

    # 迭代替换小于 min_area 的标签区域，直到满足条件或达到最大迭代次数
    # '''
    for label, count in zip(unique_labels, counts):
        if count < min_area:
            mask_class_index[mask_class_index == label] = 0
    # '''
    unique_labels, counts = np.unique(mask_class_index, return_counts=True)
    print("Unique values:", unique_labels)
    print("Counts:", counts)
    # unique_labels, counts = np.unique(mask_class_index, return_counts=True)
    # print("Unique values After:", unique_labels)
    # print("Counts After:", counts)
    # print('***'*25)

    # 定义每个类别的颜色
    colors = plt.cm.get_cmap('jet', len(tool_and_eyes_class))
    color_mapping = {label: colors(i)[:3] for i, label in enumerate(tool_and_eyes_class.values())}

    # 创建与 image 相同大小的彩色 mask
    color_mask = np.zeros((img_height, img_width, 3), dtype=np.float32)

    # 将图像从 Tensor 转为 NumPy 并调整为 (H, W, 3) 形式
    if isinstance(image, torch.Tensor):
        image = image.squeeze(0).cpu().numpy()
    if image.shape[0] == 3 and len(image.shape) == 3:
        image = np.transpose(image, (1, 2, 0))

    # 归一化图像到 [0, 1]
    image = image.astype(np.float32) / 255.0

    # 为每个类别的区域着色
    filtered_labels = np.unique(mask_class_index)
    for label in filtered_labels:
        if label == 12 and not draw_class_12:
            continue
        if label in color_mapping:
            color = color_mapping[label]
            color_mask[mask_class_index == label] = color

    # 叠加原始图像和彩色 mask

    overlayed_image = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    # 将类别名称显示在左上角
    for label in filtered_labels:
        if label == 12 and not draw_class_12:
            continue
        if label in tool_and_eyes_class.values():
            label_name = [name for name, value in tool_and_eyes_class.items() if value == label][0]
            cv2.putText(overlayed_image, label_name, (10, 30 * (label + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 4)

    # 将叠加后的图像转换回 [0, 255] 范围并转换为 uint8
    # overlayed_image = (overlayed_image * 255).astype(np.uint8)

    return overlayed_image


def gaussian_smooth(mask, kernel_size=5, sigma=1.0):
    mask_smoothed = F.gaussian_blur(mask, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
    return mask_smoothed



def test_image(image_folder_path, model, detect_landmark=False,
               target_size=(640, 640), pad_color=(0, 0, 0), alpha=0.5):
    """
    Tests a single image with the model and overlays the mask on the original image.

    Parameters:
        image_path (str): The path to the image file.
        model (torch.nn.Module): The segmentation model.
        target_size (tuple): The target size for resizing (width, height).
        pad_color (tuple): The color to use for padding as (B, G, R).
        alpha (float): The transparency level for overlaying the mask.

    Returns:
        overlayed_image (numpy.ndarray): The original image with the overlayed mask.
    """
    # Load the image
    image_folder_list = os.listdir(image_folder_path)
    image_paths = [os.path.join(image_folder_path, image) for image in image_folder_list]
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print("无法加载图像文件")
            return None

        # Prepare the input tensor
        input_tensor = prepare_input_tensor(image, target_size, pad_color)
        # input_tensor = apply_gaussian_smoothing(input_tensor, kernel_size=5, sigma=1.0)

        # Run inference
        with torch.no_grad():
            if detect_landmark:
                landmarks, class_type = model(input_tensor)
                landmarks = torch.clamp(landmarks, 0, 1)
                print(class_type.shape)
                print(torch.argmax(class_type, dim=-1))
                overlayed_image = image.copy()
                width, height = overlayed_image.shape[1], overlayed_image.shape[0]
                for i in range(len(landmarks[0])):
                    # print(landmarks[i][0])

                    x, y = int(landmarks[0][i][0] * width), int(landmarks[0][i][1] * height)
                    print(x, y)
                    cv2.circle(overlayed_image, (x, y), 5, (255, 0, 255), -1)
            else:
                output = model(input_tensor)
                overlayed_image = overlay_image_with_onehot_mask_torch(input_tensor, output, alpha=alpha, tool_and_eyes_class=params['model_param']['tool_and_eyes_class'])
        if False:
            mask_np = overlayed_image.astype(np.uint8)  # OpenCV 需要 mask 是 uint8 类型
            # print(mask_np.shape)
            brightness_factor = 50  # 可以根据需要调整
            mask_np = np.clip(mask_np * brightness_factor, 0, 255).astype(np.uint8)
        # Display results
        cv2.imshow("Original Image", image)
        cv2.imshow("Overlayed Mask Image", overlayed_image)

        # Wait for a key press and close windows
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    # return overlayed_image


def test_video(video_path, model, detect_landmark):
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
        input_tensor = prepare_input_tensor(frame, (640, 640))

        # print(input_tensor.shape)


        # 推断结果
        # with torch.no_grad():
            # output, landmarks, class_type, _ = model(input_tensor)
            # output = model(input_tensor)

        # print(final_output.shape)
        # exit(0)
        # output = apply_gaussian_smoothing(output, kernel_size=5, sigma=1.0)

        # 后处理得到最终结果
        with torch.no_grad():
            if detect_landmark:
                landmarks, class_type = model(input_tensor)

                # if int(class_type) > 0 and class_type < 12:
                landmarks = torch.clamp(landmarks, 0, 1)
                print(class_type.shape)
                print(torch.argmax(class_type, dim=-1))
                class_id = torch.argmax(class_type, dim=-1)
                print(landmarks.shape)
                # exit(0)
                if class_id[0][1] == class_id[0][0]:
                    landmarks = np.mean(landmarks.cpu().numpy(), axis=1, keepdims=True)
                    landmarks = np.tile(landmarks, (1, 2, 1))
                    print(landmarks.shape)
                overlayed_image = frame.copy()
                width, height = overlayed_image.shape[1], overlayed_image.shape[0]
                for i in range(len(landmarks[0])):
                    if class_id[0][i] > 0 and class_id[0][i] < 12:
                        x, y = int(landmarks[0][i][0] * width), int(landmarks[0][i][1] * height)
                        print(x, y)
                        cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)
            else:
                tta = TestTimeAugmentation(model, device)
                output = tta.predict(input_tensor)
                processed_frame = overlay_image_with_onehot_mask_torch(input_tensor, output, alpha=0.5, tool_and_eyes_class=params['model_param']['tool_and_eyes_class'])
        # print(processed_frame.shape, 'dealed')
                mask_mapped = (processed_frame * (255 // (16-1))).astype(np.uint8)

        # 将 mask 转换为伪彩色图像
        # color_mask = cv2.applyColorMap(mask_mapped, cv2.COLORMAP_JET)
        # color_mask = cv2.Canny(color_mask.astype(np.uint8) * 255, 100, 200)

        # 将灰度图像与原始图像叠加
        # overlay_frame = overlay_segmentation(frame, processed_frame)

        # 实时显示
        cv2.imshow('ori', frame)
        if not detect_landmark:
            cv2.imshow('img', processed_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def nms_for_mask_torch(mask, window_size=5):
    """
    使用 PyTorch 对 mask 进行非极大值抑制。

    Args:
        mask (torch.Tensor): 输入的 mask，形状为 (1, 1, H, W)。
        window_size (int): NMS 滑动窗口大小。

    Returns:
        torch.Tensor: 非极大值抑制后的 mask，形状与输入相同。
    """
    # 使用 max_pool2d 找到局部最大值
    batch, num_classes, height, width = mask.shape
    padding = window_size // 2

    # 对每个类别通道应用 max_pool2d
    max_mask = F.max_pool2d(mask, kernel_size=window_size, stride=1, padding=padding)

    # 保留局部最大值
    nms_mask = torch.where(mask == max_mask, mask, torch.tensor(0.0, device=mask.device))
    return nms_mask


def smooth_seg(predict_mask):
    kernel = np.ones((5, 5), np.uint8)
    smoothed_mask = cv2.morphologyEx(predict_mask, cv2.MORPH_OPEN, kernel)
    return smoothed_mask

def fill_holes(seg_image):
    num_label, label_im = cv2.connectedComponents(seg_image.astype(np.uint8))
    filled_img = np.zeros_like(seg_image)
    for i in range(1, num_label):
        if np.sum(seg_image == i) > 100:
            filled_img[seg_image == i] = i
    return filled_img


def accuracy(model, device):
    from preprocessing.DataLoader.cataract_dataloader import load_dataloader
    from sklearn.metrics import accuracy_score
    dataloader = load_dataloader()
    distance_total = 0
    accuracy_total = 0
    idxs = 1
    for batch_idx, (image, mask, tips_kpts) in enumerate(tqdm(dataloader)):
        image = image.to(device)
        tips_kpts = tips_kpts.to(device)
        with torch.no_grad():
            landmarks, class_type = model(image)
        class_type = torch.argmax(class_type, dim=-1)


        landmarks = torch.clamp(landmarks, 0, 1)
        # print(class_type[0])
        for i in range(class_type.shape[0]):
            for j in range(len(class_type[i])):
                if class_type[i, j] == 0:
                    # print(landmarks[i, j])
                    landmarks[i, j] = -1
                    # print(landmarks[i, j])
                    # exit(0)

        # print(tips_kpts[:, :, :2].shape)
        # print(landmarks)
        # print(tips_kpts[:, :, :2])
        distance = abs(landmarks - tips_kpts[:, :, :2])
        distance_total += distance.mean().item() / params['batch_size']

        class_type = class_type.cpu().detach().numpy().flatten()
        tips_kpts_label = tips_kpts[:,:, 2].cpu().detach().numpy().flatten()

        # print(tips_kpts_label.shape)
        # print(class_type.shape)
        acc = accuracy_score(class_type, tips_kpts_label)
        accuracy_total += acc
        idxs += 1
        # print(distance_total)
        # print(accuracy_total)

    distance_total = distance_total / len(dataloader)
    accuracy_total = accuracy_total / len(dataloader)

    print(f'distance_total: {distance_total},'
          f'accuracy_total: {accuracy_total}')



if __name__ == '__main__':
    video_path = r"C:\Users\Charl\Downloads\cataract-1k\videos\case_5309.mp4"
    # video_path = r"C:\Users\Charl\Downloads\cataract-101\Video_Classification\934\2_1261.mp4"
    detect_landmark = False
    img_folder = r'C:\Users\Charl\Downloads\cataract-1k\Annotations\Generated_Dataset\train\images'
    if not detect_landmark:
        model_path = r"C:\Users\Charl\PycharmProjects\cataract_Seg\checkpoint_fine_tune_colab\epoch_250.pth"
    # model_path = r"C:\Users\Charl\PycharmProjects\cataract_Seg\checkpoint_fine_landmark\epoch_290.pth"
    else:
        model_path = r"C:\Users\Charl\PycharmProjects\cataract_Seg\checkpoint_fine_tune\epoch_850.pth"

    model = load_model(model_path, with_kpt=True, detect_landmark=detect_landmark)
    # accuracy(model, device)
    # test_image(img_folder, model, detect_landmark=detect_landmark)
    test_video(video_path, model, detect_landmark=detect_landmark)