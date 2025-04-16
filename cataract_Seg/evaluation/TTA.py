import torch
import torch.nn.functional as F
from torchvision.transforms.functional import hflip, vflip, rotate


class TestTimeAugmentation:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def augment(self, image):
        """
        应用多种增强方法。
        """
        augmented_images = [image]  # 原始图像
        augmented_images.append(hflip(image))  # 水平翻转
        # augmented_images.append(vflip(image))  # 垂直翻转
        augmented_images.append(rotate(image, 30))  # 顺时针旋转 90 度
        augmented_images.append(rotate(image, -30))  # 逆时针旋转 90 度
        return augmented_images

    def deaugment(self, predictions):
        """
        对增强后的预测进行逆变换。
        """
        deaugmented_predictions = []
        deaugmented_predictions.append(predictions[0])  # 原始预测
        deaugmented_predictions.append(hflip(predictions[1]))  # 水平翻转还原
        # deaugmented_predictions.append(vflip(predictions[2]))  # 垂直翻转还原
        deaugmented_predictions.append(rotate(predictions[2], -30))  # 逆旋转 90 度
        deaugmented_predictions.append(rotate(predictions[3], 30))  # 顺旋转 90 度
        return deaugmented_predictions

    def predict(self, image):
        """
        对单张图像执行 TTA 并融合结果。
        """
        # 1. 数据增强
        batch_size, _, _, _ = image.shape
        augmented_images = [image]  # 原始图像
        augmented_images.append(hflip(image))  # 水平翻转
        # augmented_images.append(vflip(image))  # 垂直翻转
        augmented_images.append(rotate(image, 30))  # 顺时针旋转 90 度
        augmented_images.append(rotate(image, -30))  # 逆时针旋转 90 度

        predictions = []
        for img in augmented_images:
            with torch.no_grad():
                pred = self.model(img.to(self.device))  # 批次维度
                predictions.append(pred)

        # 逆增强
        deaugmented_predictions = self.deaugment(predictions)

        # 平均融合
        final_prediction = torch.stack(deaugmented_predictions).mean(dim=0)
        return final_prediction
