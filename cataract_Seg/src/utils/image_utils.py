import numpy as np
from PIL import Image
import cv2
import torch
import random
from torchvision.transforms import functional as F


class CustomTransform:
    def __init__(self, image_transform=None, rotation_angle=30, augmentation_prob=0.5, brightness_range=(1, 2),
                 scale_range=(0.6, 1.0), add_noise_prob=0.3, noise_std=0.05):
        """
        CustomTransform class for applying augmentations to images, masks, and landmarks.

        Args:
            image_transform (callable, optional): Additional image transformations (e.g., normalization).
            rotation_angle (float): Maximum rotation angle for augmentation.
            augmentation_prob (float): Probability of applying each augmentation.
            brightness_range (tuple of floats): Range for adjusting brightness.
        """
        self.image_transform = image_transform
        self.rotation_angle = rotation_angle
        self.prob = augmentation_prob
        self.brightness_range = brightness_range
        self.crop_scale = scale_range
        self.add_noise_prob = add_noise_prob
        self.noise_std = noise_std

    def add_gaussian_noise(self, image):
        """Adds Gaussian noise to the image."""
        mean = 0  # Mean of Gaussian noise
        std = self.noise_std  # Standard deviation of Gaussian noise
        img_np = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(mean, std, img_np.shape)
        noisy_img_np = img_np + noise
        noisy_img_np = np.clip(noisy_img_np, 0.0, 1.0)
        noisy_img = Image.fromarray((noisy_img_np * 255).astype(np.uint8))
        return noisy_img

    def adjust_brightness(self, image):
        """Randomly adjusts the brightness of the image within the specified range."""
        brightness_factor = random.uniform(*self.brightness_range)
        return F.adjust_brightness(image, brightness_factor)

    def random_crop(self, image, mask, landmarks):
        """Randomly crops the image, mask, and pads with zeros to keep the original size."""
        scale = random.uniform(*self.crop_scale)
        w, h = image.size
        crop_h, crop_w = int(h * scale), int(w * scale)

        # Randomly select the top-left corner for cropping
        top = random.randint(0, max(0, h - crop_h))
        left = random.randint(0, max(0, w - crop_w))

        # Crop the image and mask with the same region
        cropped_image = F.crop(image, top, left, crop_h, crop_w)
        cropped_mask = F.crop(mask, top, left, crop_h, crop_w)

        # Create a new blank (zero-padded) canvas of the original size
        new_image = Image.new('RGB', (w, h))
        new_mask = Image.new('L', (w, h))

        # Paste the cropped image and mask onto the zero-padded canvas at a random position
        paste_top = random.randint(0, h - crop_h)
        paste_left = random.randint(0, w - crop_w)
        new_image.paste(cropped_image, (paste_left, paste_top))
        new_mask.paste(cropped_mask, (paste_left, paste_top))

        # Adjust landmarks for the crop and paste location
        adjusted_landmarks = []
        for x, y in landmarks:
            x, y = x * w, y * h  # Scale to original image size
            if left <= x < left + crop_w and top <= y < top + crop_h:
                x, y = x - left + paste_left, y - top + paste_top
                adjusted_landmarks.append([x / w, y / h])  # Rescale to [0, 1]
            else:
                adjusted_landmarks.append([0, 0])  # Discard points outside the crop



        return new_image, new_mask, np.array(adjusted_landmarks)

    def rotate(self, image, mask, landmarks, angle):
        """Rotates the image, mask, and landmarks by a given angle."""
        # Get image dimensions
        w, h = image.size

        # Denormalize landmarks to original image scale
        # landmarks[:, 0] *= w
        # landmarks[:, 1] *= h

        # Rotate image and mask
        # print('***' * 80)
        # print(landmarks)
        image = F.rotate(image, angle)
        mask = F.rotate(mask, angle)

        # Rotate landmarks
        angle_rad = np.radians(-angle)
        center_x, center_y = 0.5, 0.5
        rotated_landmarks = []
        for x, y in landmarks:
            # 计算围绕中心点旋转后的新坐标
            x_shifted, y_shifted = x - center_x, y - center_y  # 将坐标平移到以中心为原点
            x_new = center_x + (np.cos(angle_rad) * x_shifted - np.sin(angle_rad) * y_shifted)
            y_new = center_y + (np.sin(angle_rad) * x_shifted + np.cos(angle_rad) * y_shifted)
            rotated_landmarks.append([x_new, y_new])

        rotated_landmarks = np.array(rotated_landmarks)
        # print(rotated_landmarks)
        # print(landmarks == rotated_landmarks)
        # print('***'*80)

        # Renormalize landmarks to [0, 1] scale
        # rotated_landmarks[:, 0] /= w
        # rotated_landmarks[:, 1] /= h

        return image, mask, rotated_landmarks

    def flip(self, image, mask, landmarks):
        """Horizontally flips the image, mask, and landmarks."""
        w, h = image.size

        # Denormalize landmarks to original image scale
        landmarks[:, 0] *= w
        landmarks[:, 1] *= h

        # Flip image and mask horizontally
        image = F.hflip(image)
        mask = F.hflip(mask)

        # Flip landmarks' X coordinates
        flipped_landmarks = np.array([[w - x, y] for x, y in landmarks])

        # Renormalize landmarks to [0, 1] scale
        flipped_landmarks[:, 0] /= w
        flipped_landmarks[:, 1] /= h

        return image, mask, flipped_landmarks

    def __call__(self, image, mask, landmarks):
        """Applies augmentations to the image, mask, and landmarks."""
        # Apply horizontal flip with probability self.prob
        # '''
        if random.random() < self.prob:
            image, mask, landmarks = self.flip(image, mask, landmarks)

        # Apply random rotation with probability self.prob
        if random.random() < self.prob:
            angle = random.uniform(-self.rotation_angle, self.rotation_angle)
            image, mask, landmarks = self.rotate(image, mask, landmarks, angle)
        # '''
        if random.random() < self.prob:
            image, mask, landmarks = self.random_crop(image, mask, landmarks)
        # Adjust brightness
        image = self.adjust_brightness(image)

        if random.random() < self.add_noise_prob:
            image = self.add_gaussian_noise(image)

        # Apply additional image transformations (e.g., normalization)
        if self.image_transform:
            image = self.image_transform(image)

        # Convert mask to tensor
        # print(type(mask), mask.shape)
        # mask = Image.fromarray(mask)
        # mask = mask.resize((mask.width // 4, mask.height // 4), Image.NEAREST)
        mask_np = np.array(mask)

        # Step 1: 边缘检测
        edges = cv2.Canny(mask_np, threshold1=100, threshold2=200)

        # Step 2: 边界扩展和平滑处理
        # 使用形态学操作扩展边界
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # Step 3: 平滑边界
        # 将扩展后的边界区域与原图像进行合并
        smoothed_mask = mask_np.copy()
        smoothed_mask[dilated_edges > 0] = cv2.GaussianBlur(mask_np, (5, 5), 0)[dilated_edges > 0]

        # 将处理结果转回PIL图像
        smoothed_mask_img = Image.fromarray(smoothed_mask)
        mask = torch.from_numpy(np.array(smoothed_mask_img)).long()

        landmarks = torch.from_numpy(landmarks).float()

        return image, mask, landmarks






def resize_and_pad(image, landmarks=None, target_size=(256, 256), pad_value=0):
    """
    Resize the image and pad it to the target size while adjusting the landmarks accordingly.

    Parameters:
    - image: The input image in PIL.Image.Image format.
    - landmarks: A dictionary where keys are landmark names and values are tensors of coordinates.
                 Each tensor should be of shape (1, num_points, 2) where each point is [x, y].
    - target_size: Tuple of (width, height) to which the image should be resized and padded.
    - pad_value: The value used for padding the image, default is 0 (black).

    Returns:
    - resized_padded_image: The resized and padded image in PIL.Image.Image format.
    - new_landmarks: The updated dictionary of landmark coordinates after resizing and padding.
    """

    # Convert the input PIL image to NumPy array
    image = np.array(image)

    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate the scaling factor for each dimension
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale = min(scale_w, scale_h)  # Keep aspect ratio

    # Resize the image
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height))

    # Pad the resized image to the target size
    pad_width = (target_width - new_width) // 2
    pad_height = (target_height - new_height) // 2
    resized_padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_height,
        target_height - new_height - pad_height,
        pad_width,
        target_width - new_width - pad_width,
        cv2.BORDER_CONSTANT,
        value=[pad_value, pad_value, pad_value]
    )

    # Adjust the landmarks according to the resizing and padding
    new_landmarks = {}
    for key, points in landmarks.items():
        # Ensure the points are in NumPy array format if they are tensors
        points = points.squeeze().numpy() if isinstance(points, torch.Tensor) else points
        new_points = []
        for x, y in points:
            # Denormalize the landmark (assuming input was absolute pixel values)
            x = float(x)
            y = float(y)
            # Apply scaling
            x = float(x * scale)
            y = float(y * scale)
            # Apply padding
            x = x + pad_width
            y = y + pad_height
            new_points.append((x, y))
        new_landmarks[key] = np.reshape(np.array(new_points), (-1, 2))

    # Convert the resized padded image back to PIL format
    resized_padded_image = Image.fromarray(resized_padded_image)

    return resized_padded_image, new_landmarks
