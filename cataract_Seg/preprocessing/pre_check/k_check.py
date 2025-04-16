import os

import cv2
import numpy as np
import json
from tqdm import tqdm
from numpy.core.defchararray import lower
from torch.utils.tensorboard.summary import video

colors = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (0, 255, 255),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 0, 0),       # Black
    (255, 255, 255), # White
    (255, 165, 0),   # Orange
    (128, 0, 128),   # Purple
    (128, 128, 0),   # Olive
    (0, 128, 128),   # Teal
    (128, 0, 0),     # Maroon
    (0, 128, 0),     # Dark Green
    (0, 0, 128),     # Navy Blue
    (192, 192, 192)  # Silver
]



def load_json(json_file_path):
    with open(json_file_path) as f:
        data = json.load(f)
        return data

def pic_previewer(original_path):
    img_anno_dirs_list  = [f for f in os.listdir(original_path) if os.path.isdir(os.path.join(original_path, f))]
    for img_anno_dir in img_anno_dirs_list:
        show = False
        dir_path = os.path.join(original_path, img_anno_dir)
        img_dir = os.path.join(dir_path, 'img')
        ann_dir = os.path.join(dir_path, 'ann')
        img_name_list = [i for i in os.listdir(img_dir) if i.endswith('jpg') or i.endswith('png')]
        for img_name in img_name_list:

            img_path = os.path.join(img_dir, img_name)

            img = cv2.imread(img_path)
            original_img = img.copy()
            anno_name = img_name + '.json'
            anno_path = os.path.join(ann_dir, anno_name)
            anno_content = load_json(anno_path)

            """
            :param anno_content:
                description
                tags
                size
                objects
            """
            objects = anno_content['objects']
            for idx, obj in enumerate(objects):
                tool_class = obj['classTitle']
                print(tool_class)

                kpts = np.array(obj['points']['exterior'])

                # print(kpts.shape)
                # '''
                if ('pupil' not in tool_class.lower()) and ('cornea' not in tool_class.lower()):
                    tip_kpts = find_tip_kpt(kpts)
                    sorted_arr = tip_kpts[tip_kpts[:, 1].argsort()]
                    min_y_points = sorted_arr[:2]
                    center = np.array([0.5, 0.5])
                    distances = np.linalg.norm(min_y_points - center, axis=1)
                    if 'Capsulorhexis Forceps' in tool_class:
                        closest_index = np.argsort(distances)[:2]
                    else:
                        # closest_index = np.argmin(distances)
                        print('out------------------------------------------------')
                        show = True
                        break
                    tip_kpts_landmark = tip_kpts[closest_index]

                    # lowest_two_kpts = calc_kpts_vector(kpts)
                    # print('kpts:', lowest_two_kpts)

                    img = draw_kpts(img, tip_kpts_landmark, colors[idx])
                    cv2.imshow('img', cv2.resize(img, (512, 512)))
                    cv2.imshow('ori_img', cv2.resize(original_img, (512, 512)))
                    cv2.waitKey(0)
                # exit(0)
                # '''
                else:
                    break
                if show:
                    break
                '''
                if True:
                    if ('pupil' not in tool_class.lower()) and ('cornea' not in tool_class.lower()):
                        img = draw_kpts(img, kpts, colors[idx])
                        # img = draw_extrior(img, kpts, colors[idx])
                else:
                    # img = draw_kpts(img, kpts, colors[idx])
                    img = draw_extrior(img, kpts, colors[idx])
                '''
            print('****'*20)
            # cv2.imshow('img', cv2.resize(img, (512, 512)))
            # cv2.imshow('ori_img', cv2.resize(original_img, (512, 512)))
            # cv2.waitKey(0)
                # '''


def draw_kpts(img, kpts, color):
    kpts = np.reshape(np.array(kpts), (-1, 2))
    for i in range(kpts.shape[0]):
        x, y = int(kpts[i][0]), int(kpts[i][1])
        cv2.circle(img, (x, y), 5, color, -1)
    return img


def draw_extrior(img, kpts, color):
    points = np.reshape(np.array(kpts), (-1, 1, 2))  # 将点集转换为合适的形状
    print(points.shape)
    # 使用 fillPoly 来填充多边形区域
    cv2.fillPoly(img, [points], color)
    return img


def count_unique_class_titles(original_path):
    unique_classes = set()  # 使用集合来存储唯一的 classTitle
    img_anno_dirs_list = [f for f in os.listdir(original_path) if os.path.isdir(os.path.join(original_path, f))]

    for img_anno_dir in tqdm(img_anno_dirs_list):
        dir_path = os.path.join(original_path, img_anno_dir)
        ann_dir = os.path.join(dir_path, 'ann')

        # 遍历所有标注文件
        anno_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
        for anno_file in anno_files:
            anno_path = os.path.join(ann_dir, anno_file)
            anno_content = load_json(anno_path)
            objects = anno_content['objects']

            # 提取每个对象的 classTitle 并添加到集合中
            for obj in objects:
                tool_class = obj['classTitle']
                unique_classes.add(tool_class)

    # 为每个 classTitle 分配一个唯一的数字并生成字典
    class_title_dict = {class_title: idx+1 for idx, class_title in enumerate(sorted(unique_classes))}

    # 返回 classTitle 的总数和字典
    return len(unique_classes), class_title_dict


def angle_between_vectors(v1, v2, epsilon=1e-8):
    # 点积
    dot_product = np.dot(v1, v2)
    # 向量的模
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # 计算夹角的余弦值，防止浮点数误差引发的超过[-1, 1]的情况
    denominator = (norm_v1 * norm_v2) + epsilon
    cos_angle = np.clip(dot_product / denominator, -1.0, 1.0)
    # 计算角度并返回
    return np.arccos(cos_angle)


# 遍历整个array，计算角度变化
def find_top_three_max_angle_changes(arr):
    angle_changes = []

    for i in range(1, len(arr) - 1):
        # 计算当前点的前后向量
        vector1 = arr[i] - arr[i - 1]
        vector2 = arr[i + 1] - arr[i]

        # 计算两个向量的夹角
        angle_change = angle_between_vectors(vector1, vector2)

        # 记录下角度变化和对应的索引
        angle_changes.append((i, angle_change))

    # 按照角度变化从大到小排序
    angle_changes.sort(key=lambda x: x[1], reverse=True)

    # 取出角度变化最大的前三组
    top_three = angle_changes[:5]

    return top_three

def calc_kpts_vector(arr):
    sorted_arr = arr[arr[:, 1].argsort()]
    # center_xy = (sorted_arr[-1] + sorted_arr[-2])/2
    # print(center_xy)

    # 取出y轴值最低的两个元素
    return sorted_arr


def find_tip_kpt(kpts):
    top_three = find_top_three_max_angle_changes(kpts)
    top_three_kpt = []
    for index, angle_change in top_three:
        # print(
        #     f"角度变化最大的位置索引是: {index}, 对应的点是: {kpts[index]}, 角度变化: {np.degrees(angle_change):.2f}°")
        top_three_kpt.append(kpts[index])
    top_three_kpt = np.reshape(np.array(top_three_kpt), (-1, 2))
    tip_kpts = calc_kpts_vector(top_three_kpt)

    return tip_kpts
if __name__ == '__main__':
    dataset_file_path = r'C:\Users\Charl\Downloads\cataract-1k\Annotations\Images-and-Supervisely-Annotations'

    pic_previewer(dataset_file_path)
    # class_count, unique_class_titles = count_unique_class_titles(dataset_file_path)
    # print(f"总共有 {class_count} 种不同的 classTitle:")
    # print(unique_class_titles)