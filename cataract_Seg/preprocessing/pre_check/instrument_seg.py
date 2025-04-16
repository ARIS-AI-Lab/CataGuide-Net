import numpy as np
import cv2
import os
import json
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


def pic_viewer(dictionary_path):
    pic_list = os.listdir(dictionary_path)
    pic_path = [os.path.join(dictionary_path, i) for i in pic_list]
    for pic in pic_path:
        print(pic)
        img = cv2.imread(pic)
        height, width, _ = img.shape
        print(f'img size is {height}x{width}')
        cv2.imshow('img', img)
        cv2.waitKey(0)


def anno_check(anno_folder_path, img_folder_path):
    anno_list = os.listdir(anno_folder_path)
    anno_path = [os.path.join(anno_folder_path, i) for i in anno_list]
    for anno in anno_path:
        '''
        :param anno:
            description
            tags
            size
            objects
            :parameter objects:
                description, tags, classTitle, points
                :parameter points:
                    exterior, interior

        '''

        with open(anno, 'r') as f:
            content = json.load(f)

            img_name = anno.split("\\")[-1].strip(".json")
            # print(content)
            # exit(0)
            # for i in content['tags']:
            #     print(i['name'])
            # exit(0)
            # print(content['objects'])

            img_path = os.path.join(img_folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # '''
            if True:
                color_idx = 0
                for obj in content['objects']:
                    classTitle = obj['classTitle']
                    if classTitle is 'Eye':
                        continue
                    kpts = obj['points']['exterior']
                    print(color_idx, ' is: ', obj['classTitle'])

                    print(img_name)

                    temp_color = colors[color_idx]
                    # print(np.array(kpts))
                    # exit(0)
                    for kpt_idx in range(len(kpts)):
                        x, y = kpts[kpt_idx][0], kpts[kpt_idx][1]
                        cv2.circle(img, (x, y), 1, temp_color, -1)
                    color_idx += 1
                cv2.imshow('img', img)
                cv2.waitKey(0)
            else:
                interior = content['objects'][0]['points']['interior']
                if len(interior) != 0:
                    print(interior)
            # '''
            # exit(0)

def count_class_titles(anno_folder_path):
    class_titles = set()  # 使用集合存储 classTitle，自动去重

    # 遍历注释文件
    anno_list = os.listdir(anno_folder_path)
    anno_path = [os.path.join(anno_folder_path, i) for i in anno_list]

    for anno in anno_path:
        with open(anno, 'r') as f:
            content = json.load(f)
            # 遍历每个对象，提取 classTitle
            for obj in content['objects']:
                classTitle = obj['classTitle']
                class_titles.add(classTitle)  # 将 classTitle 添加到集合中

    # 输出所有 classTitle 及其总数
    print(f"所有的 classTitle: {class_titles}")
    print(f"总共有 {len(class_titles)} 种 classTitle")


if __name__ == '__main__':
    folder_path = r'C:\Users\Charl\Downloads\Cataract_Seg\Mask_segmentation\Mask_segmentation\Instrument_detection'
    dataset_folder = os.path.join(folder_path, 'dataset_instrument', 'train')
    anno_folder = os.path.join(folder_path, 'annotations_instrument')
    # pic_viewer(dataset_folder)
    anno_check(anno_folder, dataset_folder)
    # count_class_titles(anno_folder)