import cv2
import numpy as np
import json
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_json(json_file_path):
    with open(json_file_path) as f:
        data = json.load(f)
        return data


def generate(original_path, save_path):
    anno_save_path = os.path.join(save_path, 'annotations')
    img_save_path = os.path.join(save_path, 'images')


    if not os.path.exists(anno_save_path):
        os.mkdir(anno_save_path)
    elif not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    img_anno_dirs_list  = [f for f in os.listdir(original_path) if os.path.isdir(os.path.join(original_path, f))]
    # print(img_anno_dirs_list)
    # exit(0)

    for img_anno_dir in img_anno_dirs_list:
        dir_path = os.path.join(original_path, img_anno_dir)
        img_dir = os.path.join(dir_path, 'img')
        ann_dir = os.path.join(dir_path, 'ann')
        img_name_list = [i for i in os.listdir(img_dir) if i.endswith('jpg') or i.endswith('png')]
        for img_name in tqdm(img_name_list):
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            anno_name = img_name + '.json'
            anno_path = os.path.join(ann_dir, anno_name)
            anno_content = load_json(anno_path)
            cv2.imwrite(os.path.join(img_save_path, img_name), img)
            json_save_path = os.path.join(anno_save_path, anno_name)
            with open(json_save_path, 'w') as f:
                json.dump(anno_content, f, indent=4)

            """
            :param anno_content:
                description
                tags
                size
                objects
            """

def save_dir(save_path):
    anno_save_path = os.path.join(save_path, 'annotations')
    img_save_path = os.path.join(save_path, 'images')

    if not os.path.exists(anno_save_path):
        os.mkdir(anno_save_path)
    elif not os.path.exists(img_save_path):
        os.mkdir(img_save_path)
    return anno_save_path, img_save_path


def generate_train_and_test(original_path, save_path):

    img_anno_dirs_list = [f for f in os.listdir(original_path) if os.path.isdir(os.path.join(original_path, f))]
    train_data, test_data = train_test_split(img_anno_dirs_list, test_size=0.3)
    print(len(train_data), len(test_data))
    for img_anno_dir in train_data:
        dir_path = os.path.join(original_path, img_anno_dir)

        img_dir = os.path.join(dir_path, 'img')
        ann_dir = os.path.join(dir_path, 'ann')

        img_name_list = [i for i in os.listdir(img_dir) if i.endswith('jpg') or i.endswith('png')]
        for img_name in tqdm(img_name_list):

            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            anno_name = img_name + '.json'
            anno_path = os.path.join(ann_dir, anno_name)
            anno_content = load_json(anno_path)
            image_save_dir_path = os.path.join(save_path, 'test')
            if not os.path.exists(image_save_dir_path):
                os.mkdir(image_save_dir_path)
            anno_save_path, img_save_path = save_dir(image_save_dir_path)
            cv2.imwrite(os.path.join(img_save_path, img_name), img)
            json_save_path = os.path.join(anno_save_path, anno_name)
            with open(json_save_path, 'w') as f:
                json.dump(anno_content, f, indent=4)
    print('train_data_completed')

    for img_anno_dir in test_data:
        dir_path = os.path.join(original_path, img_anno_dir)

        img_dir = os.path.join(dir_path, 'img')
        ann_dir = os.path.join(dir_path, 'ann')

        img_name_list = [i for i in os.listdir(img_dir) if i.endswith('jpg') or i.endswith('png')]
        for img_name in tqdm(img_name_list):

            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            anno_name = img_name + '.json'
            anno_path = os.path.join(ann_dir, anno_name)
            anno_content = load_json(anno_path)
            image_save_dir_path = os.path.join(save_path, 'train')
            if not os.path.exists(image_save_dir_path):
                os.mkdir(image_save_dir_path)
            anno_save_path, img_save_path = save_dir(image_save_dir_path)
            cv2.imwrite(os.path.join(img_save_path, img_name), img)
            json_save_path = os.path.join(anno_save_path, anno_name)
            with open(json_save_path, 'w') as f:
                json.dump(anno_content, f, indent=4)
    print('test_data_completed')


if __name__ == '__main__':
    save_path = r"C:\Users\Charl\Downloads\cataract-1k\Annotations\Generated_Dataset"
    data_path = r'C:\Users\Charl\Downloads\cataract-1k\Annotations\Images-and-Supervisely-Annotations'
    # generate(data_path, save_path)
    generate_train_and_test(data_path, save_path)
