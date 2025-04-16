import numpy as np
import cv2
import os
import json

def pic_viewer(dictionary_path):
    pic_list = os.listdir(dictionary_path)
    pic_path = [os.path.join(dictionary_path, i) for i in pic_list]
    for pic in pic_path:
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
            img_path = os.path.join(img_folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            if True:
                kpts = content['objects'][0]['points']['exterior']
                print(content)
                print(img_name)
                print(np.array(kpts).shape)
                for kpt_idx in range(len(kpts)):
                    x, y = kpts[kpt_idx][0], kpts[kpt_idx][1]
                    cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                exit(0)
            interior = content['objects'][0]['points']['interior']
            if len(interior) != 0:
                print(interior)
            # exit(0)





if __name__ == '__main__':
    test_path = r'C:\Users\Charl\Downloads\Cataract_Seg\Mask_segmentation\Mask_segmentation\Cornea_detection\dataset'
    img_folder = os.path.join(test_path, 'train')
    anno_folder = os.path.join(r'C:\Users\Charl\Downloads\Cataract_Seg\Mask_segmentation\Mask_segmentation\Cornea_detection', 'annotations')
    # pic_viewer(img_folder)
    anno_check(anno_folder, img_folder)