import os
from cv2 import cv2

def gen_mirrored_img(folder_path: str, mirror_type: int = 1):
    for img in os.listdir(folder_path):
        original_name = os.path.join(folder_path,img)
        originalImage = cv2.imread(original_name)
        ext = os.path.splitext(img)

        mirrored = cv2.flip(originalImage, mirror_type)
        mirrored_name = ext[0] + '_m' + ext[1] 

        cv2.imwrite(os.path.join(folder_path,mirrored_name), mirrored)

def mirror_folder(path: str, mirror_type: int = 1):
    for sub_folder in os.listdir(path):
           gen_mirrored_img(os.path.join(path,sub_folder), mirror_type)

def clear_mirrored_folder(path: str):
    for sub_folder in os.listdir(path):
        for img in os.listdir(os.path.join(path,sub_folder)):
            if (os.path.join(path,sub_folder,img)).find("_m") != -1:
                os.remove(os.path.join(path,sub_folder,img))

if __name__ == '__main__':
    cs_path = './resources/'

    mirror_folder(cs_path)
    clear_mirrored_folder(cs_path)