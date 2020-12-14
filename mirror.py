import os
from cv2 import cv2
from common import misc
from common import image_generation

if __name__ == '__main__':
    cs_path = './data/test/Val_HCP/'

    image_generation.gen_mirrored_img(cs_path)