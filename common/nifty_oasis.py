import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pyplot as plt
from cv2 import cv2

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

def img_to_jpg(img_path, slice_level, axis):
    img = nib.load(img_path)
    img = nib.load(img_path)
    img_data = img.get_fdata()
    img_data = img_data[:, :, :,0]
    if axis == 0:
        jpg = img_data[slice_level, :, :]
    elif axis == 1:
        jpg = img_data[:, slice_level, :]
    elif axis == 2:
        jpg = img_data[:, :, slice_level]
    else:
        pass 
    return(jpg)

def extract_jpg(folder_path, file_location, detection_str, output_path, slice_level: int = 90, axis: int=2):
    for f in os.listdir(folder_path):
        folder = os.path.join(folder_path,f,file_location)
        for i in os.listdir(folder):
            img_path = os.path.join(folder,i)
            if img_path.find(detection_str) != -1:
                jpg = img_to_jpg(img_path, slice_level, axis)
                ext = os.path.splitext(i)
                jpg_name = ext[0] + '.jpg'
                fig = plt.figure(frameon=False)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(jpg, cmap="gray", aspect='auto')
                fig.savefig(os.path.join(output_path,jpg_name))
                plt.close()
                #cv2.imwrite(os.path.join(output_path,jpg_name), jpg)

def display_roi(folder_path, file_location, detection_str, upper_thres: int = 80, lower_thresh: int=100):
    for f in os.listdir(folder_path):
        folder = os.path.join(folder_path,f,file_location)
        for i in os.listdir(folder):
            img_path = os.path.join(folder,i)
            if img_path.find(detection_str) != -1:
                img = nib.load(img_path)
                img_data = img.get_fdata()
                img_data = img_data[:, :, :,0]
                print(img_data.shape)
                print('\n')

                for j in range (80,100):
                    slice_0 = img_data[j, :, :]
                    slice_1 = img_data[:, j, :]
                    slice_2 = img_data[:, :, j]
                    show_slices([slice_0, slice_1, slice_2])
                    plt.suptitle("Center slices for EPI image %i" %j) 
                    plt.show()

if __name__ == '__main__':
    folder_path = './resources/OASIS/oasis_cross-sectional_disc1/disc1'
    file_location = 'PROCESSED/MPRAGE/T88_111'
    detection_str = 'masked_gfc.img'

    output_path = './data/output/'

    #display_roi(folder_path, file_location, detection_str)
    extract_jpg(folder_path,file_location, detection_str, output_path)


