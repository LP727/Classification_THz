import os
import pandas as pd
import shutil
from cv2 import cv2

def ppm_to_jpg(file_path, output_path):
    for f in os.listdir(file_path):
        ext = os.path.splitext(f)
        i = cv2.imread(os.path.join(file_path,f))
        
        output_name = ext[0] + '.jpg'
        cv2.imwrite(os.path.join(output_path,output_name),i)

def get_ID(file_name):
    ID = file_name + '.jpg'
    return(ID)

def read_FERET_csv(csv_path, colnames):
    data = pd.read_csv(csv_path, names= colnames)
    ID_list = data.Name.tolist()
    ID_list = ID_list[1:]
    gender_list = data.Label.tolist()
    gender_list = gender_list[1:]
    return(ID_list,gender_list)

def find_gender(file_name, ID_list, gender_list):
    gender = 'X'
    for ID, gen in zip(ID_list,gender_list):
        if file_name == ID:
            gender = gen
    return(gender)

def classify_files(file_path, output_path, ID_list, gender_list):
    for f in os.listdir(file_path):
        f_ID = get_ID(f)
        g = find_gender(f_ID, ID_list, gender_list)
        if g == '0':
            gender = 'F'
        elif g == '1':
            gender = 'M'
        else:
            continue
        image_path = os.path.join(file_path,f)
        destination = os.path.join(output_path, gender, f)
        dest = shutil.copyfile(image_path, destination)
    return (dest)

def classify_files_V2(file_path, output_path, ID_list, gender_list):
    for ID, Label in zip(ID_list,gender_list):
        if Label == '0':
            gender = 'F'
        elif Label == '1':
            gender = 'M'
        ext = os.path.splitext(ID)
        image_path = os.path.join(file_path,ext[0])
        destination = os.path.join(output_path, gender, ext[0])
        dest = shutil.copyfile(image_path, destination)
    return (dest)

def convert_folder(folder_path, output_path):
    for folder in os.listdir(folder_path):
        file_input_path = os.path.join(folder_path,folder)
        file_output_path = os.path.join(output_path,folder)
        ppm_to_jpg(file_input_path, file_output_path)

if __name__ == '__main__':
    file_path = './resources/FERET_SYS866_2020/FERET_SYS866_2020/'
    csv_path = './resources/FERET_SYS866_2020/FERET_labels_sex.csv'
    output_path = './data/output/FERET/'
    jpg_path = './data/output/FERETJP'
    colnames = ['Name', 'Label']

    ID_list, gender_list = read_FERET_csv(csv_path, colnames)
    #print(len(ID_list))
    classify_files_V2(file_path, output_path, ID_list, gender_list)

    convert_folder(output_path,jpg_path)
