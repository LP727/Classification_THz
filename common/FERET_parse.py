import os
import pandas as pd
import shutil


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
        image_path = os.path.join(file_path,f)
        destination = os.path.join(output_path, gender, f)
        dest = shutil.copyfile(image_path, destination)
    return (dest)


if __name__ == '__main__':
    file_path = './resources/FERET_SYS866_2020/FERET_SYS866_2020/'
    csv_path = './resources/FERET_SYS866_2020/FERET_labels_sex.csv'
    output_path = './data/output/FERET/'
    colnames = ['Name', 'Label']

    ID_list, gender_list = read_FERET_csv(csv_path, colnames)
    classify_files(file_path, output_path, ID_list, gender_list)
