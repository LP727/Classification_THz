import os
import pandas as pd
import shutil


def get_ID(file_name):
    ext = os.path.splitext(file_name)
    ID = ext[0]
    return(ID[3:9])

def read_HCP_csv(csv_path, colnames):
    data = pd.read_csv(csv_path, names= colnames)
    ID_list = data.Subject_Code.tolist()
    ID_list = ID_list[1:]
    gender_list = data.Sex.tolist()
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
            gender = 'M'
        elif g == '1':
            gender = 'F'
        image_path = os.path.join(file_path,f)
        destination = os.path.join(output_path, gender, f)
        dest = shutil.copyfile(image_path, destination)
    return (dest)


if __name__ == '__main__':
    file_path = './resources/HCP_SYS866_2020/HCP_SYS866_2020/'
    csv_path = './resources/HCP_SYS866_2020/HCP_SYS866_2020_labels.csv'
    output_path = './data/output/HCP/'
    colnames = ['Subject_Code', 'Age_in_Yrs', 'ZygositySR', 'Mother_Code', 'Father_Code', 'Handedness', 'Height', 'Weight', 'Sex']
    ID_list, gender_list = read_HCP_csv(csv_path, colnames)
    classify_files(file_path, output_path, ID_list, gender_list)