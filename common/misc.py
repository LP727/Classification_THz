#misc.py

import os
import shutil
import pandas as pd

def list_files(path: str, complete_path: bool = False,
               hidden_files: bool = False):

    if hidden_files:
        def isfile_condition(p, f): return os.path.isfile(os.path.join(p, f))
    else:
        def isfile_condition(
            p, f): return os.path.isfile(
            os.path.join(
                p, f)) and not f.startswith('.')

    if complete_path:
        def result_format(p, f): return os.path.join(p, f)
    else:
        def result_format(p, f): return f

    file_list = [
        result_format(
            path,
            f) for f in os.listdir(path) if isfile_condition(
            path,
            f)]
    return sorted(file_list)

def gen_dataframe(images: list, labels: list):
    # dictionary of lists
    dict = {'filename': images, 'label': labels}

    df = pd.DataFrame(dict)
    print(df.iloc[0])
    return df

def setup_data_dir(images_path: list, labels: list, data_dir: str):
    id = 0
    for img, labels in zip(images_path, labels):
            ext = os.path.splitext(img)
            name = 'img' + str(id) + '_label' + str(labels) + ext[1]
            shutil.copyfile(img,os.path.join(data_dir, name))
            id += 1
    return data_dir

def gen_lists(data_path: str, test_threshold: float):
    img_test_list = []
    img_train_list = []
    label_test_list = []
    label_train_list = []
    label = 0
    for folder in os.listdir(data_path):
        files = list_files(os.path.join(data_path, folder), complete_path= True)
        num_files = len(files)
        threshold = round(num_files * test_threshold)
        labels = ([label] * num_files)

        img_train_list = img_train_list + files[:threshold]
        img_test_list = img_test_list + files[threshold:]
        label_train_list = label_train_list + labels[:threshold]
        label_test_list = label_test_list + labels[threshold:]

        label +=1

    return img_train_list, img_test_list, label_train_list, label_test_list

def clear_data_folder(data_path: str):
    for folder in os.listdir(data_path):
        for f in  os.listdir(os.path.join(data_path, folder)):
            os.remove(os.path.join(data_path, folder, f))

if __name__ == '__main__':
    resources_path = './resources/'
    data_path = './data/'
    train_data_path = os.path.join(data_path,'train')
    
    test_threshold = 0.85

    train_img_list, test_img_list, train_labels, test_labels = gen_lists(resources_path, test_threshold)

    setup_data_dir( train_img_list, train_labels,train_data_path)

    clear_data_folder(data_path)