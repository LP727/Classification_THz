import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from common import misc
from common import image_generation

#Key parameters
shuffle_buffer_size = 100
BATCH_SIZE = 16
IMG_X_SIZE = 160
IMG_Y_SIZE = 160
initial_epochs = 30

def process_path(file_path, label):

    # load the raw data from the file as a string
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # prerpocessing 
    image = tf.cast(image, tf.float32)

    image = tf.image.resize_with_pad(
        image, IMG_X_SIZE, IMG_Y_SIZE)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image) #set value between -1 to 1

    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
data_path = './data/output/FERETJP'

train_data_path = os.path.join(data_path,'train')    
test_threshold = 0.85
train_img_list, test_img_list, train_labels, test_labels = misc.gen_lists(data_path, test_threshold)

test_labels = tf.keras.utils.to_categorical(test_labels)

num_train = len(train_img_list)

test_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(test_img_list), tf.constant(test_labels)))
test_dataset = test_dataset.map(process_path,
                              num_parallel_calls=AUTOTUNE)

test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

basic_model = tf.keras.models.load_model('./data/output/rep/FERET_soft_no_red.hdf5')
rl_model = tf.keras.models.load_model('./data/output/ckp/FERET.hdf5')

basic_y_pred = basic_model.predict(test_dataset).ravel()
rl_y_pred = rl_model.predict(test_dataset).ravel()

y_true = []
for i in test_labels:
    y_true.append(i[0])
    y_true.append(i[1])

basic_fpr, basic_tpr, basic_thresholds = roc_curve(y_true, basic_y_pred)
basic_auc = auc(basic_fpr, basic_tpr)

rl_fpr, rl_tpr, rl_thresholds = roc_curve(y_true, rl_y_pred)
rl_auc = auc(rl_fpr, rl_tpr)

plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(basic_fpr, basic_tpr, label='FERET model without adjusted learning rate(area = {:.3f})'.format(basic_auc))
plt.plot(rl_fpr, rl_tpr, label='FERET model with adjusted learning rate (area = {:.3f})'.format(rl_auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()