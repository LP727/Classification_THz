import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

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

    # img = tf.image.convert_image_dtype(img, tf.float32) # do
    # prerpocessing instead
    image = tf.cast(image, tf.float32)

    image = tf.image.resize_with_pad(
        image, IMG_X_SIZE, IMG_Y_SIZE)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image) #set value between -1 to 1, this is for mobilenetV2

    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
data_path = './data/output/FERETJP'

#image_generation.mirror_folder(resources_path)

train_data_path = os.path.join(data_path,'train')    
test_threshold = 0.85
train_img_list, test_img_list, train_labels, test_labels = misc.gen_lists(data_path, test_threshold)

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
#print(tf.shape(train_labels))
num_train = len(train_img_list)

train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(train_img_list), tf.constant(train_labels)))
train_dataset = train_dataset.map(process_path,
                              num_parallel_calls=AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(test_img_list), tf.constant(test_labels)))
val_dataset = val_dataset.map(process_path,
                              num_parallel_calls=AUTOTUNE)

train_dataset = train_dataset.repeat().shuffle(buffer_size=shuffle_buffer_size).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

first = 16
second = 47
third = 100
prev_max = 0.00

best_config = [first, second, third]

for i in range(1):
    for j in range(1):
        for k in range (1):
            layer1 = first# + k * 2
            layer2 = second# + j * 4
            layer3 = third# + i * 10
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Conv2D(layer1, (6, 6), activation='relu', input_shape=(IMG_X_SIZE, IMG_Y_SIZE, 3)))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(layer2, (6, 6), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((4, 3)))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(layer3, activation='relu'))
            model.add(tf.keras.layers.Dense(2, activation='softmax'))
            model.summary()
            
            base_learning_rate = 0.0001
            model.compile(optimizer=tf.keras.optimizers.RMSprop(
                                  learning_rate=base_learning_rate),
                          loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('./data/output/ckp/FERET_pres.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')
            reduce_lr_acc = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='max')
            # train model 
            history = model.fit(train_dataset,
                                steps_per_epoch=(num_train)//BATCH_SIZE,
                                epochs=initial_epochs,
                                callbacks=[model_checkpoint_callback, reduce_lr_acc],
                                validation_data=val_dataset)
                                #callbacks=[model_checkpoint_callback]
            
            acc = history.history['accuracy'] 
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            max_acc = max(val_acc)
            if (max_acc > prev_max):
                prev_max = max_acc
                best_config = [layer1, layer2, layer3]
                print('New best:')
                print(best_config)
                print(' val: ')
                print(prev_max)
                print('\n')

print('Last best:')
print(best_config)
print('\n')
print(' val: ')
print(prev_max)
print('\n')
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('result.png')
plt.show()

#image_generation.clear_mirrored_folder(resources_path)