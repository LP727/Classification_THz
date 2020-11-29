import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from common import misc
from common import image_generation

#Key parameters
shuffle_buffer_size = 100
BATCH_SIZE = 8
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
data_path = './data/output/HCP'

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

model_vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
inputt =  tf.keras.Input(shape=(IMG_X_SIZE, IMG_Y_SIZE, 3),name = 'image_input')
output_vgg16_conv = model_vgg16(inputt)

x = tf.keras.layers.Flatten(name='flatten')(output_vgg16_conv)
x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
x = tf.keras.layers.Dense(2, activation='softmax', name='predictions')(x)

model = tf.keras.Model(inputs=inputt, outputs=x)

model.summary()

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(
                      learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('./data/output/ckp/VGGHCP.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')
# train model head.
history = model.fit(train_dataset,
                    steps_per_epoch=(num_train)//BATCH_SIZE,
                    epochs=initial_epochs,
                    callbacks=[model_checkpoint_callback],
                    validation_data=val_dataset)
                    #callbacks=[model_checkpoint_callback]

acc = history.history['accuracy'] 
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
max_acc = max(val_acc)

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