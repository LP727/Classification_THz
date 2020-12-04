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
initial_epochs = 1
fine_tune_epochs = 10
epochs_per_saves = 2
total_epochs =  initial_epochs + fine_tune_epochs
num_loop = int(fine_tune_epochs/epochs_per_saves)

def get_model_name(k):
    return 'model_'+str(k)+'.h5'

def process_path(file_path, label):

    # load the raw data from the file as a string
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # prerpocessing instead
    image = tf.cast(image, tf.float32)

    image = tf.image.resize_with_pad(
        image, IMG_X_SIZE, IMG_Y_SIZE)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image) #set value between -1 to 1, this is for mobilenetV2

    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
data_path = './data/'
resources_path = './resources/brains/'
save_dir = './saved_models/'
fold_var = 2

image_generation.clear_mirrored_folder(resources_path)
#image_generation.mirror_folder(resources_path)

train_data_path = os.path.join(data_path,'train')    
val_threshold = 0.85
train_img_list, val_img_list, train_labels, val_labels = misc.gen_lists(resources_path, val_threshold)

num_train = len(train_img_list)

train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(train_img_list), tf.constant(train_labels)))
train_dataset = train_dataset.map(process_path,
                              num_parallel_calls=AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(val_img_list), tf.constant(val_labels)))
val_dataset = val_dataset.map(process_path,
                              num_parallel_calls=AUTOTUNE)


train_dataset = train_dataset.repeat().shuffle(buffer_size=shuffle_buffer_size).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
# 
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

base_model = tf.keras.applications.vgg16.VGG16(input_shape=(IMG_X_SIZE, IMG_Y_SIZE, 3),
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(
                      learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train model head.
history = model.fit(train_dataset,
                    steps_per_epoch=(num_train)//BATCH_SIZE,
                    epochs=initial_epochs,
                    validation_data=val_dataset)
                        # Train whole model
base_model.trainable = True
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

model.save_weights((save_dir+get_model_name(fold_var)).format(epoch=0))   
model.evaluate(val_dataset)
for i in range(num_loop):

    #create checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var), 
                            monitor='val_accuracy', verbose=1, 
                            save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history_fine = model.fit(train_dataset,
                             steps_per_epoch=(num_train)//BATCH_SIZE,
                             epochs=epochs_per_saves,
                             initial_epoch=history.epoch[-1],
                             validation_data=val_dataset)

    # LOAD BEST MODEL to evaluate the performance of the model
    model.load_weights("./saved_models/model_"+str(fold_var)+".h5")

acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.5, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('result.png')
plt.show()

image_generation.clear_mirrored_folder(resources_path)