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

val_labels = test_labels
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

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

layer1 = 16
layer2 = 47
layer3 = 100
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
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('./data/output/ckp/FERET_rep.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')
#reduce_lr_acc = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=7, verbose=1, min_delta=1e-4, mode='max')
# train model 
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



y_pred = model.predict(val_dataset).ravel()
y_true = []
for i in test_labels:
    y_true.append(i[0])
    y_true.append(i[1])

fpr, tpr, thresholds = roc_curve(y_true, y_pred)
auc_model = auc(fpr, tpr)

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

plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='FERET model (area = {:.3f})'.format(auc_model))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
