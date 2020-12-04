import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from common import misc

def get_model_name(k):
    return 'model_'+str(k)+'.h5'

#Key parameters
shuffle_buffer_size = 100
BATCH_SIZE = 8
IMG_X_SIZE = 160
IMG_Y_SIZE = 160
initial_epochs = 1000

resources_path = './resources/brains/'
data_path = './data/'
healthy_dir = os.path.join(resources_path, 'Healthy')
alz_dir = os.path.join(resources_path, 'Alzheimer')

train_data_path = os.path.join(data_path,'train')    
test_threshold = 0.85
train_img_list, test_img_list, train_labels, test_labels = misc.gen_lists(resources_path, test_threshold)

misc.setup_data_dir(train_img_list, train_labels, data_path)

num_train = len(train_img_list)
num_test = len(test_img_list)
num_epochs = 10

misc.setup_data_dir(train_img_list, train_labels , data_path)

image_list = misc.list_files(data_path, complete_path= False)
Y =[]
for i in image_list:
    if (i.find("_label0")) != -1:
        Y.append(0)
    elif (i.find("_label1")) != -1:
        Y.append(1)

train_data = misc.gen_dataframe( image_list, Y)
train_data['label'] = train_data['label'].astype('str')

kf = KFold(n_splits = 5)               
skf = StratifiedKFold(n_splits = 5, random_state = 7, shuffle = True) 

idg = ImageDataGenerator(width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.3,
    fill_mode='nearest',
    horizontal_flip = True,
    rescale=1./255)

VALIDATION_ACCURACY = []
VALIDATION_LOSS = []

save_dir = './saved_models/'
fold_var = 1

for train_index, val_index in skf.split(np.zeros(len(Y)),Y):
    training_data = train_data.iloc[train_index]
    validation_data = train_data.iloc[val_index]

    train_data_generator = idg.flow_from_dataframe(training_data, directory = data_path,
                            target_size=(IMG_X_SIZE, IMG_Y_SIZE),
                            x_col = "filename", y_col = "label",
                            class_mode = "binary", shuffle = True)

    valid_data_generator  = idg.flow_from_dataframe(validation_data, directory = data_path,
                            target_size=(IMG_X_SIZE, IMG_Y_SIZE),
    						x_col = "filename", y_col = "label",
    						class_mode = "binary", shuffle = True)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(IMG_X_SIZE, IMG_Y_SIZE, 3)))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(25, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='softmax'))

    model.summary()

    # COMPILE NEW MODEL
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(
        learning_rate=base_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])   

    # CREATE CALLBACKS
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var), 
                            monitor='val_accuracy', verbose=1, 
    						save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # There can be other callbacks, but just showing one because it involves the model name
    # This saves the best model
    # FIT THE MODEL
    history = model.fit(train_data_generator,
    		    epochs=(num_train)//BATCH_SIZE,
    		    callbacks=callbacks_list,
    		    validation_data=valid_data_generator)

    # LOAD BEST MODEL to evaluate the performance of the model
    model.load_weights("./saved_models/model_"+str(fold_var)+".h5")

    results = model.evaluate(valid_data_generator)
    results = dict(zip(model.metrics_names,results))
    
    VALIDATION_ACCURACY.append(results['accuracy'])
    VALIDATION_LOSS.append(results['loss'])
    
    tf.keras.backend.clear_session()
    fold_var += 1