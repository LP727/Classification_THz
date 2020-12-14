from cv2 import cv2
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
IMG_X_SIZE = 160
IMG_Y_SIZE = 160

def process_path(file_path, label):

    # load the raw data from the file as a string
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # prerpocessing instead
    image = tf.cast(image, tf.float32)

    image = tf.image.resize_with_pad(
        image, IMG_X_SIZE, IMG_Y_SIZE)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image) #set value between -1 to 1

    return image, label

LAYER_NAME = 'conv2d_1'
MALE_CLASS_INDEX = 1
FEMALE_CLASS_INDEX = 0
data_path = './data/test/Val_mrd_HCP'
output_path = './data/test/report_mrd_HCP'
image_files = os.listdir(data_path)
idx = 0
thres = len(image_files) / 2

for file_name in os.listdir(data_path):
    if idx < thres:
        CLASS = FEMALE_CLASS_INDEX
    else:
        CLASS = MALE_CLASS_INDEX

    img_out = tf.keras.preprocessing.image.load_img(os.path.join(data_path,file_name), target_size=(IMG_X_SIZE, IMG_Y_SIZE))
    img_out = tf.keras.preprocessing.image.img_to_array(img_out)

    img, label = process_path(os.path.join(data_path,file_name), FEMALE_CLASS_INDEX)
    img = np.reshape(img, [1, 160, 160, 3])
    model = tf.keras.models.load_model('./data/output/ckp/HCP.hdf5')
    #model.summary()

    prediction = model.predict([img])
    print(prediction)

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, CLASS]
        #print(loss)

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    cam = np.ones(output.shape[0: 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam.numpy(), (IMG_X_SIZE, IMG_Y_SIZE))
    cam = np.maximum(cam, 0)

    check1 = (cam - cam.min())
    check2 = (cam.max() - cam.min())

    heatmap = (cam - cam.min()) / (cam.max() - cam.min())

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

    output_image = cv2.addWeighted(cv2.cvtColor(img_out.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)



    cv2.imwrite('./data/test/something_grad_cam.png', output_image)

    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.title('Female: ' + str(prediction[:,0]) + ' Male: ' + str(prediction[:,1]))
    plt.axis('off')
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), alpha=0.9)
    plt.subplot(1, 2, 2)
    plt.title('Originale')
    plt.axis('off')
    plt.imshow(img_out.astype('uint8'), alpha=0.9)
    plt.savefig(os.path.join(output_path,file_name))
    #plt.show()