from cv2 import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
IMG_X_SIZE = 160
IMG_Y_SIZE = 160

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

MALE_PATH = './data/test/MALE.jpg'
FEMALE_PATH = './data/test/Maria_forehead.jpg'
LAYER_NAME = 'conv2d_1'
MALE_CLASS_INDEX = 1
FEMALE_CLASS_INDEX = 0

img_out = tf.keras.preprocessing.image.load_img(FEMALE_PATH, target_size=(IMG_X_SIZE, IMG_Y_SIZE))
img_out = tf.keras.preprocessing.image.img_to_array(img_out)

img1, label = process_path(MALE_PATH, MALE_CLASS_INDEX)
img2, label = process_path(FEMALE_PATH, FEMALE_CLASS_INDEX)
img = np.reshape(img2, [1, 160, 160, 3])
model = tf.keras.models.load_model('./data/output/ckp/FERET.hdf5')
#model.summary()

prediction = model.predict([img])
print(prediction)

grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img)
    loss = predictions[:, FEMALE_CLASS_INDEX]
    print(loss)
#print(conv_outputs)
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



cv2.imwrite('./data/test/Forehead_grad_cam.png', output_image)

plt.figure(figsize=(15, 10))
plt.title('Female: ' + str(prediction[:,0]) + ' Male: ' + str(prediction[:,1]))
plt.axis('off')
#plt.imshow(load_image(img_path, preprocess=False))
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.show()