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

def grad_cam(input_model, image, cls, layer_name):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = tf.keras.backend.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    # grads = normalize(grads)
    gradient_function = tf.keras.backend.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (IMG_X_SIZE, IMG_Y_SIZE), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam_max = cam.max() 
    if cam_max != 0: 
        cam = cam / cam_max
    return cam

def prepare(filepath, size_X, size_Y):
    img_array = cv2.imread(filepath, cv2.COLOR_RGB2BGR)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (size_X, size_Y))  # resize image to match model's expected sizing
    return new_array.reshape(-1, size_X, size_Y, 3)  # return the image with shaping that TF wants.

MALE_PATH = './data/test/Val_HCP/test_1.png'
FEMALE_PATH = './data/test/Val_HCP/test_174.png'
LAYER_NAME = 'conv2d_1'
MALE_CLASS_INDEX = 0
FEMALE_CLASS_INDEX = 1

img_out = tf.keras.preprocessing.image.load_img(MALE_PATH, target_size=(IMG_X_SIZE, IMG_Y_SIZE))
img_out = tf.keras.preprocessing.image.img_to_array(img_out)

img1, label = process_path(MALE_PATH, MALE_CLASS_INDEX)
img2, label = process_path(FEMALE_PATH, FEMALE_CLASS_INDEX)
img = np.reshape(img1, [1, 160, 160, 3])
model = tf.keras.models.load_model('./data/output/ckp/HCP.hdf5')
#model.summary()

prediction = model.predict([img])
print(prediction)

grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img)
    loss = predictions[:, MALE_CLASS_INDEX]
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



cv2.imwrite('./data/test/male_brain_cam.png', output_image)

#othercam = grad_cam(model, img, FEMALE_CLASS_INDEX, LAYER_NAME)
plt.figure(figsize=(15, 10))
plt.subplot(131)
plt.title('Male: ' + str(prediction[:,0]) + ' Female: ' + str(prediction[:,1]))
plt.axis('off')
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.show()