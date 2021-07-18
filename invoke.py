
from ocr_engine.models.resnet import load_model
import cv2
import numpy as np
import string

N_CLASS = 36
_labelNames = string.digits + string.ascii_lowercase
_labelNames = [l for l in _labelNames]


# NOTE: ResNet Camstroke Model must have white background and black foreground
model = load_model(path_to_weight="./results/ocr_13072021.model", n_class=N_CLASS)
model.summary()

def read_image(path):
    # read image as grayscale
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im = cv2.bitwise_not(im)
    im = cv2.resize(im, (32, 32))
    im = np.array(im, dtype="float32")
    # convert to rank 4 tensor for single image prediction
    # model's input shape must be (batch_size, width, height, channel)
    im = np.expand_dims(im, axis=0)
    # add a channel dimension to every image in the dataset and scale the
    # pixel intensities of the images from [0, 255] down to [0, 1]
    im = np.expand_dims(im, axis=-1)
    im /= 255.0

    return im

def predict():
    im = read_image("../Results/captured_keystroke_1.png")
    probs = model.predict(im)
    prediction = probs.argmax(axis=1)
    label = _labelNames[prediction[0]]
    return label

print(predict())