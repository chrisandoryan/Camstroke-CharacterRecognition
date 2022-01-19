import numpy as np
import matplotlib.pyplot as plt
from imutils import build_montages
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from ocr_engine.models.resnet import ResNet, create_model
from ocr_engine.dataset.helpers import load_az_dataset, load_dataset, load_mnist_dataset, stack_dataset
import argparse

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to Camstroke dataset")
args = vars(ap.parse_args())

print("[INFO] Loading datasets...")
camstrokeImages, camstrokeLabels = load_dataset(args['dataset'])
assert len(camstrokeImages) == len(camstrokeLabels)

# azImages, azLabels = load_az_dataset(args["az"])
# digitImages, digitLabels = load_mnist_dataset()

# images, labels = stack_dataset((camstrokeImages, azImages), (camstrokeLabels, azLabels))
images, labels = (camstrokeImages, camstrokeLabels)

print("[INFO] Total data: ", len(images))

# define the list of label names
labelNames = np.unique(labels)
print(labelNames)

# initialize the number of epochs to train for, initial learning rate,
# and batch size, read them from config file.
from configparser import ConfigParser
config = ConfigParser()
config.read('./config/config.ini')

EPOCHS = config.getint('resnet', 'EPOCHS')
INIT_LR = config.getfloat('resnet', 'INIT_LR')
BS = config.getint('resnet', 'BS')

# each image in the datasets are 28x28 pixels;
# however, the architecture we're using is designed for 32x32 images,
# so we need to resize them to 32x32
images = [cv2.resize(image, (32, 32)) for image in images]
images = np.array(images, dtype="float32")

# add a channel dimension to every image in the dataset and scale the
# pixel intensities of the images from [0, 255] down to [0, 1]
images = np.expand_dims(images, axis=-1)
images /= 255.0

# print(type(images))
# input()

# convert the labels from integers to vectors
le = LabelBinarizer()
labels = le.fit_transform(labels)
counts = labels.sum(axis=0)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = {}

# loop over all classes and calculate the class weight
for i in range(0, len(classTotals)):
	classWeight[i] = classTotals.max() / classTotals[i]

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(images,
                                                  labels, test_size=0.20, stratify=labels, random_state=42)
(testX, valX, testY, valY) = train_test_split(testX,
                                                  testY, test_size=0.50, stratify=testY, random_state=42)
# construct the image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	fill_mode="nearest")

# initialize and compile our deep neural network
print("[INFO] Compiling model...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = create_model(n_class=len(le.classes_))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] Training network...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(valX, valY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS,
	class_weight=classWeight,
	verbose=1)

# evaluate the network
print("[INFO] Evaluating network...")
# labelNames = [chr(l + 97) for l in labelNames]
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

# save the model to disk
print("[INFO] serializing network...")
model.save("ocr.model", save_format="h5")

# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

# initialize our list of output test images
images = []
# randomly select a few testing characters
for i in np.random.choice(np.arange(0, len(testY)), size=(49,)):
	# classify the character
	probs = model.predict(testX[np.newaxis, i])
	prediction = probs.argmax(axis=1)
	label = labelNames[prediction[0]]
	# extract the image from the test data and initialize the text
	# label color as green (correct)
	image = (testX[i] * 255).astype("uint8")
	color = (0, 255, 0)
	# otherwise, the class label prediction is incorrect
	if prediction[0] != np.argmax(testY[i]):
		color = (0, 0, 255)
	# merge the channels into one image, resize the image from 32x32
	# to 96x96 so we can better see it and then draw the predicted
	# label on the image
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
		color, 2)
	# add the image to our list of output images
	images.append(image)
# construct the montage for the images
montage = build_montages(images, (96, 96), (7, 7))[0]
# show the output montage
cv2.imshow("OCR Results", montage)
# cv2.waitKey(0)
