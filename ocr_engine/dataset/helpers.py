import numpy as np
import os
import cv2
from tensorflow.keras import datasets
from tensorflow.keras.datasets import mnist

def load_dataset(datasetPath):
	images = []
	labels = []
	for root, dirs, files in os.walk(datasetPath):
		label = os.path.basename(os.path.normpath(root))
		for fname in files:
			if fname.startswith('.') and os.path.isfile(os.path.join(root, fname)):
				continue
			image = cv2.imread(os.path.join(root, fname), cv2.IMREAD_GRAYSCALE)
			image = cv2.resize(image, (28, 28))
			image = np.array(image, dtype="float32")
			images.append(image)
			labels.append(ord(label) - 97)

	return images, labels

def load_az_dataset(datasetPath):
	# initialize the list of data and labels
	images = []
	labels = []
	# loop over the rows of the A-Z handwritten digit dataset
	for row in open(datasetPath):
		# parse the label and image from the row
		row = row.split(",")
		label = int(row[0])
		image = np.array([int(x) for x in row[1:]], dtype="uint8")
		# images are represented as single channel (grayscale) images
		# that are 28x28=784 pixels -- we need to take this flattened
		# 784-d list of numbers and repshape them into a 28x28 matrix
		image = image.reshape((28, 28))
		# update the list of data and labels
		images.append(image)
		labels.append(label)

	# convert the data and labels to NumPy arrays
	images = np.array(images, dtype="float32")
	labels = np.array(labels, dtype="int")

	# return a 2-tuple of the A-Z images and labels
	return (images, labels)

def load_mnist_dataset():
	# load the MNIST dataset and stack the training data and testing
	# data together (we'll create our own training and testing splits
	# later in the project)
	((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
	images = np.vstack([trainData, testData])
	labels = np.hstack([trainLabels, testLabels])
	# return a 2-tuple of the MNIST data and labels
	return (images, labels)

def stack_dataset(datasets, labels):
	# # the MNIST dataset occupies the labels 0-9, so let's add 10 to every
	# # A-Z label to ensure the A-Z characters are not incorrectly labeled
	# # as digits

	# # stack the A-Z data and labels with the MNIST digits data and labels
	images = np.vstack(list(datasets))
	labels = np.hstack(list(labels))

	return images, labels