import numpy as np
import os
import cv2

def load_dataset(datasetPath):
    images = []
    labels = []
    for root, dirs, files in os.walk(datasetPath):
        label = os.path.basename(os.path.normpath(root))
        for fname in files:
            if fname.startswith('.') and os.path.isfile(os.path.join(root, fname)):
                continue
            image = cv2.imread(os.path.join(root, fname), cv2.IMREAD_GRAYSCALE)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            labels.append(label)
    return images, labels


