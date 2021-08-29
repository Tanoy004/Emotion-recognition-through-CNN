
# data_labels_file.py

# import the necessary packages
from lbp1_localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
from time import time
import pickle
import numpy as np


start_time=time()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
                help="path to the training images")
args = vars(ap.parse_args())

if args == None:
    raise Exception("could not load image !")


# the data and label lists
data2 = []
labels = []
w = 256  # 48 for FER2013
h = 256

# print(data2.shape)


# loop over the training images
for imagePath in paths.list_images(args["training"]):
    # load the image, convert it to grayscale, and describe it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    pixels = gray.flatten().reshape(h*w, )
    # print("data2 ", pixels.shape)

    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split("/")[-2])
    data2.append(pixels)



print(len(data2))
val = np.array(data2)
print(val.shape)

f = open('dataset/' + 'data_plots' + '.pckl', 'wb')

pickle.dump([data2, labels], f)
f.close()

print('dumped')


# Command:
# python3 data_labels_file.py --training dataset/ck+/