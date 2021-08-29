
# images_to_csv.py

from PIL import Image
import numpy as np
import os, os.path, time, sys
import argparse
from imutils import paths
import cv2
import csv


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	            help="path to input image")
args = vars(ap.parse_args())
if args == None:
    raise Exception("could not load image !")

with open("dataset/jaffe_pixels.csv", 'a') as csvfile:
    row = ['emotion', 'pixels', 'Usage']
    writer = csv.writer(csvfile)
    writer.writerow(row)

for inputPath in paths.list_images(args["input"]):

    image = cv2.imread(inputPath)

    label = (inputPath.split("/")[-2])

    print(label)

    usage = 'Training'

    width, height = image.shape[:2]

    # print(image[:, :])

    print(width)
    print(height)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    value = gray
    print(value.shape)

    value = value.flatten()

    print(len(value))

    n_value = ' '.join(map(str, value))

    row = [label, n_value, usage]
    # print(value)


    with open("dataset/jaffe_pixels.csv", 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)


# Command:
# python3 images_to_csv.py --input dataset/expression/jaffe

# in image folder different label images should be saved in different folder named 0 to 6
