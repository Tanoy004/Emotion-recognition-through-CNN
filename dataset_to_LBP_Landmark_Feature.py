
# dataset_to_LBP_Landmark_Feature.py


import numpy as np
import pandas as pd
import os
import argparse
import errno
import scipy.misc
import dlib
import cv2

from skimage.feature import hog
from skimage import feature

# initialization
image_height = 256  # 48
image_width = 256  # 48
ONE_HOT_ENCODING = False
SAVE_IMAGES = False
GET_LANDMARKS = False
GET_HOG_FEATURES = False
SELECTED_LABELS = []
IMAGES_PER_LABEL = 1000000000
OUTPUT_FOLDER_NAME = "jaffe_features"  # "fer2013_features"

# parse arguments and initialize variables:
parser = argparse.ArgumentParser()
parser.add_argument("-j", "--jpg", default="no", help="save images as .jpg files")
parser.add_argument("-l", "--landmarks", default="yes", help="extract Dlib Face landmarks")
parser.add_argument("-ho", "--hog", default="yes", help="extract HOG features")
parser.add_argument("-lbp", "--lbp", default="yes", help="extract LBP features")
parser.add_argument("-o", "--onehot", default="no", help="one hot encoding")
parser.add_argument("-e", "--expressions", default="0,1,2,3,4,5,6", help="choose the faciale expression you want to use: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral")
args = parser.parse_args()


if args.jpg == "yes":
    SAVE_IMAGES = True
if args.landmarks == "yes":
    GET_LANDMARKS = True
if args.hog == "yes":
    GET_HOG_FEATURES = True
if args.lbp == "yes":
    GET_LBP_FEATURES = True
if args.onehot == "yes":
    ONE_HOT_ENCODING = True
if args.expressions != "":
    expressions  = args.expressions.split(",")
    for i in range(0 ,len(expressions)):
        label = int(expressions[i])
        if (label >= 0 and label <= 6):
            SELECTED_LABELS.append(label)
if SELECTED_LABELS == []:
    SELECTED_LABELS = [0 ,1 ,2 ,3 ,4 ,5 ,6]


# loading Dlib predictor and preparing arrays:
print("preparing")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
original_labels = [0, 1, 2, 3, 4, 5, 6]
new_labels = list(set(original_labels) & set(SELECTED_LABELS))
nb_images_per_label = list(np.zeros(len(new_labels), 'uint8'))
try:
    os.makedirs(OUTPUT_FOLDER_NAME)
except OSError as e:
    if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
        pass
    else:
        raise


def get_landmarks(image, rects):
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


def get_new_label(label, one_hot_encoding=False):
    if one_hot_encoding:
        new_label = new_labels.index(label)
        label = list(np.zeros(len(new_labels), 'uint8'))
        label[new_label] = 1
        return label
    else:
        return new_labels.index(label)


print("importing csv file")
# data = pd.read_csv('fer2013_modified.csv')#
data = pd.read_csv('jaffe_pixels.csv'  )#


for category in data['Usage'].unique():

    print("converting set: " + category + "...")
    # create folder
    if not os.path.exists(category):
        try:
            os.makedirs(OUTPUT_FOLDER_NAME + '/' + category)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
                pass
            else:
                raise


    # get samples and labels of the actual category
    category_data = data[data['Usage'] == category]
    samples = category_data['pixels'].values
    labels = category_data['emotion'].values

    print("Folder Created")

    # get images and extract features
    images = []
    labels_list = []
    landmarks = []
    hog_features = []
    hog_images = []
    lbp_features = []
    lbp_images = []

    print(len(samples))
    length = len(samples);

    for i in range(0, length):
        try:
            if labels[i] in SELECTED_LABELS and nb_images_per_label[get_new_label(labels[i])] < IMAGES_PER_LABEL:
                image = np.fromstring(samples[i], dtype=int, sep=" ").reshape((image_height, image_width))
                images.append(image)

                if SAVE_IMAGES:
                    scipy.misc.imsave(category + '/' + str(i) + '.jpg', image)

                if GET_HOG_FEATURES:
                    features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                              cells_per_block=(1, 1), visualise=True)
                    hog_features.append(features)
                    hog_images.append(hog_image)

                if GET_LBP_FEATURES:
                    lbp = feature.local_binary_pattern(image, 24 ,3, method="uniform")
                    n_bins = int(lbp.max() + 1)

                    bins = np.arange(0, n_bins + 1)
                    size = (0, n_bins)
                    (hist, _) = np.histogram(lbp.ravel(), bins, size)

                    hist = np.array(hist)

                    pad = False
                    padding = len(hist)
                    for j in range(0, 26 - padding):
                        pad = True
                        print('padded')
                        print(hist)
                        hist = np.append(hist, 0)
                        print(hist)

                    hist = np.array(hist)
                    hist = hist.astype(np.float)
                    hist /= (hist.sum() + .00000001)

                    # print(hist.shape)
                    # hist=hist.reshape(1,26)
                    # print(hist.shape)

                    lbp_features.append(hist)
                    if pad==True:
                        print(hist)


                if GET_LANDMARKS:
                    scipy.misc.imsave('temp.jpg', image)
                    image2 = cv2.imread('temp.jpg')
                    face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
                    face_landmarks = get_landmarks(image2, face_rects)
                    landmarks.append(face_landmarks)
                labels_list.append(get_new_label(labels[i], one_hot_encoding=ONE_HOT_ENCODING))
                nb_images_per_label[get_new_label(labels[i])] += 1

                if i % 1000==0:
                    val = i/ 1000
                    print(val, ' number of data is featured ', i)
                    print(lbp_features[i])
                    print(len(lbp_features[i]))
                    # print(len(hog_features[i]))

        except Exception as e:
            print(n_bins)
            print(padding)
            print(hist.shape)
            print("error in image: " + str(i) + " - " + str(e))

    print("image and extract features")
    np.save(OUTPUT_FOLDER_NAME + '/' + category + '/images.npy', images)
    if ONE_HOT_ENCODING:
        np.save(OUTPUT_FOLDER_NAME + '/' + category + '/labels.npy', labels_list)
    else:
        np.save(OUTPUT_FOLDER_NAME + '/' + category + '/labels.npy', labels_list)
    if GET_LANDMARKS:
        np.save(OUTPUT_FOLDER_NAME + '/' + category + '/landmarks.npy', landmarks)
    if GET_HOG_FEATURES:
        np.save(OUTPUT_FOLDER_NAME + '/' + category + '/hog_features.npy', hog_features)
        np.save(OUTPUT_FOLDER_NAME + '/' + category + '/hog_images.npy', hog_images)
        print(np.array(hog_features).shape)

    if GET_LBP_FEATURES:
        # lbp_features_ = np.array(lbp_features).reshape(length,26)
        np.save(OUTPUT_FOLDER_NAME + '/' + category + '/lbp_features.npy', lbp_features)
        print(np.array(lbp_features).shape)
        print(np.array(lbp_features).size)

# Command:
# python convert_fer2013_to_images_and_landmarks.py


