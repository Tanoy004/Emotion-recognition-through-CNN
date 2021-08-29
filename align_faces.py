
# align_faces.py

# import the necessary packages

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import paths
import argparse
import imutils
import dlib
import cv2
import os


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--input", required=True,
	            help="path to input image")
args = vars(ap.parse_args())

if args == None:
    raise Exception("could not load image !")


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256, desiredFaceHeight=256)


print("Image pre-processing is starting. Aligning image according to facial landmarks.")
# loop over the input images
for inputPath in paths.list_images(args["input"]):
    # load the image, convert it to grayscale, and describe it
    image = cv2.imread(inputPath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale image
    rects = detector(gray, 2)

    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        print("path = ", inputPath)

        faceAligned = fa.align(image, gray, rect)
        print(inputPath.split("/")[-1])
        print(inputPath.split("/")[-2])

        # write the output image to disk
        path = '/media/ankur/Administrator/Administration/My_Codes/Python/ml/dataset/output'

        cv2.imwrite(os.path.join(path, inputPath.split("/")[-1]), faceAligned)

        # display the output images
        cv2.imshow("Aligned", faceAligned)

        cv2.waitKey(1)

print("Image face alignment is completed.")



# Command:
# python3 align_faces.py --shape-predictor dataset/shape_predictor_68_face_landmarks.dat --input dataset/input

# Reference:
# https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/