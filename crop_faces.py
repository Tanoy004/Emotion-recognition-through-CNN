
# crop_faces.py

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import paths
from PIL import Image
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


print("Image pre-processing is starting. Cropping image according to facial landmarks.")
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

        (x, y, w, h) = rect_to_bb(rect)

        faceOrig = imutils.resize(gray[y:y + h, x:x + w])
        faceOrig = imutils.resize(gray, width=256, height=256)
        print("path = ", inputPath,' of ', faceOrig.shape, faceOrig.dtype)


        # write the output image to disk
        path = '/media/ankur/Administrator/Administration/My_Codes/Python/ml/dataset/output'

        cv2.imwrite(os.path.join(path, inputPath.split("/")[-1]), faceOrig)

        # display the output images
        cv2.imshow("Original", faceOrig)

        cv2.waitKey(1)

print("Image cropping is completed.")

for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        print(os.path.join(root, name))
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".jpg":
            if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".png"):
                print("A png file already exists for %s" % name)
            # If a png is *NOT* present, create one from the tiff.
            else:
                outfile = os.path.splitext(os.path.join(root, name))[0] + ".png"  # .jpg
                try:
                    im = Image.open(os.path.join(root, name))
                    print("Generating png for %s" % name)
                    im.thumbnail(im.size)
                    im.save(outfile, "PNG", quality=100)  # JPEG
                    # Display the first image in training data
                except:
                    print("Unexpected error")



# Command:
# python3 crop_faces.py --shape-predictor dataset/shape_predictor_68_face_landmarks.dat --input dataset/output

# Reference:
# https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/