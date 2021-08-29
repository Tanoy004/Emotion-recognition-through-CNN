
# resize_image_in_required_format.py

# import the necessary packages
from imutils import paths
from PIL import Image
import argparse
import imutils
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	            help="path to input image")

args = vars(ap.parse_args())

if args == None:
    raise Exception("could not load image !")


print("Image processing is starting.")
# loop over the input images
for inputPath in paths.list_images(args["input"]):
    # load the image, convert it to grayscale, and describe it
    image = cv2.imread(inputPath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # write the output image to disk
    path = '/media/ankur/Administrator/Administration/My_Codes/Python/ml/dataset/output'

    resized = imutils.resize(gray, width=256, height=256)

    cv2.imwrite(os.path.join(path, inputPath.split("/")[-1]), resized)

    # display the output images
    cv2.imshow("Resized", resized)

    cv2.waitKey(1)

print("Image resizing is completed. Format changing is starting")

path = '/media/ankur/Administrator/Administration/My_Codes/Python/ml/dataset/output'
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
# python3 resize_image_in_required_format.py --input dataset/expression/2


# Reference:
# https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/