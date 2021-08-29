
# lbp1_testing.py

# import the necessary packages
from imutils import paths
import argparse, cv2, pickle, dlib
import imutils
from time import time
import numpy as np
from sklearn.metrics import accuracy_score


database_name = 'ck+'
use_landmarks  = False

f = open('dataset/' + database_name + '.pckl', 'rb')
data, my_model, labels, desc, training_time, image_w, image_h, numPoints, ran, ker, gam = pickle.load(f)
f.close()

predictor = dlib.shape_predictor('dataset/shape_predictor_68_face_landmarks.dat')

def get_landmarks(image, rects):
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

def evaluate(model, X, Y):
    predicted_Y = model.predict(X)
    accuracy = accuracy_score(Y, predicted_Y)
    return accuracy


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--testing", required=True,
                help="path to the tesitng images")
args = vars(ap.parse_args())

if args == None:
    raise Exception("could not load image !")


print("Testing is staring:\n")

true = 0
false = 0
labels.clear()
prediction = 0
landmarks = []
lbp_features = []
data = []
labels = []

# loop over the testing images
for imagePath in paths.list_images(args["testing"]):
    # load the image, convert it to grayscale, describe it,
    # and classify it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width=image_w, height=image_h)

    hist = desc.describe(gray)

    if use_landmarks:

        labels.append(imagePath.split("/")[-2])
        lbp_features.append(hist)

        face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
        face_landmarks = get_landmarks(gray, face_rects)
        landmarks.append(face_landmarks)

    else:

        hist = hist.reshape(1, -1)  # for not reshaping here
        prediction = my_model.predict(hist)[0]  # DeprecationWarning is showed for this line without reshape

        # append the labels of testing folder of given dataset in cleared label array
        labels.append(imagePath.split("/")[-2])
        print("actually = ", labels[-1])
        print("predicted = ", prediction)
        print("path = ", imagePath)

        if labels[-1]==prediction:
            true=true+1
        else:
            false=false+1


        # display the image and the prediction
        cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.imshow("Sent Image", gray)
        cv2.waitKey(1)

#

if use_landmarks:
    data = np.concatenate((landmarks, lbp_features), axis=1)
    test_accuracy = evaluate(my_model, data,  labels)
    print("Test accuracy = {0:.1f}".format(test_accuracy*100))


end_time=time()
testing_labels = labels
#print("\ntesting labels = \n",testing_labels)
print("total tested data = ", true+false)
print("correctly predicted = ", true)
print("wrongly predicted = ", false)
print("training time = ", training_time, 'seconds\n')
print("testing accuracy", (true/(true+false))*100, "%\n")



#

# 0=neutral, 1=anger, 2=disgust, 3=fear, 4=happy, 5=sadness, 6=surprise

# for classifier same type of image should be in dedicated folder

# Command:

# ck+ (trained using ck+ - in expression folder)
# python3 lbp1_testing.py --testing dataset/expression/ck+

# jaffe (trained using jaffe - in expression folder)
# python3 lbp1_testing.py --testing dataset/expression/jaffe

# ck+jaffe (trained using ck+jaffe - in expression folder)
# python3 lbp1_testing.py --testing dataset/expression/ck+jaffe

# object
# python3 lbp1_testing.py --testing dataset/object/test  or  python3 lbp1_testing.py --testing dataset/object/train


