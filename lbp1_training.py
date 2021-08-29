
# lbp1_training.py

# import the necessary packages
from lbp1_localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC, SVC, NuSVC
from imutils import paths
import numpy as np
import argparse, cv2, dlib, pickle, imutils
from time import time
from sklearn.model_selection import train_test_split,  KFold, cross_val_score, ShuffleSplit, GridSearchCV, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt


database_name = 'train_fer'
load_lbp_feature = False
load_landmarks = False
save_model = True
method = 'SVhlhC'  # SVC or LinearSVC
plot_training_curve = True

image_w = 48
image_h = 48
radius = 3  # 8
numPoints = 8*radius  # 24
split = 10
data_test = 0.2


classifier1, classifier2, c, ran, ker, gam, it = (SVC, LinearSVC, 175, None, 'rbf', 0.001, -1)

# my_model = classifier1(C=c, random_state=ran, kernel=ker, gamma=gam)
# my_model = classifier2(C=c, random_state=ran)

my_model = classifier1(random_state=0, kernel=ker)

grid_pram_svc = {'kernel': ['rbf', 'linear', 'poly'],
                 'C': [50, 100, 125, 150, 175, 200, 225, 250],
                 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01],
                 'random_state': [None, 0, 100]}

grid_pram_lin = {'C': [1, 25, 50, 75, 100, 125, 150, 175, 200],
                 'random_state': [None, 0, 100]}

shuf = ShuffleSplit(n_splits=split, test_size=data_test, random_state=0)
predictor = dlib.shape_predictor('dataset/shape_predictor_68_face_landmarks.dat')

def get_landmarks(image, rects):
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


start_time=time()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
                help="path to the training images")

args = vars(ap.parse_args())

if args == None:
    raise Exception("could not load image !")

# initialize the local binary patterns descriptor along with
# the data and label lists

desc = LocalBinaryPatterns(numPoints, radius)  # (24,8)

data = []
labels = []
landmarks = []
lbp_features = []
imagePath = paths.list_images(args["training"])
before_training=time()
print("Training is starting:\n")

cnt = 0

# loop over the training images
# saving data and labels. useful for large dataset with (8,1)-variable

featstart_time = time()

if load_lbp_feature == False:

    for imagePath in paths.list_images(args["training"]):

        cnt+=1
        print(cnt, "path = ", imagePath)

        image = cv2.imread(imagePath)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = imutils.resize(gray, width=image_w, height=image_h)

        face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
        face_landmarks = get_landmarks(gray, face_rects)
        landmarks.append(face_landmarks)

        hist = desc.describe(gray)

        labels.append(imagePath.split("/")[-2])
        lbp_features.append(hist)
    np.save('dataset/expression/' + database_name + '_' + str(radius) + '_' + str(numPoints) + '_lbp_features.npy', lbp_features)
    np.save('dataset/expression/' + database_name + '_' + str(radius) + '_' + str(numPoints) + '_landmarks.npy', landmarks)
    np.save('dataset/expression/' + database_name + '_' + str(radius) + '_' + str(numPoints) + '_labels.npy', labels)

else:

    lbp_features = np.load('dataset/expression/' + database_name + '_' + str(radius) + '_' + str(numPoints) + '_lbp_features.npy')
    labels = np.load('dataset/expression/' + database_name + '_' + str(radius) + '_' + str(numPoints) + '_labels.npy')

if load_landmarks:
    landmarks = np.load('dataset/expression/' + database_name + '_' + str(radius) + '_' + str(numPoints) + '_landmarks.npy')
    landmarks = np.array([x.flatten() for x in landmarks])
    data = np.concatenate((landmarks, lbp_features),  axis=1)
else:
    data = lbp_features


featend_time = time()
print(np.array(data).shape)

print("Choosing best parameters")

grid_svc = GridSearchCV(estimator = my_model, cv = shuf, param_grid = grid_pram_svc)
grid_lin = GridSearchCV(estimator = my_model, cv = split, param_grid = grid_pram_lin)
check = 0

if method =='SVC':
    grid_svc.fit(data, labels)
    # print(grid_svc.cv_results_)
    print(grid_svc.best_estimator_ )
    print(grid_svc.best_score_)
    print(grid_svc.best_params_)
    check = grid_svc.best_estimator_

    scores = cross_val_score(check, data, labels, cv=shuf)
    print(method)
    print("Mean Accuracy on Best parameter : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    test_labels = check.predict(data)
    print(classification_report(labels, test_labels))
    mat = confusion_matrix(labels, test_labels)
    print(mat)

if method == 'LinearSVC':
    grid_lin.fit(data, labels)
    # print(grid_lin.cv_results_)
    print(grid_lin.best_estimator_)
    print(grid_lin.best_score_)
    print(grid_lin.best_params_)
    check = grid_lin.best_estimator_

    scores = cross_val_score(check, data, labels, cv=shuf)
    print(method)
    print("Mean Accuracy on Best parameter : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    test_labels = check.predict(data)
    print(classification_report(labels, test_labels))
    mat = confusion_matrix(labels, test_labels)
    print(mat)


# split in train and test set for my_model

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle = True)

check2 = my_model.fit(X_train, y_train)
predictions = check2.predict(X_test)
print(np.array(X_test).shape, np.array(y_test).shape)
print('test_Score of my model:', check2.score(X_test, y_test))


training_scores = cross_val_score(my_model, data, labels, cv=shuf)

print(np.array(my_model.support_vectors_).shape)
# print(my_model.support_)

test_labels = my_model.predict(data)
print(classification_report(labels, test_labels))
mat = confusion_matrix(labels, test_labels)
print(mat)
print("Mean Accuracy on Training data : %0.2f (+/- %0.2f)" % (training_scores.mean(), training_scores.std() * 2))


# print trainig time and show data and labels

training_labels=labels
after_training=time()
training_time=after_training-before_training


print("total trained data = ", len(labels))
print("training time = ", training_time, " seconds\n")


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

if plot_training_curve:

    title = 'Learning Curves'
    plot_learning_curve(my_model, title, data, labels, cv=shuf)
    plt.show()


if save_model:

    # saving data in file for split training & testing to different python script
    f = open('dataset/' + database_name + '.pckl', 'wb')

    pickle.dump([data, my_model, labels, desc,
                 training_time, image_w, image_h, numPoints, c, ran, ker, gam], f)
    f.close()

#

# 0=neutral, 1=anger, 2=disgust, 3=fear, 4=happy, 5=sadness, 6=surprise

# for classifier same type of image should be in dedicated folder

# ck+ (trained using ck+ - in expression folder)


# Command:
# python3 lbp1_training.py --training dataset/expression/ck+
