
# cv_file_to_image_dataset.py

import numpy as np
import matplotlib.image as mpim
from matplotlib import cm


with open('dataset/fe/fer2013_modified.csv') as f:
    content = f.readlines()

lines = np.array(content)
num_classes = 7
width = 48
height = 48

tr0, tr1, tr2, tr3, tr4, tr5, tr6 = (0 for i in range(7))
v0, v1, v2, v3, v4, v5, v6 = (0 for i in range(7))
te0, te1, te2, te3, te4, te5, te6 = (0 for i in range(7))

# a0=0
# a1=0
# a2=0
# a3=0
# a4=0
# a5=0
# a6=0

num_of_instances = lines.size
# num_of_instances = 3
print("number of instances: ", num_of_instances)

for i in range(1, num_of_instances):
    try:
        emotion, img, usage = lines[i].split(",")

        val = img.split(" ")
        pixels = np.array(val, 'float32')

        result = np.fromstring(img, dtype=int, sep=" ").reshape((48, 48))

        if 'Training' in usage:
            name = str(i)+'_training_' + emotion +'.png'

            if '0' in emotion:
                mpim.imsave('dataset/fe/train_fer/0/' + name, np.uint8(result), cmap=cm.gray)
                tr0+=1
            if '1' in emotion:
                mpim.imsave('dataset/fe/train_fer/1/' + name, np.uint8(result), cmap=cm.gray)
                tr1+=1
            if '2' in emotion:
                mpim.imsave('dataset/fe/train_fer/2/' + name, np.uint8(result), cmap=cm.gray)
                tr2+=1
            if '3' in emotion:
                mpim.imsave('dataset/fe/train_fer/3/' + name, np.uint8(result), cmap=cm.gray)
                tr3+=1
            if '4' in emotion:
                mpim.imsave('dataset/fe/train_fer/4/' + name, np.uint8(result), cmap=cm.gray)
                tr4+=1
            if '5' in emotion:
                mpim.imsave('dataset/fe/train_fer/5/' + name, np.uint8(result), cmap=cm.gray)
                tr5+=1
            if '6' in emotion:
                mpim.imsave('dataset/fe/train_fer/6/' + name, np.uint8(result), cmap=cm.gray)
                tr6+=1

        # if 'PublicTest' or 'PrivateTest' in  usage:

        if 'PublicTest' in usage:
            name = str(i) + '_test_' + emotion + '.png'

            if '0' in emotion:
                mpim.imsave('dataset/fe/test_fer1/0/' + name, np.uint8(result), cmap=cm.gray)
                v0 += 1
            if '1' in emotion:
                mpim.imsave('dataset/fe/test_fer1/1/' + name, np.uint8(result), cmap=cm.gray)
                v1 += 1
            if '2' in emotion:
                mpim.imsave('dataset/fe/test_fer1/2/' + name, np.uint8(result), cmap=cm.gray)
                v2 += 1
            if '3' in emotion:
                mpim.imsave('dataset/fe/test_fer1/3/' + name, np.uint8(result), cmap=cm.gray)
                v3 += 1
            if '4' in emotion:
                mpim.imsave('dataset/fe/test_fer1/4/' + name, np.uint8(result), cmap=cm.gray)
                v4 += 1
            if '5' in emotion:
                mpim.imsave('dataset/fe/test_fer1/5/' + name, np.uint8(result), cmap=cm.gray)
                v5 += 1
            if '6' in emotion:
                mpim.imsave('dataset/fe/test_fer1/6/' + name, np.uint8(result), cmap=cm.gray)
                v6 += 1

        if 'PrivateTest' in usage:
            name = str(i) + '_test_' + emotion + '.png'

            if '0' in emotion:
                mpim.imsave('dataset/fe/test_fer2/0/' + name, np.uint8(result), cmap=cm.gray)
                te0 += 1
            if '1' in emotion:
                mpim.imsave('dataset/fe/test_fer2/1/' + name, np.uint8(result), cmap=cm.gray)
                te1 += 1
            if '2' in emotion:
                mpim.imsave('dataset/fe/test_fer2/2/' + name, np.uint8(result), cmap=cm.gray)
                te2 += 1
            if '3' in emotion:
                mpim.imsave('dataset/fe/test_fer2/3/' + name, np.uint8(result), cmap=cm.gray)
                te3 += 1
            if '4' in emotion:
                mpim.imsave('dataset/fe/test_fer2/4/' + name, np.uint8(result), cmap=cm.gray)
                te4 += 1
            if '5' in emotion:
                mpim.imsave('dataset/fe/test_fer2/5/' + name, np.uint8(result), cmap=cm.gray)
                te5 += 1
            if '6' in emotion:
                mpim.imsave('dataset/fe/test_fer2/6/' + name, np.uint8(result), cmap=cm.gray)
                te6 += 1

    except:

        print(" error occured ", end="")

print('image dataset in .cv format is stored as original image')

print("Number of images for per emotion label in training set: ")
print(tr0,' ',tr1,' ',tr2,' ',tr3,' ',tr4,' ',tr5,' ',tr6)

print("Number of images for per emotion label in PublicTest set: ")
print(v0,' ',v1,' ',v2,' ',v3,' ',v4,' ',v5,' ',v6)

print("Number of images for per emotion label in PrivateTest set: ")
print(te0,' ',te1,' ',te2,' ',te3,' ',te4,' ',te5,' ',te6)







