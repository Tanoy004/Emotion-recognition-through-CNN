
# plot_dataset.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import Isomap
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA as sklearnPCA

# database_name = 'ck+'
# f = open('dataset/' + database_name + '.pckl', 'rb')
# data, my_model, labels, desc, training_time, image_w, image_h, numPoints, ran, ker, gam = pickle.load(f)
# f.close()


f = open('dataset/' + 'data_plots' + '.pckl', 'rb')

data, labels = pickle.load(f)
f.close()

data2 = np.array(data)
a,b = data2.shape

f = open('dataset/plot_dataset.data', 'w')

for i in range(0,a):
    f.write(str(labels[i]) + ',   ')
    for j in range(0,b-1):
        f.write(str(data2[i][j])+',   ')

    f.write(str(data2[i][j+1])+'\n\n')
f.close()

url = 'dataset/plot_dataset.data'

cols=[]
cols.append('Class')
for i in range(0,b):
    cols.append(str(i))

dat = pd.read_csv(url, names=cols)

y = dat['Class']     # Split off classifications
X = dat.ix[:, '0':]  # Split off features
X_norm = (X - X.min())/(X.max() - X.min())

# print(cols)
# print(X_norm)
# print(y)

# PCA plotting
# plot_method = sklearnPCA(n_components=2) #2-dimensional PCA

# LDA plotting
plot_method = LDA(n_components=2) #2-dimensional LDA

transformed = pd.DataFrame(plot_method.fit_transform(X_norm, y))


plt.scatter(transformed[y==0][0], transformed[y==0][1], label='0=neutral', c='black')
plt.scatter(transformed[y==1][0], transformed[y==1][1], label='1=anger', c='red')
plt.scatter(transformed[y==3][0], transformed[y==3][1], label='3=disgust', c='orange')
plt.scatter(transformed[y==4][0], transformed[y==4][1], label='4=fear', c='yellow')
plt.scatter(transformed[y==5][0], transformed[y==5][1], label='5=happy', c='green')
plt.scatter(transformed[y==6][0], transformed[y==6][1], label='6=sadness', c='blue')
plt.scatter(transformed[y==7][0], transformed[y==7][1], label='7=surprise', c='violet')


plt.title('Projected normalized features of all classes in 2D plot for CK+ Dataset')
# plt.xlabel('no_of_features = %i'%(numPoints+2))

# plt.ylabel('no_of_features = %i,  C = %.1f,  random_state = %i'%(numPoints+2, c,ran))
# plt.xlabel('classifier = %s,  kernel = %s'%(classifier, ker))


plt.legend()
plt.show()


###################################################

# transformed2 = pd.DataFrame(X, y).T.plot()  # T.plot(kind='bar')
# plt.title('Projected features of all classes in 2D plot')
# plt.show()

# iso = Isomap(n_components=2)
# transformed3 = iso.fit_transform(X,y)
# plt.scatter(transformed3[:, 0], transformed3[:, 1], lw=0.1,c=y, cmap=plt.cm.get_cmap('cubehelix', 7))
#
# plt.colorbar(ticks=range(7))
# plt.clim(-0.5, 5.5)
# plt.show()


# X, y = make_blobs(n_samples=200, centers=3, n_features=2, random_state=0)
# plt.scatter(X[:,0], X[:,1], c=y)
# plt.show()