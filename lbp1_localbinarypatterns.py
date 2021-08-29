
# lbp1_localbinarypatterns.py

# import the necessary packages
from skimage import feature
import numpy as np

from scipy.stats import itemfreq
import matplotlib.pyplot as plt
import cv2


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-15):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns

        lbp = feature.local_binary_pattern(image, self.numPoints,self.radius, method="uniform")  # method="nri_uniform")


        n_bins = int(lbp.max() + 1)
        # =  self.numPoints + 2)  # when rotationally invariant features are required for method="uniform"
        # = (self.numPoints*(self.numPoints - 1)) + 3  # when LBP features do not encode rotation information for method="nri_uniform")


        bins = np.arange(0,n_bins+1)
        ran = (0, n_bins)
        (hist, _) = np.histogram(lbp.ravel(), bins, ran)

        hist = np.array(hist)
        pad = False
        padding = len(hist)
        for j in range(0, self.numPoints+2 - padding):
            pad = True
            print('padded')
            # print(hist)
            hist = np.append(hist, 0)
            # print(hist)

        # normalize the histogram
        hist = hist.astype(np.float)
        hist /= (hist.sum() + eps)


        info_print  = False

        if info_print:

            print("n_bins = ", n_bins)
            print("bins = ", bins)
            print("ran = ", ran)
            print("hist = ", hist)

            plt.hist(lbp.ravel(), bins, ran)
            plt.title('Histogram of LBP image')
            plt.show()

            cv2.imshow("LBP image", lbp)

            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == 27: break  # ESC key to exit
            cv2.destroyAllWindows()

            # Calculate the histogram
            x = itemfreq(lbp.ravel())
            # Normalize the histogram
            hist2 = x[:, 1] / sum(x[:, 1])

            print(x)
            print(hist2)
            print('image\n',image)

        #

        # return the histogram of Local Binary Patterns
        return hist



# Reference:

# https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/