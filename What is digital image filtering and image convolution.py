# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from skimage import io, img_as_float
import cv2
import numpy as np

#read our image and convert it into floating point value
img = img_as_float(io.imread("BSE_25sigma_noisy.jpg"))

#define kernel
kernel = np.ones((5,5), np.float32) / 25
#we divide it by 25 because it is 5*5 matrix

#define gaussian kernel
gaussian_kernel = np.array([[1/16, 1/8, 1/16],
                            [1/8, 1/4, 1/8],
                            [1/16, 1/8, 1/16]])

#define convolutional filter
#conv_using_cv2 = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
conv_using_cv2 = cv2.filter2D(img, -1, gaussian_kernel, borderType=cv2.BORDER_CONSTANT)

#show original and convolututional image
cv2.imshow("Original", img)
#cv2.imshow("Cv2 Filter", conv_using_cv2)
cv2.imshow("gaussian filter",conv_using_cv2)

cv2.waitKey(0)
cv2.destroyAllWindows()


#########################################################################################

#another way of doing this using scipy 

from skimage import io, img_as_float
import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve

#read our image and convert in into floating point value
img = img_as_float(io.imread("BSE_25sigma_noisy.jpg", as_gray=True))

#define kernel
kernel = np.ones((5,5), np.float32) / 5

#define gaussian kernel
gaussian_kernel = np.array([[1/16, 1/8, 1/16],
                           [ 1/8, 1/4, 1/8],
                           [1/16, 1/8, 1/16]])


conv_using_cv2 = cv2.filter2D(img, -1, gaussian_kernel, borderType=cv2.BORDER_CONSTANT)

conv_using_scipy_signal = convolve2d(img, kernel, mode="same")

conv_using_scipy_ndimage = convolve(img, kernel, mode="constant", cval=0.0)


#Show the image
cv2.imshow("Original", img)
cv2.imshow("cv2 filter", conv_using_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()


