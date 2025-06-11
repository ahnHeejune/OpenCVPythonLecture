#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
#  Binarization and morphology 
#

import numpy as np
import cv2
import matplotlib.pyplot as plt

    
im_path = 'coins.png'
im_gray  = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)

# 1. otsu binarization 
###############################################

# histogram 
hist = cv2.calcHist([im_gray], channels =[0], mask = None, histSize=[256],ranges=[0.0,256.0]) 
print("sum:", np.sum(hist), " = res:", im_gray.shape[0]*im_gray.shape[1])

# binarization
thres, im_bin = cv2.threshold(im_gray, thresh = 0 , maxval = 255, type = cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(thres) # 126

# ostue could not detect one (range around 100 - 120) 
thres, im_bin2 = cv2.threshold(im_gray, thresh = thres - 30 , maxval = 255, type = cv2.THRESH_BINARY)

plt.subplot(2,2,1), plt.imshow(im_gray, cmap='gray'), plt.axis('off')
plt.subplot(2,2,2), plt.plot(hist), plt.grid()
plt.subplot(2,2,3), plt.imshow(im_bin, cmap='gray'), plt.title('ostu'), plt.axis('off')
plt.subplot(2,2,4), plt.imshow(im_bin2, cmap='gray'), plt.title('manual'), plt.axis('off')
plt.show()    

#
# Morphological Filter 
#img = cv.imread('j.png',0)
kernel = np.ones((5,5),np.uint8)
# or using API
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

erosion = cv2.erode(im_bin,kernel,iterations = 1)
dilation = cv2.dilate(im_bin,kernel,iterations = 1)
opening = cv2.morphologyEx(im_bin, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(im_bin, cv2.MORPH_CLOSE, kernel)

plt.subplot(2,2,1), plt.imshow(erosion, cmap='gray'), plt.title('erosion'), plt.axis('off')
plt.subplot(2,2,2), plt.imshow(dilation, cmap='gray'), plt.title('dilation'), plt.axis('off')
plt.subplot(2,2,3), plt.imshow(opening, cmap='gray'), plt.title('openning'), plt.axis('off')
plt.subplot(2,2,4), plt.imshow(closing, cmap='gray'), plt.title('closing'), plt.axis('off')
plt.show() 
