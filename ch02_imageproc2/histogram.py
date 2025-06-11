#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# normal_curve.py

import numpy as np
import matplotlib.pyplot as plt
import cv2
    
im_path = 'sky_castle.jpg'
im_gray  = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)

# 1. histogram using opencv  
###############################################
hist = cv2.calcHist([im_gray], channels =[0], mask = None, histSize=[256],ranges=[0.0,256.0]) 
print("sum:", np.sum(hist), " = res:", im_gray.shape[0]*im_gray.shape[1])
plt.subplot(1,2,1)
plt.imshow(im_gray, cmap='gray')
plt.subplot(1,2,2)
plt.plot(hist)
plt.grid()
plt.show()    


# 2. 2D histogram using opencv  
#    (H => 1st index => y in image ,S => 2nd index in x in image)   
###############################################
im_bgr  = cv2.imread(im_path)
im_hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([im_hsv], channels =[0,1], mask = None, histSize=[18, 25],ranges=[0.,181., 0., 256.]) 
print("sum:", np.sum(hist), " = res:", im_hsv.shape[0]*im_hsv.shape[1])

#plt.subplot(1,2,1)
plt.imshow(hist, cmap='gray')
plt.xlabel('Saturation')
plt.ylabel('Hue')
#plt.grid()
plt.show()    
    
    