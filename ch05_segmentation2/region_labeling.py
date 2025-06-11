#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# 1. 이진화 및 모폴로지컬 필터  
# 2. 연결역영 검출 
# 3. 컨투어 검출 및 그리기 

import numpy as np
import matplotlib.pyplot as plt
import cv2
 
im_path = 'coins.png'
im_bgr  = cv2.imread(im_path)
im_gray  = cv2.imread(im_path, cv2.COLOR_BGR2GRAY)

# 1. otsu binarization 
###############################################
# binarization
thres, im_bin = cv2.threshold(im_gray, thresh = 0 , maxval = 255, type = cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#thres, im_bin = cv2.threshold(im_gray, thresh = 100 , maxval = 255, type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
print(thres) # 126


#
# Morphological Filter 
kernel = np.ones((3,3),np.uint8)
# or using API
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

im_dilation = cv2.dilate(im_bin,kernel,iterations = 1)
kernel = np.ones((3,3),np.uint8)
im_closing = cv2.morphologyEx(im_dilation, cv2.MORPH_CLOSE, kernel)

################################################
# 연결 영역 검출 
##############################################
ret, im_labels = cv2.connectedComponents(im_closing)  # ret 는 레이블(연결된 영역수)
print('label 수:', ret - 1)  # 0은 백그라운드라서 제외 함
scaling = 255//ret          # 차이가 잘보이기 위하여 

plt.subplot(2,2,1), plt.imshow(im_bin, cmap='gray'), plt.title('binary'), plt.axis('off')
plt.subplot(2,2,2), plt.imshow(im_dilation, cmap='gray'), plt.title('dilation'), plt.axis('off')
plt.subplot(2,2,3), plt.imshow(im_closing, cmap='gray'), plt.title('closing'), plt.axis('off')
plt.subplot(2,2,4), plt.imshow(im_labels*scaling, cmap='gray'), plt.title('label'), plt.axis('off')
plt.show() 

# contours 
######################################################
im2, contours, hierarchy = cv2.findContours(im_closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
plt.subplot(1,2,1), plt.imshow(im_labels, cmap='gray'), plt.title('label'), plt.axis('off')

#np_contours = np.array(contours)
#print(np_contours.dtype) # object, so not very usefull for multi-D cases
print("contours dim:", len(contours))
print("contours[0]:", contours[0])
print("contours[0]:", contours[0].shape)
print("hierarchy:", hierarchy)
cv2.drawContours(im_bgr, contours, -1, (0,255,0), 1)
plt.subplot(1,2,2), plt.imshow(im_bgr[:,:,::-1]), plt.title('closing'), plt.axis('off')
plt.show() 

'''
cnt = contours[4]
cv.drawContours(img, [cnt], 0, (0,255,0), 3)
'''