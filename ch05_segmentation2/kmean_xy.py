#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# basic drawing 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv2
 
im_bgr = cv2.imread('sunflowers.png')

Z = im_bgr.reshape((-1,3))  # Kmean 는 영상전용이 아니므로 
# 숙제로 => 위치 정보를 넣도록 함. 
 
# convert to np.float32
Z = np.float32(Z)
 
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3  # K-means
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
 
# Now convert back into uint8, and make original image
#  현재 예에서는 별로의미가 없음.
center = np.uint8(center)
res = center[label.flatten()]

# 이미지로 변환 
im_seg_k3 = res.reshape((im_bgr.shape))

##############
K = 7  # K-means
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
#  현재 예에서는 별로의미가 없음.
center = np.uint8(center)
res = center[label.flatten()] 
# 이미지로 변환 
im_seg_k7 = res.reshape((im_bgr.shape))
 
plt.subplot(3,1,1), plt.imshow(im_bgr[:,:,::-1]), plt.axis('off'), plt.title('original')
plt.subplot(3,1,2),plt.imshow(im_seg_k3[:,:,::-1]), plt.axis('off'), plt.title('k=7')
plt.subplot(3,1,3), plt.imshow(im_seg_k7[:,:,::-1]), plt.axis('off'), plt.title('k=15')
plt.show()

