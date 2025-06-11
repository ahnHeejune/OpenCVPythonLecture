#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as  plt

imageFile = 'Lena.png'

# 1. BGR
imgBGR = cv2.imread(imageFile)
if imgBGR is None:
    print("cannot find " , imageFile)
    exit()
plt.axis('off')
plt.imshow(imgBGR)
plt.title('BGR: mismatch of channels')
plt.show()

# 2. RGB
imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
plt.imshow(imgRGB)
plt.title('RGB by cvtColor')
plt.show()

# 2-2. RGB
plt.imshow(imgBGR[:,:,::-1]) # reverse the channel index
plt.title('RGB by indexing trick')
plt.show()

# 3. Split and Merge 
b, g, r= cv2.split(imgBGR)
imgRGB2 = cv2.merge((r,g,b))
imgBGR2 = cv2.merge((b,g,r))

plt.subplot(2,3,1)
plt.imshow(imgBGR)
plt.title('original')
plt.subplot(2,3,2)
plt.imshow(r, cmap = 'gray')
plt.title('r')
plt.subplot(2,3,3)
plt.imshow(g, cmap = 'gray')
plt.title('g')
plt.subplot(2,3,4)
plt.imshow(b, cmap = 'gray')
plt.title('b')
plt.subplot(2,3,5)
plt.imshow(imgRGB2)
plt.title('RGB:split-merge')
plt.subplot(2,3,6)
plt.imshow(imgBGR2)
plt.title('BGR:split-merge')
plt.show()
