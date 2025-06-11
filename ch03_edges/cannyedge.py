#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Compare Canny with Sobel
import numpy as np
import matplotlib.pyplot as plt
import cv2
    
#im_path = "Lena.png"
im_path = "building.jpg"
    
im_src  = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)

#
# make sobel edge magnitude and binary image (0/255uint8)
#
def sobel_edge(im_gray, T):

    #1 dx, dy
    gx = cv2.Sobel(im_gray, cv2.CV_32F, 1, 0, ksize = 3)  # x order = 1 only 
    gy = cv2.Sobel(im_gray, cv2.CV_32F, 0, 1, ksize = 3)  # y 
    #print("gx:", gx.dtype, gx.shape)
    
    dstM   = cv2.magnitude(gx, gy) # need  CV_32F or CV_64F
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dstM)
    print('mag:', minVal, maxVal, minLoc, maxLoc)
    #dstM = cv2.normalize(dstM, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    _, im_bin = cv2.threshold(dstM, T, 255, cv2.THRESH_BINARY)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(im_bin)
    print('mag:', minVal, maxVal, minLoc, maxLoc)
    print("im_bin:", im_bin.dtype, im_bin.shape)
    return dstM, im_bin.astype(np.uint8)


# sobel edge for comparsion    
im_mag, im_sobel = sobel_edge(im_src, T = 127)
    
# sobel edge for comparison 
im_canny1 = cv2.Canny(im_src, threshold1= 127, threshold2 = 127)
im_canny2 = cv2.Canny(im_src, threshold1= 127, threshold2 = 61)

print("canny edge:", im_canny1.dtype, im_canny1.shape )
print("min,   max:", cv2.minMaxLoc(im_canny1))  # 0/255 (uint8)


plt.subplot(2,4,1), plt.imshow(im_mag, cmap="gray"), plt.title('src')
plt.subplot(2,4,2), plt.imshow(im_sobel, cmap="gray"), plt.title('sobel')
plt.subplot(2,4,3), plt.imshow(im_canny1, cmap="gray"), plt.title('canny')
plt.subplot(2,4,4), plt.imshow(im_canny2, cmap="gray"), plt.title('canny')
plt.subplot(2,4,5), plt.imshow(im_mag[300:400, 600:700], cmap="gray"), plt.title('mag')
plt.subplot(2,4,6), plt.imshow(im_sobel[300:400, 600:700], cmap="gray"), plt.title('sobel')
plt.subplot(2,4,7), plt.imshow(im_canny1[300:400, 600:700], cmap="gray"), plt.title('canny')
plt.subplot(2,4,8), plt.imshow(im_canny2[300:400, 600:700], cmap="gray"), plt.title('canny')

plt.show()
