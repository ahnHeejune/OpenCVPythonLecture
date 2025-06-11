#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# basic drawing 
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1. making a black image
im = np.zeros([500, 500, 3], dtype= np.uint8)

#im[100,100] = (0, 0, 255) # bgr 

# 2. draw some points
pos = (100, 300) # (x, y)
im = cv2.drawMarker(im, pos, (255,0,0), cv2.MARKER_CROSS, markerSize=15,thickness=2)
pos = (200,200)
im = cv2.drawMarker(im,  pos, (0,255,0), markerType = cv2.MARKER_STAR, markerSize=10,thickness=2)
pos = (300,300)
im = cv2.drawMarker(im, pos,  (0, 0, 255), markerType = cv2.MARKER_DIAMOND, markerSize=20,thickness=2)
pos = (400,400)
im = cv2.drawMarker(im, pos,  (255, 255, 255), markerType = cv2.MARKER_TRIANGLE_UP, markerSize=30,thickness=2)
plt.imshow(im[:,:,::-1])
plt.axis('off')
plt.show()
    
# 3. draws lines 
pt1 = (50, 50)
pt2 = (450, 450)
im = cv2.line(im, pt1, pt2, color =(255,255,255), thickness=2)
plt.imshow(im[:,:,::-1])
plt.axis('off')
plt.show()

# 4. draw circles.
center = (100, 200)
im = cv2.circle(im, center, 150, color =(255,255,255), thickness=2)
plt.imshow(im[:,:,::-1])
plt.axis('off')
plt.show()

# 5. draw rectangle
pt1 = (50, 50)
pt2 = (450, 450)
im = cv2.rectangle(im, pt1, pt2, color = (255,0,0), thickness=1)
plt.imshow(im[:,:,::-1])
plt.axis('off')
plt.show()


plt.show()
