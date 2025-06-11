#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# http://www.gisdeveloper.co.kr/?paged=2&cat=130
# hough transform 
 
import numpy as np
import matplotlib.pyplot as plt
import cv2


# 1. 
im_bgr = cv2.imread('lane1.jpg')
im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)

# 2. Canny edge 
im_gray = cv2.GaussianBlur(im_gray, ksize = (13,13), sigmaX= 2.0)
Tlow = 50
Thigh = 150
im_edge = cv2.Canny(im_gray, Tlow, Thigh, apertureSize = 3)

 
# 3. Hough Transform using 
T = 100
im_hough1 = im_bgr.copy()
lines = cv2.HoughLines(im_edge, 1, np.pi/180, T) 
print('Hough Result:')
for i, line in enumerate(lines):
    
    rho,theta = line[0]     # rho and theta
    print(i, rho, theta*180./np.pi)
    #  note: 수업에서는  y cos(theta) + x sin(theta) = rho
    #        그런데, slide_theta = opencv_theta - 90 이므로
    #        x cos (opencv_theta)  + y cos (opencv_theta) = rho 가 됨.    
   
    a = np.cos(theta)         
    b = np.sin(theta)
    x0 = a*rho              # 수직교점 
    y0 = b*rho
    t = 1000
    x1 = int(x0 + t*(-b))    
    y1 = int(y0 + t*(a))
    x2 = int(x0 - t*(-b))
    y2 = int(y0 - t*(a))
    if  rho > 0 and theta < np.pi/2.:
        print('postive rho, theta < 90')
        cv2.line(im_hough1,(x1,y1),(x2,y2),(255,0,0),1) # blue
    elif rho > 0 and theta > np.pi/2.:
        print('postive rho, theta > 90')
        cv2.line(im_hough1,(x1,y1),(x2,y2),(0,0,255),1) # red
    elif rho < 0 and theta < np.pi/2.:
        print('negative rho, theta < 90')
        cv2.line(im_hough1,(x1,y1),(x2,y2),(0,255,0),1) # green
    elif rho < 0 and theta > np.pi/2.:
        print('negative rho, theta > 90 ')
        cv2.line(im_hough1,(x1,y1),(x2,y2),(0,0,0),1) # black
        
minLineLength = 100
maxLineGap = 20
lines = cv2.HoughLinesP(im_edge, 1, np.pi/180, T, minLineLength, maxLineGap)
if lines is None:
    print("no line detected")
    
im_hough2 = im_bgr.copy()
for i, line in enumerate(lines):
    print(i, line)
    x1,y1,x2,y2 = line[0]
    cv2.line(im_hough2,(x1,y1),(x2,y2),(255,0,0),3)

    
plt.subplot(1,3, 1)    
plt.imshow(im_edge, cmap='gray')
plt.axis('off')
plt.title('edge')

plt.subplot(1,3, 2)    
plt.imshow(im_hough1[:,:,::-1])
plt.axis('off')
plt.title('HoughLines')

plt.subplot(1,3, 3)    
plt.imshow(im_hough2[:,:,::-1])
plt.axis('off')
plt.title('HoughLinesP')


plt.show()
