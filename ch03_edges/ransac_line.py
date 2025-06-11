#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# http://www.gisdeveloper.co.kr/?paged=2&cat=130
# hough transform 
 
import numpy as np
import matplotlib.pyplot as plt
import cv2


# 1. empty images
height = 500
width  = 500
im_bgr = np.zeros([height, width, 3], dtype=np.uint8)

# 2. draw points on the lines 
pt1 = (0, 0)
pt2 = (width, height)


step = 25
num_inliers = 500//25
pts_line = np.zeros([500//step, 2], dtype= np.int32)

print(im_bgr.shape)

for x in range(0,500,step):
    im = cv2.drawMarker(im_bgr, (x,x), (255,255,255), cv2.MARKER_CROSS, markerSize=5,thickness=2)
    pts_line[x//step] = (x,x)

    
#im_bgr = cv2.line(im_bgr, pt1, pt2, color =(255,255,255), thickness=1)
plt.imshow(im_bgr[:,:,::-1])
plt.axis('off')
plt.show()

# 3. make a random points 
num_outliers = 20
noise = np.random.randint(0, 500, [num_outliers,2], np.int32)

for i in range(num_outliers):
    x, y = noise[i] 
    #im_bgr[y, x] = (0, 0, 255)
    im = cv2.drawMarker(im_bgr, (x,y), (255,0,0), cv2.MARKER_CROSS, markerSize=5,thickness=2)

    
plt.imshow(im_bgr[:,:,::-1])
plt.axis('off')
plt.show()
     
# 4. RANSAC 준비: 두점을 랜던하게 5개를 선택하여 선을 그려본다. 
pts_total =  np.concatenate((pts_line , noise), axis=0)
num_pts = num_outliers + num_inliers
for i in range(5):
    i1 = np.random.randint(0, num_pts, 1, np.int32)
    i2 = np.random.randint(0, num_pts, 1, np.int32)
    pt1 = (pts_total[i1, 0],pts_total[i1, 1]) 
    pt2 = (pts_total[i2, 0],pts_total[i2, 1]) 
    #print(tuple(pt1), tuple(pt2))
    print(i1, i2)
    cv2.line(im_bgr, pt1, pt2, color =(0,0,255), thickness=1)


plt.imshow(im_bgr[:,:,::-1])
plt.axis('off')
plt.show()
    
    
# 숙제에서 당신은 다음과 같은 구현을 하여야한다. 

# 2. RANSAC 구현 
#
# 1. 다음을 N 번을 반복한다. 
# 2. 랜덤하게 두점을 고른다.
# 3. 라인 파라메터를 계산한다. (theta rho 던, a b 던) 
# 4. 데이터셋안의 점들 중 위 선과 거리가 T 보다 작은 것의 갯수를 샌다. 
# 5. 그 갯수가 기존의 것보다 큰지 확인하다(conensus)
# 6. 만약 그런경우 최대치를 업데이트 한다.
