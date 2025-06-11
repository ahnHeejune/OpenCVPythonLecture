#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# 1. 이진화 및 코폴로지컬 필터  
# 2. 연결역영 검출 
# 3. 컨투어 검출 및 그리기 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# basic drawing 
import numpy as np
import matplotlib.pyplot as plt
import cv2


def floodFillPostProcess(src, diff=(2,2,2)):
    img = src.copy()
    rows, cols = img.shape[:2]
    mask   = np.zeros(shape=(rows+2, cols+2), dtype=np.uint8)
    num_segments = 0
    for y in range(rows):
        for x in range(cols):
            if mask[y+1, x+1] == 0:
                r = np.random.randint(256)
                g = np.random.randint(256)
                b = np.random.randint(256)
                cv2.floodFill(img,mask,(x,y),(b,g,r),diff,diff)
                num_segments +=1
                
                
    return img, num_segments
 
im_bgr = cv2.imread('water_coins.jpg') #'coins.png') #'Thiemo.jpg') #'street.jpg')

 
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#K = 3  # K-means
#ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

#img_seg1	=	cv2.pyrMeanShiftFiltering(	im_bgr, sp = 20, sr = 10, maxLevel = 2, termcrit = criteria)
#print('done')
img_seg1	=	cv2.pyrMeanShiftFiltering(	im_bgr, sp = 5*8, sr = 10*4, maxLevel = 4) 
img_seg2, nseg    =  floodFillPostProcess(img_seg1) 
print('num:', nseg)
 
plt.subplot(3,1,1), plt.imshow(im_bgr[:,:,::-1]), plt.axis('off'), plt.title('original')
plt.subplot(3,1,2),plt.imshow(img_seg1[:,:,::-1]), plt.axis('off'), plt.title('meanshiftFiter')
plt.subplot(3,1,3), plt.imshow(img_seg2[:,:,::-1]), plt.axis('off'), plt.title('segmened')
plt.show()
