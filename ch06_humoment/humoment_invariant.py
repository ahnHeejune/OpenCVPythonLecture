# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# hu-moment and digit matching 
#

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
 
np.set_printoptions(precision=2) 


def compute_hu_moments(image):

    # binarization 
    ret, thresh = cv2.threshold(image,150,255,  cv2.THRESH_OTSU )
    # contours  
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    moments = cv2.moments(contours[0]) #, True)
    hu_moment = cv2.HuMoments(moments) 
    
    
    # from https://learnopencv.com/shape-matching-using-hu-moments-c-python/
    for i in range(0,7):
        hu_moment[i] = -1* math.copysign(1.0, hu_moment[i]) * math.log10(abs(hu_moment[i]))
    
    return hu_moment



file = "./mnist_png/training/%1d.png"%(0)

# loading images 
img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

# 1. original 
hu = compute_hu_moments(img)
print(hu.T)

rows, cols = img.shape

# 2. rotation 45 degree 
M = cv2.getRotationMatrix2D( (rows/2, cols/2),  45, 1.0 )  # center, rotation, scale 
img_rot = cv2.warpAffine( img, M, (rows, cols))
hu = compute_hu_moments(img_rot)
print(hu.T)


# 3. translation  
M = cv2.getRotationMatrix2D( (rows/2, cols/2),  0, 1.0 )  # center, rotation, scale  
M[0,2] = 3  
M[1,2] = 3  
img_trans = cv2.warpAffine( img, M, (rows, cols))
hu = compute_hu_moments(img_trans)
print(hu.T)

# 4. scaling 
M = cv2.getRotationMatrix2D( (rows/2, cols/2),  0, 0.5 )  # center, rotation, scale  
img_scale = cv2.warpAffine( img, M, (rows, cols))
hu = compute_hu_moments(img_scale)
print(hu.T)

plt.subplot(1,4,1), plt.imshow(img, cmap='gray'), plt.title('img')
plt.subplot(1,4,2), plt.imshow(img_rot, cmap='gray'), plt.title('rot')
plt.subplot(1,4,3), plt.imshow(img_trans, cmap='gray'), plt.title('trans')
plt.subplot(1,4,4), plt.imshow(img_scale, cmap='gray'), plt.title('scale')
plt.show()