# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# http://www.gisdeveloper.co.kr/?p=6747
#

import numpy as np
import cv2
from matplotlib import pyplot as plt
 
img = cv2.imread('messi5.jpg')
mask = np.zeros(img.shape[:2],np.uint8)
 
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
 
# Step 1
# 1. GC_INIT_WITH_RECT   
# 첫번째는 이미지에서 전경이 포함되는 영역을 사각형으로 대략적으로 지정합니다. 단, 이때 지정한 사각형 영역 안에는 전경이 모두 포함되어 있어야 합니다. 

rect = (50,50,450,290)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
img1 = img*np.where((mask==2)|(mask==0),0,1).astype('uint8')[:,:, np.newaxis]
mask1 = mask.copy()  # for debugging 
 
# Step 2
# 2. GC_INIT_WITH_MASK
# 그리고 두번째는 첫번째에서 얻어진 전경 이미지의 내용중 포함되어진 배경 부분은 어디인지, 
# 누락된 전경 부분은 어디인지를 마킹하면 이를 이용해 다시 전경 이미지가 새롭게 추출됩니다.

newmask = cv2.imread('gc_mask.png',0)
mask[newmask == 0] = 0
mask[newmask == 255] = 1
mask_before = mask.copy()  # for debugging
cv2.grabCut(img,mask,None,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)
 
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img2 = img*mask2[:,:,np.newaxis]

plt.subplot(3,2,1), plt.imshow(mask1, cmap='gray'), plt.title('by rect'), plt.axis('off') 
plt.subplot(3,2,2), plt.imshow(img1[:,:,::-1]), plt.title('by rec'), plt.axis('off') 
plt.subplot(3,2,3), plt.imshow(newmask, cmap='gray'), plt.title('handwork'), plt.axis('off') 
plt.subplot(3,2,4), plt.imshow(mask_before, cmap='gray'), plt.title('for mask'), plt.axis('off') 
plt.subplot(3,2,5), plt.imshow(mask, cmap='gray'), plt.title('after mask'), plt.axis('off') 
plt.subplot(3,2,6), plt.imshow(img2[:,:,::-1]), plt.title('final'),  plt.axis('off') 

plt.show()