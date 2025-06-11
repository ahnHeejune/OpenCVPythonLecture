import cv2
import numpy as np
import matplotlib.pyplot as plt


# 1. load images
left_img = cv2.imread('left.png')
right_img = cv2.imread('right.png')

# 2.1 BM

stereo_bm = cv2.StereoBM_create(16) #32)
dispmap_bm = stereo_bm.compute(cv2.cvtColor(left_img,cv2.COLOR_BGR2GRAY), cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY))

# 2.2 SGBM
blockSize = 1
stereo_sgbm = cv2.StereoSGBM_create(0,32, blockSize=blockSize)
#print(dir(stereo_sgbm))
dispmap_sgbm1 = stereo_sgbm.compute(left_img, right_img)

blockSize = 4
stereo_sgbm.setBlockSize(blockSize)
dispmap_sgbm4 = stereo_sgbm.compute(left_img, right_img)
blockSize = 8
stereo_sgbm.setBlockSize(blockSize)
dispmap_sgbm8 = stereo_sgbm.compute(left_img, right_img)
blockSize = 16
stereo_sgbm.setBlockSize(blockSize)
dispmap_sgbm16 = stereo_sgbm.compute(left_img, right_img)
blockSize = 32
stereo_sgbm.setBlockSize(blockSize)
dispmap_sgbm32 = stereo_sgbm.compute(left_img, right_img)

# 3. display 
plt.figure(figsize=(12,10))
plt.subplot(2,4,1)
plt.title('left')
plt.imshow(left_img[:,:,::-1])
plt.subplot(2,4,2)
plt.title('right')
plt.imshow(right_img[:,:,::-1])
plt.subplot(2,4,3)
plt.title('bm')
plt.imshow(dispmap_bm, cmap='gray')
plt.subplot(2,4,4)
plt.title('sgbm-1')
plt.imshow(dispmap_sgbm1, cmap='gray')
plt.subplot(2,4,5)
plt.title('sgbm-4')
plt.imshow(dispmap_sgbm4, cmap='gray')
plt.subplot(2,4,6)
plt.title('sgbm-8')
plt.imshow(dispmap_sgbm8, cmap='gray')
plt.subplot(2,4,7)
plt.title('sgbm-16')
plt.imshow(dispmap_sgbm16, cmap='gray')
plt.subplot(2,4,8)
plt.title('sgbm-32')
plt.imshow(dispmap_sgbm32, cmap='gray')
plt.show()

