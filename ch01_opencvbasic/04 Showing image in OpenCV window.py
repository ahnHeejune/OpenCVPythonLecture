#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import argparse
import cv2
import numpy as np

imgfilepath = 'lena.png'
orig = cv2.imread(imgfilepath)
assert orig is not None
orig_size = orig.shape[0:2]

cv2.imshow("Original image", orig)
k = cv2.waitKey(5000)

# not used ?
modified = orig.copy()  #######

#modified = orig
#white = np.ones_like(orig)
modified[:,:,0] = 0
cv2.imshow("modified", modified)
k = cv2.waitKey(10000)
cv2.destroyWindow("modified")

cv2.namedWindow("original", cv2.WINDOW_NORMAL)

while True:
    k = cv2.waitKey(0)
    print(f"key:{k}")
    if k == ord('k'):
       cv2.imshow("original", orig)   
    elif k == ord('q'):
        break
    else:
       pass
   
cv2.destroyAllWindows()

