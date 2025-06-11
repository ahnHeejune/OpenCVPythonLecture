#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import argparse
import cv2
import numpy as np

inimgfile = '../kame/data/Lena.png'
outpngfile = 'Lena_compressed.png'
outjpgfile = 'Lena_compressed.jpg'

img = cv2.imread(inimgfile)

# 2. PNG: save image with lower compression - bigger file size but faster decoding
cv2.imwrite(outpngfile, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# check that image saved and loaded again image is the same as original one
saved_img = cv2.imread(outpngfile)
tmp = (saved_img == img)
print(tmp.shape[0]*tmp.shape[1]*tmp.shape[2])
print(np.sum(tmp))
print(tmp.all())
assert saved_img.all() == img.all()

# 3. JPG save image with lower quality - smaller file size
cv2.imwrite(outjpgfile, img, [cv2.IMWRITE_JPEG_QUALITY, 0])
saved_img = cv2.imread(outjpgfile)
tmp = (saved_img == img)
print(tmp.shape[0]*tmp.shape[1]*tmp.shape[2])
print(np.sum(tmp))
print(tmp.all())
#assert saved_img.all() == img.all()