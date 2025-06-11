#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

imgfilePath = '../kame/data/Lena.png'
img = cv2.imread(imgfilePath)
# Check if image was successfully read.
assert img is not None

print('read {}'.format(imgfilePath))
print('shape:', img.shape)
print('dtype:', img.dtype)

img = cv2.imread(imgfilePath, cv2.IMREAD_GRAYSCALE)
assert img is not None
print('read {} as grayscale'.format(imgfilePath))
print('shape:', img.shape)
print('dtype:', img.dtype)
