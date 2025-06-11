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

# loading images 
image = cv2.imread('./mnist_png/training/%1d.png'%(0), cv2.IMREAD_GRAYSCALE)
training_images = image.copy()
train_images =  [image]
for i in range(1,10):
    image = cv2.imread('./mnist_png/training/%1d.png'%(i), cv2.IMREAD_GRAYSCALE)
    training_images = np.hstack((training_images, image))
    train_images.append(image)
    
image = cv2.imread('./mnist_png/testing/%1d.png'%(0), cv2.IMREAD_GRAYSCALE )
testing_images = image.copy()
test_images = [image]
for i in range(1, 10):
    image = cv2.imread('./mnist_png/testing/%1d.png'%(i), cv2.IMREAD_GRAYSCALE)
    testing_images = np.hstack((testing_images, image)) 
    test_images.append(image)
    
plt.subplot(2,1, 1), plt.imshow(training_images), plt.title('training'), plt.axis('off') 
plt.subplot(2,1, 2), plt.imshow(testing_images), plt.title('testing'), plt.axis('off') 

plt.suptitle("gray")
plt.show()


# binarization 
train_bin_images = []
for i in range(10):
    ret, thresh = cv2.threshold(train_images[i],150,255,  cv2.THRESH_OTSU )
    train_bin_images.append(thresh)

test_bin_images = []
for i in range(10):
    ret, thresh = cv2.threshold(test_images[i],150,255,  cv2.THRESH_OTSU )
    test_bin_images.append(thresh)
        
for i in range(10):
    plt.subplot(2,10, i+1), plt.imshow(train_bin_images[i]), plt.title('%d'%(i)), plt.axis('off') 
    plt.subplot(2,10, 10+ i+1), plt.imshow(test_bin_images[i]), plt.title('%d'%(i)), plt.axis('off') 

plt.suptitle("binary")
plt.show()


# contours  
train_contours = []
train_contour_images = [ ]
for i in range(10):
    contours, _ = cv2.findContours(train_bin_images[i], cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    train_contours.append(contours[0])
    img = np.zeros_like(test_bin_images[i])
    cv2.drawContours(img, [contours[0]], -1, 255, 1)
    train_contour_images.append(img)  

test_contours = []
test_contour_images = [ ]
for i in range(10):
    contours, _ = cv2.findContours(test_bin_images[i], cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    test_contours.append(contours[0])
    img = np.zeros_like(test_bin_images[i])
    cv2.drawContours(img, [contours[0]], -1, 255, 1)
    test_contour_images.append(img)    

for i in range(10):
    plt.subplot(2,10, i+1), plt.imshow(train_contour_images[i]), plt.title('%d'%(i)), plt.axis('off') 
    plt.subplot(2,10, 10+ i+1), plt.imshow(test_contour_images[i]), plt.title('%d'%(i)), plt.axis('off') 

plt.suptitle("contours")
plt.show()


hu_train = []
hu_test = []
for i in range(10):
    
    moments = cv2.moments(train_contours[i]) #, True)
    #moments = cv2.moments(train_bin_images[i]) #, True)
    hu_moment1 = cv2.HuMoments(moments) 
    
    # from https://learnopencv.com/shape-matching-using-hu-moments-c-python/
    for j in range(7):
        hu_moment1[j] = -1* math.copysign(1.0, hu_moment1[j]) * math.log10(abs(hu_moment1[j]))
     
    hu_train.append(hu_moment1)
    print(f"train: {hu_moment1.T}")
    
    moments = cv2.moments(test_contours[i]) #, True)
    #moments = cv2.moments(test_bin_images[i]) #, True)
    hu_moment2 = cv2.HuMoments(moments) 
    # from https://learnopencv.com/shape-matching-using-hu-moments-c-python/
    for j in range(7):
        hu_moment2[j] = -1* math.copysign(1.0, hu_moment2[j]) * math.log10(abs(hu_moment2[j]))
  
    hu_test.append(hu_moment2)
    print(f"test: {hu_moment2.T}")
    
   
       
        
    
print(">>> using humMoment")
diff =  np.zeros((10,10), np.float32)
for n in range(10):
    for m in range(10):
        diff[n,m] = np.sum(np.absolute(hu_train[n] - hu_test[m]))
        #print("%d %d %.8f"%(n, m, np.sum(np.absolute(diff))))
        
print(diff)
matched = np.argmin(diff, axis = 1)    
print(matched)
   
    
print(">>> using matchShape")
# check the similairity 
# Note: cv2.matchShapes(contours, contours2, method, parameter)
diff =  np.zeros((10,10), np.float32)
for n in range(10):
    for m in range(10):
        diff[n,m] = dist = cv2.matchShapes(train_contours[n], test_contours[m], cv2.CONTOURS_MATCH_I3, 0)
        
print(diff)
matched = np.argmin(diff, axis = 1)    
print(matched)
    
    

