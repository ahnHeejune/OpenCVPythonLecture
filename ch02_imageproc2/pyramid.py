#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


def laplacian():
    #   Down        Up
    # 
    #   0           4
    #   v           ^
    #   1           3
    #   v           ^ 
    #   2           2 
    #   v           ^  
    #   3           1 
    #   v           ^
    #   4 --------> 0
     
    
    im_path = 'tsukuba_l.png'
    im_src = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE) 
    print(im_src.shape)

    # down scale 
    imsGaussian = [im_src]  
    for s in range(1,5):
        im = cv2.pyrDown(imsGaussian[s-1])
        imsGaussian.append(im)
     
    ims2 = [ ]  
    for s in range(4,0,-1):
        im = cv2.pyrUp(imsGaussian[s])
        ims2.insert(0,im)
    
    imsLaplacian = []  
    for s in range(0,4):
        im = cv2.subtract(imsGaussian[s],ims2[s])
        imsLaplacian.append(im)
        #print(im.shape)
    imsLaplacian.append(imsGaussian[4])     
    
    '''    
    ims2 = [ims1[4]]  
    for s in range(0,4):
        im = cv2.pyrUp(ims2[s])
        ims2.append(im)   #ims2.insert(0,im)
    '''    
    print(len(imsGaussian))
    print(len(ims2))
    print(len(imsLaplacian))

    for s in range(0,5):
        plt.subplot(5, 3, s*3 + 1), plt.imshow(imsGaussian[s], cmap='gray')
        plt.title(str(imsGaussian[s].shape[0])), plt.axis('off')
        if s != 4:
            plt.subplot(5, 3, s*3 + 2), plt.imshow(ims2[s], cmap='gray') 
            plt.title(str(ims2[s].shape[0])), plt.axis('off')
        plt.subplot(5, 3, s*3 + 3), plt.imshow(imsLaplacian[s], cmap='gray')
        plt.title(str(imsLaplacian[s].shape[0])), plt.axis('off')

    plt.suptitle('Gaussian vs Laplacian Pyramid')    
    plt.show()    

    
def aliasing_antialiasing():
    
    #   Down-sampling with aliasing and anti        
    # 
    #   0          
    #   v           
    #   1          
    #   v            
    #   2            
    #   v            
    #   3            
    #   v           
    #   4  
    
    im_path = 'tsukuba_l.png'
    im_src = cv2.imread(im_path)     
    print(im_src.shape)

    # down scale 
    ims1 = [im_src]  
    for s in range(1,5):
        im = ims1[s-1][::2,::2,:]
        ims1.append(im)
   
    ims2 = [im_src]  
    for s in range(1,5):
        im = cv2.pyrDown(ims2[s-1])
        ims2.append(im)
  
    for s in range(0,5):
        plt.subplot(5, 2, s*2 + 1), plt.imshow(ims1[s]), plt.title(str(ims1[s].shape)), plt.axis('off')
        plt.subplot(5, 2, s*2 + 2), plt.imshow(ims2[s]), plt.title(str(ims2[s].shape)), plt.axis('off')

    plt.suptitle('aliasing vs anti-aliasing')    
    plt.show()    
    

    
#aliasing_antialiasing()
    
laplacian()
    