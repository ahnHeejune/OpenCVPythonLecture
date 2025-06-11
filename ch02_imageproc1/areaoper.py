# 
# filter operations demo
#


import cv2
import numpy as np
from matplotlib import pyplot as plt


def demo_filter():

    ###########################################################################
    # How to use filter2D
    ###########################################################################
    img = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    #averaging filter 
    ksize = 10
    kernel = np.ones((ksize,ksize),np.float32)/(ksize*ksize)
    img_box = cv2.filter2D(img,-1,kernel)
    print(kernel)

    #gaussian filter 
    ksize = 31
    sigma = 10
    #kernel = np.ones((ksize,ksize),np.float32)/(ksize*ksize)
    img_g1 = cv2.GaussianBlur(img,(ksize,ksize),sigma)
    
    kernel_1d	=	cv2.getGaussianKernel(ksize = ksize, sigma = sigma)
    kernel = np.outer(kernel_1d, kernel_1d.transpose())
    print(kernel)
    img_g2 = cv2.filter2D(img,-1,kernel)

    
    plt.subplot(2,2,1),plt.imshow(img, cmap='gray'),plt.title('Original'), plt.axis('off')
    plt.subplot(2,2,2),plt.imshow(img_box, cmap='gray'),plt.title('box u filter2D'), plt.axis('off')
    plt.subplot(2,2,3),plt.imshow(img_g1, cmap='gray'),plt.title('gauss u. blur'), plt.axis('off')
    plt.subplot(2,2,4),plt.imshow(img_g2, cmap='gray'),plt.title('gauss u. filter2D'), plt.axis('off')
   
    plt.show()
    

def demo_denoising():

    ###########################################################################
    # Denoising 
    #
    ########################################################################### 

    img = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # add salt noise 
    noise_level = 0.05
    img_noise = img.copy()

    num_noise = noise_level*img.shape[0]*img.shape[1]
    coords_x = [np.random.randint(0, img.shape[1] - 1, int(num_noise))] 
    coords_y = [np.random.randint(0, img.shape[0] - 1, int(num_noise))] 
    img_noise[coords_y, coords_x]  = 255 # white salt
    
    # pepper
    num_noise = noise_level*img.shape[0]*img.shape[1]
    coords_x = [np.random.randint(0, img.shape[1] - 1, int(num_noise))] 
    coords_y = [np.random.randint(0, img.shape[0] - 1, int(num_noise))] 
    img_noise[coords_y, coords_x]  = 0   # black pepper

    blur_gauss = cv2.GaussianBlur(img_noise,(5,5),2.0)
    blur_median = cv2.medianBlur(img_noise,5)
    #blur_bilateral = cv2.bilateralFilter(img,9,75,75)   

    plt.subplot(2,2,1),plt.imshow(img, cmap='gray'),plt.title('original'),plt.axis('off')
    plt.subplot(2,2,2),plt.imshow(img_noise,  cmap='gray'),plt.title('S&P noise'), plt.axis('off')
    plt.subplot(2,2,3),plt.imshow(blur_gauss,  cmap='gray'),plt.title('avg'),plt.axis('off')
    plt.subplot(2,2,4),plt.imshow(blur_median,  cmap='gray'),plt.title('median'),plt.axis('off')
    plt.show()
    
  
#demo_filter()  
demo_denoising()
    