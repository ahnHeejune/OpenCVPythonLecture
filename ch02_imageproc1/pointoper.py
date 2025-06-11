# 
# pixel operations demo
#
#
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

###########################################################
# 1. Pixel Operation: numpy vs opencv 
##########################################################

def demo_pixel_operation():

    img1 = cv2.imread('perlgirl.jpg', cv2.IMREAD_GRAYSCALE) # rgb to gray 

    # 1. direct pixel operation with numpy : care!

    delta = 100
    img_brighter = img1 + delta
    img_darker   = img1 - delta
    img_flip     = 255 - img1

    plt.subplot(2,2,1), plt.imshow(img1, cmap='gray'), plt.title('original')
    plt.subplot(2,2,2), plt.imshow(img_brighter, cmap='gray'), plt.title('brighter')
    plt.subplot(2,2,3), plt.imshow(img_darker, cmap='gray'), plt.title('darker')
    plt.subplot(2,2,4), plt.imshow(img_flip, cmap='gray'), plt.title('flip')

    plt.suptitle('pixel operation w. Numpy: OV or UF') 
    plt.show()


    # do it by OpenCV api
    #img1 = cv2.imread('perlgirl.jpg', cv2.IMREAD_GRAYSCALE) # rgb to gray 

    img_brighter = cv2.add(img1, delta)
    img_darker   = cv2.subtract(img1, delta)
    img_flip     = cv2.subtract(255, img1)

    plt.subplot(2,2,1), plt.imshow(img1, cmap='gray'), plt.title('original')
    plt.subplot(2,2,2), plt.imshow(img_brighter, cmap='gray'), plt.title('brighter')
    plt.subplot(2,2,3), plt.imshow(img_darker, cmap='gray'), plt.title('darker')
    plt.subplot(2,2,4), plt.imshow(img_flip, cmap='gray'), plt.title('flip')
  
    plt.suptitle('pixel operation with opencv') 
    plt.show()

    img_perlgirl = cv2.imread('perlgirl.jpg') # rgb to gray 
    plt.imshow(img_perlgirl[:,:,::-1])  
    plt.suptitle('enjoy!') 
    plt.show()



#######################################################
# 2. Gamma Correction
#######################################################
def demo_gamma_correction():

    img1 = cv2.imread('harvest.jpg', cv2.IMREAD_GRAYSCALE) # rgb to gray 

    img1 = img1.astype(np.float32)/255.0

    gamma = 0.5
    img_0_5 = np.power(img1, gamma)
    img_0_5 = cv2.multiply(img_0_5, 255, cv2.CV_8UC3)
    gamma = 1.0
    img_1_0 = np.power(img1, gamma)
    img_1_0 = cv2.multiply(img_1_0, 255, cv2.CV_8UC3)
    gamma = 2.0
    img_2_0 = np.power(img1, gamma)
    img_2_0 = cv2.multiply(img_2_0, 255, cv2.CV_8UC3)


    plt.subplot(2,2,1)
    plt.imshow(img1, cmap='gray')  # 
    plt.title('original')

    plt.subplot(2,2,2)
    plt.imshow(img_0_5, cmap='gray')  # 
    plt.title('0.5')

    plt.subplot(2,2,3)
    plt.imshow(img_1_0, cmap='gray')  # 
    plt.title('1.0')

    plt.subplot(2,2,4)
    plt.imshow(img_2_0, cmap='gray')  # 
    plt.title('2.0')

    plt.suptitle('Gamma correction') 
    plt.show()



#######################################################
# 3. Blending
#######################################################
def demo_alpha_blending():
    plt.ion() # making the plt non-blocking mode

    img_day = cv2.imread('palaceday.jpg') 
    img_night = cv2.imread('palacenight.jpg') 

    for n in range(20):

        #dst = src1*alpha + src2*beta + gamma;
        alpha = 1. - n/20.
        img_blended = cv2.addWeighted(img_day, alpha, img_night, 1. - alpha, 0)
        plt.imshow(img_blended)
        plt.title("alpha : %.2f"%(alpha))
        #plt.show()
        plt.pause(1)

plt.show()

    
    
#demo_pixel_operation()    

demo_gamma_correction()
    
#demo_alpha_blending()    
    