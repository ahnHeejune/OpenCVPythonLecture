# imimtating Lindberg experiment
#
import cv2
import numpy as np
import matplotlib.pyplot as plt

    
#########################################################
# diagram test 
#########################################################
gray = np.zeros([200, 300], dtype = 'uint8')

obj_scales  = [7, 11] # [2, 4, 6, 8, 10] 

centers = []
masks = []
for  i, r in enumerate(obj_scales):

    # draw blob with center and radius 
    center = (i*100 + 100, 100)
    cv2.circle(gray, center, r, 255, -1)
    centers.append(center)
    
    # make masks that cover each object area for later we can check the extremes 
    mask = np.zeros([200, 300], dtype = 'uint8')
    mask = cv2.circle(mask, center, r*3//2, 255, -1)
    masks.append(mask)
 
 
sigmas = np.arange(1.0,10.0,0.5)    

for s in sigmas:

    ksize = int(s*6)
    if ksize%2 == 0:
        ksize = ksize +1
        
    print(s)
    print(ksize)    
    
    im_gaussian = cv2.GaussianBlur(gray, (ksize,ksize), s)
    im = cv2.Laplacian(im_gaussian, cv2.CV_32F)
    im = s*s*im
    
    #im_bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) #.astype(np.uint8)
    
    mstr = ""
    for i, center in enumerate(centers):
        print("center:", center)
        print("response:", im[center[1], center[0]])
         
        mm = cv2.minMaxLoc(im, masks[i])
        print(mm)
        cv2.drawMarker(im, (mm[2][0],mm[2][1]), 255, markerSize = 5) 
    
        mstr += "%1d c_res=%.1f min=%.1f"%(i, im[center[1], center[0]], mm[0]) 

    '''
    plt.subplot(1,2,1)
    #mask = cv2.add(masks[0],masks[1])
    #plt.imshow(mask)
    plt.imshow(im_gaussian)
    plt.axis('off')
    
        
    plt.subplot(1,2,2)
    '''
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.title('sigma=' + str(s) + mstr )
    plt.show()









