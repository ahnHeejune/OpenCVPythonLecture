# demonstrate  SIFT/SURF Detector 
#
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_sift(gray):
   

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    im_sift1 =cv2.drawKeypoints(gray,kp, None)

    print(kp[:5])
    print(kp[0].class_id, kp[0].octave, kp[0].pt, kp[0].response, kp[0].size, kp[0].angle )


    #im = cv2.imread('box_in_scene.png')
    #gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #kp = sift.detect(gray,None)
    im_sift2=cv2.drawKeypoints(gray,kp, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    print(kp[:5])
    print(kp[0].class_id, kp[0].octave, kp[0].pt, kp[0].response, kp[0].size, kp[0].angle )

    #cv.imwrite('sift_keypoints.jpg',img)
    plt.subplot(1,2,1)
    plt.imshow(im_sift1)
    plt.axis('off')
    plt.title('im_sift1')

    plt.subplot(1,2,2)
    plt.imshow(im_sift2)
    plt.axis('off')
    plt.title('im_sift2')

    plt.show()

    
    
def show_surf(gray):
   

    surf = cv2.xfeatures2d.SURF_create()
    kp = surf.detect(gray,None)
    im_surf1 =cv2.drawKeypoints(gray,kp, None)

    print(kp[:5])
    print(kp[0].class_id, kp[0].octave, kp[0].pt, kp[0].response, kp[0].size, kp[0].angle )

    #im = cv2.imread('box_in_scene.png')
    #gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #kp = surf.detect(gray,None)
    im_surf2=cv2.drawKeypoints(gray,kp, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    print(kp[:5])
    print(kp[0].class_id, kp[0].octave, kp[0].pt, kp[0].response, kp[0].size, kp[0].angle )

    #cv.imwrite('sift_keypoints.jpg',img)
    plt.subplot(1,2,1)
    plt.imshow(im_surf1)
    plt.axis('off')
    plt.title('im_surf1')

    plt.subplot(1,2,2)
    plt.imshow(im_surf2)
    plt.axis('off')
    plt.title('im_surf2')

    plt.show()

    
#########################################################
# diagram test 
#########################################################
gray = np.zeros([400, 500], dtype = 'uint8')

scales  = [2, 4, 8, 16, 32] #[5, 10, 20, 30, 40]

for  i, r in enumerate([5, 10, 20, 30, 40]):
    cv2.circle(gray, (i*100 +10, 100), r, 255, -1)
    cv2.rectangle(gray, (i*100 -r + 10, 300-r), (i*100 +r + 10, 300+r), 255, -1)
    
show_sift(gray)   
show_surf(gray)   


#########################################################
# image test 
#########################################################


im = cv2.imread('box.png')
gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)    
show_sift(gray)
show_surf(gray)





