# 
# Feature Matching 데모 
#
# SIFT/SURF/ORB 특징검출과 기술자 생성 
# 

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


def detect_compute(img_gray, N = -1):

    start = time.time()  # 시작 시간 저장
    kps, descs = feature2d_extractor.detectAndCompute(img_gray, None)
    print("Time:", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

    print(f"num of keypoints: {len(kps)}")

    # 3. 갯수가 너무 많아서 강한것만 N 고름 
    if N == -1:
        top_N_kps = kps
        top_N_descs = descs
    else:    
        # Sort keypoints based on their response
        kps_sorted = sorted(kps, key=lambda x: x.response, reverse=True)
        top_N_kps = kps_sorted[:N]
        # Find corresponding descriptors and indices
        top_N_descs = [descs[kps.index(kp)] for kp in top_N_kps]

    if N != -1:
        for kp, desc in zip(top_N_kps, top_N_descs):
            print("Angle:", kp.angle)        # the major orientation in angles  
            print("Class ID:", kp.class_id)  # internal parameter
            print("Octave:", kp.octave)
            print("Point:", kp.pt)           # (x, y) in original image coordinate  
            print("Response:", kp.response)  # strength of response (DOG value)
            print("Size:", kp.size)          # scale in original image 
            print("Desc:", desc)
            
    return top_N_kps, top_N_descs

if __name__ == "__main__":

    
    img_file  = 'eiffel.jpg' #'sunflowers.jpg'
    img_color = cv2.imread(img_file)

    # rotation and scaling
    angle = 15 #45
    scale = 1.0 #0.75
    # Calculate rotation matrix
    height, width = img_color.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    img_rotated = cv2.warpAffine(img_color, rotation_matrix, (width, height))


    for method in ["SIFT", "ORB","SURF"]:  
        
        print("Using {}".format(method))
        
        if method == "SIFT": #  SIFT (not patent is expired 
            feature2d_extractor = cv2.xfeatures2d.SIFT_create()
        elif method == "SURF": #  SURF, license issue  
            feature2d_extractor = cv2.xfeatures2d.SURF_create()     
        elif method == "ORB":  
            feature2d_extractor = cv2.ORB_create()  
        else:
            print("Not support the Detector: ", method) 
            continue
  
        N = 100
        img_gray = cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY) # 그레이 영상 이용  
        kps, desc =  detect_compute(img_gray, N)
        img_annotated = cv2.drawKeypoints(cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR), kps, (255, 0, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 0: default, 4: rich 
     
        img_gray = cv2.cvtColor(img_rotated,cv2.COLOR_BGR2GRAY) # 그레이 영상 이용  
        kps_aug, desc_aug =  detect_compute(img_gray, N)
        img_annotated_aug = cv2.drawKeypoints(cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR), kps_aug, (255, 0, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 0: default, 4: rich 
        
        plt.subplot(1,2,1), plt.imshow(img_annotated)
        plt.subplot(1,2,2), plt.imshow(img_annotated_aug)
        plt.suptitle(method)
        plt.show()
