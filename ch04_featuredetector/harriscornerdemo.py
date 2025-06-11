# demonstrate  Harris Corner
#
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1. 실험 이미지 읽기 
#filepath = 'triangle.png' # 'zener_star.jpg' 
filepath = 'zener_star.jpg' 

src = cv2.imread(filepath)

# 변형  
'''
rows, cols, channels = src.shape

# Rotation 변화, Invariant?
M = cv2.getRotationMatrix2D( (cols/2, rows/2),  45, 1.0 )  # center, rotation, scale 
src = cv2.warpAffine( src, M, (cols, rows), borderValue=(255, 255, 255))  # 변형하고 주변을 하얀색으로 채움 

# Scale 변화! invariant?
M = cv2.getRotationMatrix2D( (cols/2, rows/2),  45, 1.5 )  # center, rotation, scale 
src = cv2.warpAffine( src, M, (cols, rows), borderValue=(255, 255, 255))  # 변형하고 주변을 하얀색으로 채움 
'''

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

plt.subplot(2,2,1)
plt.imshow(src)
plt.axis('off')
plt.title('src')

#2 해리스 코너 적용 
bs = 5;  #5
res = cv2.cornerHarris(gray, blockSize=bs, ksize=3, k=0.04)
plt.subplot(2,2,2)
plt.imshow(res, cmap='gray')
plt.axis('off')
plt.title('response')


#3. Non-Max Supression 알고리즘 대신 간단하게 Cetroid 를 찾음.
#   TODO: 이진영상에서 연결영역을 찾고, 각 연결 영역 단위로 mask를 만들어서 
#         MinMaxLoc(arr, mask=NULL) 사용하여 위치를 찾으면 될것으로 보임. 
#         관심있는 학생들을 한번 시도해 보길. 
T_harris = 0.01 #0.02
ret, res_bin = cv2.threshold(res, T_harris, 255, cv2.THRESH_BINARY)
res_bin = cv2.dilate(res_bin, None) # 3x3 rect kernel
res_bin = np.uint8(res_bin)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(res_bin)
print('centroids.shape=', centroids.shape)
print('centroids=',centroids)
centroids = np.float32(centroids[1:, :])  # 배경영역제거 



plt.subplot(2,2,3)
plt.imshow(res_bin, cmap='gray')
plt.axis('off')
plt.title('binary')


#3. subpixel 검색 
term_crit=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,10, 0.001)
corners = cv2.cornerSubPix(gray, centroids, (5,5), (-1,-1), term_crit)
print('corners=',corners)

dst = src.copy()
for x, y in corners:    
    cv2.circle(dst, (x, y), 3, (0,0,255), 2)

plt.subplot(2,2,4)
plt.imshow(dst)
plt.axis('off')
plt.title('dst')

plt.show()