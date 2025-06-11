# 
# Feature Matching 데모 
#
# 1. SIFT/SURF 특징검출과 기술자 생성 
# 2. 특징 매칭(BFMatcher & FLANMatcher)
# 3. 매칭점을 사용한 기하변환 계산 (RANSAC)
# 

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


opt_detector = "ORB"   # option: "SIFT", "SURF", "ORB"
opt_matcher = "BF"     # option: "BF", "FLAN"


# 0. 입력 파일 로딩      
src1 = cv2.imread('./book1.jpg')
src2 = cv2.imread('./book2.jpg')
plt.subplot(1,2,1), plt.imshow(src1)
plt.subplot(1,2,2), plt.imshow(src2)
plt.show()


# 1. 피쳐 검출 
print("Using {}".format(opt_detector))

img1= cv2.cvtColor(src1,cv2.COLOR_BGR2GRAY) # 그레이 영상 이용  
img2= cv2.cvtColor(src2,cv2.COLOR_BGR2GRAY) 

start = time.time()  # 시작 시간 저장
if opt_detector == "SIFT": #  SIFT 사용 
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
elif opt_detector == "SURF": #  SURF 사용 
    surf = cv2.xfeatures2d.SURF_create()
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)
elif opt_detector == "ORB": #  ORB 사용 
    orb = cv2.ORB_create()  # not ORB() due to the version issue 
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
else:
    print("Not support the Detector: ", opt_detector) 

print("Detector and Descriptor :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

# 결과 확인 
kpimg1  = cv2.drawKeypoints(src1, kp1, (255, 0, 0), flags = 4)  # 0: default, 4: rich 
kpimg2  = cv2.drawKeypoints(src2, kp2, (255, 0, 0), flags = 0)  # 
plt.subplot(1,2,1), plt.imshow(kpimg1)
plt.subplot(1,2,2), plt.imshow(kpimg2)
plt.show()


######################################################################
# 2.  피쳐 매칭 
######################################################################
start = time.time()  # 시작 시간 저장
if opt_matcher == "BF":
    if opt_detector == "ORB":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING) # crossCheck=True 를 사용하면 k 수를 2개로 할 수가 없음?
    else:
        matcher = cv2.BFMatcher()  # cv2.NORM_L2
        
    matches = matcher.knnMatch(des1, des2, k = 2)  # 가장 가까운 두개 (ratio test를 위하여)
elif opt_matcher == "FLAN":
    matcher = cv2.FlannBasedMatcher_create() 
    matches = matcher.knnMatch(des1, des2, k = 2) 
else:
    print("Not support the Matcher: ", opt_matcher) 

print("Matching :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

# check the distances 
if True:
    matches2 = matcher.match(des1, des2)  # knnMatch return list of list (mutiple Matches) 
    #print(matches2[0])
    sortedMatches = sorted(matches2, key = lambda x: x.distance)
    print("distances:", sortedMatches[0].distance, sortedMatches[1].distance, sortedMatches[-1].distance)
 


# 매칭 결과 확인 
# queryIdx는 첫번째 입력 영상의 피쳐, trainIdx는 두번째 입력 영상의 피쳐
#
print(type(matches))
print(matches[0])
print('len(matches)=', len(matches))
for i, m in enumerate(matches[:3]):  # 너무 많으므로 일부만 출력 
    for j, n in enumerate(m):
        print('matches[{}][{}]=(queryIdx:{} => (trainImg:{}, trainIdx:{}), distance:{})'.format(
            i, j, n.queryIdx, n.imgIdx, n.trainIdx, n.distance))

dst = cv2.drawMatchesKnn(img1,kp1, img2, kp2, matches, None, flags=0) # 모든 매칭을 표시  
# 초벌 매칭 결과를 그림 

#cv2.imshow('AllRawMaches',  dst)
plt.imshow(dst)
plt.axis('off')
plt.title('all raw matches')
plt.show()

#####################################################################
# 3. 필터링. 좋은 매칭들 차이가 많이나는 경우만을 선택  
#####################################################################
nndrRatio = 0.7 #0.45  # SIFT 논문에서는 0.7 을 사용함.
##good_matches = []
##for f1, f2 in matches: # k = 2
##    if f1.distance < nndrRatio*f2.distance:
##        good_matches.append(f1)
good_matches = [f1 for f1, f2 in matches
                   if f1.distance < nndrRatio*f2.distance]
print('len(good_matches)=', len(good_matches))
if len(good_matches) < 5:
    print('sorry, too small good matches')
    exit()

# 좋은 매치들로만 다시 구성  
src1_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches])
src2_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches])

# 화면 출력 
dst2 = cv2.drawMatches(src1,kp1,src2,kp2, good_matches, None,flags = 0)

#cv2.imshow('GoodMaches',  dst2)
plt.imshow(dst2)
plt.axis('off')
plt.title('good matches')
plt.show()

##########################################################################
# 4. RANSAC을 이용 이미지간의 기하변환을 계산
##########################################################################

H, mask = cv2.findHomography(src1_pts, src2_pts, cv2.RANSAC, 0.1)#cv2.LMEDS
ransac_mask_matches = mask.ravel().tolist() # list(mask.flatten())
print("ransaced:", ransac_mask_matches)
print("H:", H)

# 기하변환을 이용하여 이미지의 아웃라인을 그림 (대략 잘 맞쳐 진것인지 확인)
h,w = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
pts2 = cv2.perspectiveTransform(pts, H)
src2 = cv2.polylines(src2,[np.int32(pts2)],True,(255,0, 0),2)
# @TODO: 입력이미지를 변환하여 타겟 위치로 와핑을 해 보라.         
        
        
draw_params = dict( #matchColor = (0,255,0),
                   singlePointColor = None,
                   matchesMask = ransac_mask_matches, flags = 2)                 
dst3 = cv2.drawMatches(src1,kp1,src2,kp2, good_matches, None,**draw_params)

#cv2.imshow('RANSACedMAtch',  dst3)
#cv2.waitKey(60*1000*2)
#cv2.destroyAllWindows()

plt.imshow(dst3)
plt.axis('off')
plt.title('RNSACed matches')
plt.show()


plt.subplot(3,1,1)
plt.imshow(dst)
plt.axis('off')
plt.title('all raw matches')

plt.subplot(3,1,2)
plt.imshow(dst2)
plt.axis('off')
plt.title('good matches')

plt.subplot(3,1,3)
plt.imshow(dst3)
plt.axis('off')
plt.title('RNSACed matches')
plt.show()
