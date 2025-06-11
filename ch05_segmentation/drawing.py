# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# OpenCV  마우스와 키입력을 받아서 처리하는 데모 
#

import cv2
import numpy as np


img = cv2.imread('messi5.jpg') # 입력 영상 
mask = np.ones([img.shape[0], img.shape[1]], dtype = 'uint8') * 128  # 패인트 작업 저장 

key_pressed = False
mouse_pressed = False
color =  0


# 마우스에 특정 이벤트가 밸생하면 호출됨 
def mouse_callback(event, x, y, flags, param):
    
    print('event:', event, 'x:', x, 'y:', y, 'flags:', flags)
    
    global mouse_pressed, key_pressed, color  # 함수 외부에 선언된 변수를 사용하기 위한 선언 
    
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        cv2.circle(img, (x, y), 5, (color, color, color), cv2.FILLED)
        cv2.circle(mask, (x, y), 5, color, cv2.FILLED)
 
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            cv2.circle(img, (x, y), 5, (color, color, color), cv2.FILLED)
            cv2.circle(mask, (x, y), 5, color, cv2.FILLED)   
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
   

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)
   
while True:
    cv2.imshow('image', img)
    
    k = cv2.waitKey(1)  # 이 안에서 기다리다가 마우스 이벤트가 발생하면 콜백을 호출하고, 키보드 입력은 반환이 됨.
    
    if k == ord('q'):   # 종료 명령
        cv2.imwrite('mask.png', mask)  
        break
    elif k == ord('0'): # 0은 배경 지정 
        color = 0 #(0,0,0)
    elif k == ord('1'): # 1은  전경 지정 
        color = 255 #(255,255,255)
        
        
cv2.destroyAllWindows()
  