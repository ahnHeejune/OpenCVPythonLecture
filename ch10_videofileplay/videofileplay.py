import cv2

# 동영상 파일에서 입력
cap = cv2.VideoCapture('viplanedeparture.avi') # 

if not cap.isOpened :
    print("cannot open file") 
    exit()

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height =  cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps =  cap.get(cv2.CAP_PROP_FPS)
print(f"{int(width)}x{int(height)}@{fps}")

while True:
    retval, frame = cap.read() # 프레임 캡처 
    if not retval:
        break       
    cv2.imshow("video", frame)
    key = cv2.waitKey(round(1000/fps))
    if key == ord('q'):
        break
    
if cap.isOpened():
    cap.release()
