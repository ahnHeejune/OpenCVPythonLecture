'''

https://gaussian37.github.io/vision-opencv-lkf/

'''

import cv2
import numpy as np


# measuement error 
R =  5.0 
 
width, height = 800, 800
measured=[]
predicted=[]
dr_frame = np.zeros((height,width,3), np.uint8)
mp = np.array((2,1), np.float32)
tp = np.zeros((2,1), np.float32)

def on_mouse(k,x,y,s,p):
    global mp,measured, R 
    
    noise = np.random.normal(loc=0.0, scale=R, size=2)
    x += noise[0]
    y += noise[1]
    
    mp = np.array([[np.float32(x)],[np.float32(y)]])
    measured.append((int(x),int(y)))

cv2.namedWindow("Sample")
cv2.imshow("Sample", dr_frame)  # for same window size
cv2.setMouseCallback("Sample",on_mouse);


def paint_canvas():
    global dr_frame,measured,predicted
    for i in range(len(measured)-1): 
        cv2.line(dr_frame,measured[i],measured[i+1],(0,100,0), thickness = 1)
    
    if len(measured) > 0:
        cv2.circle(dr_frame,measured[len(measured)-1],3, (0,100,0), thickness = -1)  # lastest measure
            
    for i in range(len(predicted)-1): 
        cv2.line(dr_frame,predicted[i],predicted[i+1],(0,0,200), thickness = 1)

def reset_canvas():
    global measured,predicted,dr_frame
    measured=[]
    predicted=[]
    dr_frame = np.zeros((height,width,3), np.uint8)


def kalman_init():
    
    global R
    
    ''' define dynamic model and statistical parameters for Kalman Filter '''
    
    kalman_fil = cv2.KalmanFilter(4,2)  # 4 state (x, y vx, vy), 2 meaasurement  x, y   
    
    # measured x, y => state x, y 
    kalman_fil.measurementMatrix = np.array([[1,0,0,0],
                                             [0,1,0,0]],np.float32)
    '''
    ### if when we measure x, y, x', y'
    kalman_fil.measurementMatrix = np.array([[1,0,0,0],
                                             [0,1,0,0],
                                             [0,0,1,0],
                                             [0,0,0,1]],np.float32)
    '''
    
    # system dyamics (x,y,vx,vy)= M* (x,y,vx,vy)                                                    
    kalman_fil.transitionMatrix = np.array([[1,0,1,0],   
                                            [0,1,0,1],
                                            [0,0,1.0,0],
                                            [0,0,0,1.0]],np.float32)
    # Q 
    kalman_fil.processNoiseCov = np.array([ [1,0,0,0],
                                            [0,1,0,0],
                                            [0,0,1,0],
                                            [0,0,0,1]],np.float32) * 0.03
    # R     
   
    kalman_fil.measurementNoiseCov = np.array([[1,0],
                                               [0,1]],np.float32) * R
  
    '''
    ## if when measure x, y, x', y' assuming all measure error are cross-independent C(*,*) = 0 
    kalman_fil.measurementNoiseCov = np.array([ [1,0, 0,0],
                                                [0,1, 0, 0],
                                                [0,0, 1, 0],
                                                [0,0, 0, 1]],np.float32) * R
    '''
                                               
    return kalman_fil
    

if __name__ == "__main__":

    kalman_fil = kalman_init()
    print(f"{kalman_fil.measurementNoiseCov}")

     
    while True:

        kalman_fil.correct(mp)     # measured (mouse position)  
        
        tp = kalman_fil.predict()  # predict by model 
        predicted.append((int(tp[0]),int(tp[1])))
        
        # examining the kalman filter 
        #print(f"errorCovPrev:{kalman_fil.errorCovPre[0][0]}, gain:{kalman_fil.gain}")
        
        # drawinging 
        paint_canvas()
        #cv2.imshow("Output",dr_frame)
        cv2.imshow("Sample",dr_frame)
        
        k = cv2.waitKey(30*5) &0xFF
        if k == 27: break
        if k == 32: reset_canvas()