# 0603.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread('./building.jpg', cv2.IMREAD_GRAYSCALE)

#1 gradient dx, dy
gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize = 3)  # x order = 1 only 
gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize = 3)  # y 
print("gx:", gx.dtype, gx.shape)

#2. magnitude
# you can use this, but be carefull angle range is 0 to 360 degree
# mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
dstM   = cv2.magnitude(gx, gy)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dstM)
print('mag:', minVal, maxVal, minLoc, maxLoc)

# 3. angle 
angles = np.arctan(gy/gx)
print("nan?", np.isnan(angles))  # some are NAN becuase deivide-by-zero
np.nan_to_num(angles, copy= False, nan=0.0) # replace by zero
print("nan?", np.isnan(angles))  # some are NAN becuase deivide-by-zero
print(cv2.minMaxLoc(angles))
angles = angles/np.pi*180.0
print(cv2.minMaxLoc(angles))  # should be (-90, 90) degree
dstAngles  = np.zeros([angles.shape[0], angles.shape[1]], dtype = np.uint8)

# Quantization by conditional indexing (much faster than for-loop
dstAngles[ angles < -67.5 ] = 1
dstAngles[ (angles >= -67.5) & (angles < -22.5) ] = 2
dstAngles[ (angles >= -22.5) & (angles < 22.5)  ] = 3
dstAngles[ (angles >= 22.5)  & (angles < 67.5)  ] = 4
dstAngles[ angles >= 67.5 ] = 5

'''
for y in range(angles.shape[0]):
    #print('y:', y)
    for x in range(angles.shape[1]):
        #if y == 0:
        #    print('y:', y, 'x:', x, 'val:', angles[y,x])
        if np.isnan(angles[y,x]): # maybe 90 or - 90
            dstAngles[y,x] = 0
        elif angles[y,x] < -67.5:
            dstAngles[y,x] = 1
        elif angles[y,x] < - 22.5:
            dstAngles[y,x] = 2
        elif angles[y,x] < 22.5:
            dstAngles[y,x] = 3
        elif angles[y,x] < 67.5:
            dstAngles[y,x] = 4
        else:
            dstAngles[y,x] = 5
'''        

   
dstAngles[dstM < 30] = 0  # masking non edge

print(np.count_nonzero(dstAngles[dstAngles == 1]))
print(np.count_nonzero(dstAngles[dstAngles == 2]))
print(np.count_nonzero(dstAngles[dstAngles == 3]))
print(np.count_nonzero(dstAngles[dstAngles == 4]))
print(np.count_nonzero(dstAngles[dstAngles == 5]))

  
# for visualizing
dstX = cv2.sqrt(gx)
dstY = cv2.sqrt(gy)
dstX = cv2.normalize(dstX, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
dstY = cv2.normalize(dstY, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
dstM = cv2.normalize(dstM, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

plt.subplot(5,2,1), plt.imshow(src, cmap ='gray'), plt.title('src'), plt.axis('off')
plt.subplot(5,2,2), plt.imshow(dstX, cmap ='gray'), plt.title('dx'), plt.axis('off')   
plt.subplot(5,2,3), plt.imshow(dstY, cmap ='gray'), plt.title('dy'), plt.axis('off')
plt.subplot(5,2,4), plt.imshow(dstM, cmap ='gray'), plt.title('mag'), plt.axis('off')

t = dstAngles.copy()
t[dstAngles != 1] = 0
plt.subplot(5,2,5), plt.imshow(t, cmap ='gray'), plt.title('angle <-67 '), plt.axis('off')
t = dstAngles.copy()
t[dstAngles != 2] = 0
plt.subplot(5,2,6), plt.imshow(t, cmap ='gray'), plt.title('angle < -22.5'), plt.axis('off')
t = dstAngles.copy()
t[dstAngles != 3] = 0
plt.subplot(5,2,7), plt.imshow(t, cmap ='gray'), plt.title('angle < 22.5'), plt.axis('off')
t = dstAngles.copy()
t[dstAngles != 4] = 0
plt.subplot(5,2,8), plt.imshow(t, cmap ='gray'), plt.title('angle < 67.5'), plt.axis('off')
t = dstAngles.copy()
t[dstAngles != 5] = 0
plt.subplot(5,2,9), plt.imshow(t, cmap ='gray'), plt.title('angle > 67.5'), plt.axis('off')

plt.show()

