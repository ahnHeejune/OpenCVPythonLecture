''' 

 Laplacian of Gausian 
 
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2

src  = cv2.imread('building.jpg', cv2.IMREAD_GRAYSCALE)

# LOG kernel generation manually.
def logFilter(sigma = 3.0):
    
    # kernel size
    ksize = int(sigma*2*3.) # 97.5%
    if ksize % 2 == 0:
        ksize = ksize+1
        
    k_half = ksize//2
    
    print('sigma=', sigma)
    LoG = np.zeros((ksize, ksize), dtype=np.float32)
    for y in range(-k_half, k_half+1):
        for x in range(-k_half, k_half+1):
            g = -(x*x+y*y)/(2.0*sigma**2.0)
            LoG[y+k_half, x+k_half] = -(1.0+g)*np.exp(g)/(np.pi*sigma**4.0)
            
    return LoG

def demo_3dmesh(Z):
    from mpl_toolkits.mplot3d import Axes3D
    X = np.arange(0, kernel_log_5.shape[1] )
    Y = np.arange(0, kernel_log_5.shape[0])
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.plot_surface(X = X, Y = Y, Z = Z)
    ax.plot_wireframe(X = X, Y = Y, Z = Z)
    plt.show()
    
    
# using Gaussian and Laplacian
im_gaussian_1 = cv2.GaussianBlur(src, ksize=(7, 7), sigmaX=1.0)
im_lap1 = cv2.Laplacian(im_gaussian_1, cv2.CV_16S,3)

im_gaussian_5 = cv2.GaussianBlur(src, ksize=(31, 31), sigmaX=5.0)
im_lap5 = cv2.Laplacian(im_gaussian_5, cv2.CV_16S,3)

#2 using LOG filter
kernel_log_1 = logFilter(sigma = 1.0) #7, 15, 31, 51
print(kernel_log_1)
kernel_log_5 = logFilter(sigma = 5.0) #7, 15, 31, 51
#print(kernel_log_5)
print(kernel_log_5.shape)
demo_3dmesh(kernel_log_5)


im_log1 = cv2.filter2D(src, cv2.CV_16S, kernel_log_1)
im_log5 = cv2.filter2D(src, cv2.CV_16S, kernel_log_5)

plt.subplot(2,2,1), plt.imshow(im_log1, cmap='gray'), plt.title('LOG std=1')
plt.subplot(2,2,2), plt.imshow(im_lap1, cmap='gray'), plt.title('Laplcian of Gaussian std=1')
plt.subplot(2,2,3), plt.imshow(im_log5, cmap='gray'), plt.title('LOG std=5')
plt.subplot(2,2,4), plt.imshow(im_lap5, cmap='gray'), plt.title('Laplcian of Gaussian std=5')

plt.show()
