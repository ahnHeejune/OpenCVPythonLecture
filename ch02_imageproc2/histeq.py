import numpy as np
import matplotlib.pyplot as plt
import cv2
 
def demo_histeq_explain():

    img = cv2.imread('tsukuba_l.png', cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    #print(img.shape)
     
    hist, bins = np.histogram(img.flatten(),256,[0,256]) # 2D to 1D to use np historam  
     
    cdf = hist.cumsum()
    #cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_normalized = cdf / cdf.max()


    '''
    # for-loop version 
    img_eq = np.zeros_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img_eq[y,x] = round(255.0*cdf_normalized[img[y,x]]) 
    '''
    # vector processing version for speed-up 
    img_eq = 255.0*cdf_normalized[img.flatten()]
    img_eq = img_eq.reshape(img.shape).astype(np.uint8)
   
    plt.subplot(1,2,1)
    plt.plot(cdf_normalized*float(hist.max()), color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
   
    plt.subplot(2,2,2), plt.imshow(img, cmap="gray"), plt.axis('off'), plt.title('input')
    plt.subplot(2,2,4), plt.imshow(img_eq, cmap="gray"), plt.axis('off'), plt.title('equalized')
    
    plt.show()
    
def demo_global_histeq():

    img = cv2.imread('tsukuba_l.png', cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    img_eq = cv2.equalizeHist(img)
    
    plt.subplot(1,2,1), plt.imshow(img, cmap="gray"), plt.axis('off'), plt.title('input')
    plt.subplot(1,2,2), plt.imshow(img_eq, cmap="gray"), plt.axis('off'), plt.title('equalized')
    plt.show()
    
def demo_adaptive_histeq():
          
    img = cv2.imread('tsukuba_l.png',0)
    assert img is not None, "file could not be read, check with os.path.exists()"
   
    img_eq = cv2.equalizeHist(img)
  
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_cl1 = clahe.apply(img)
 
    plt.subplot(1,3,1), plt.imshow(img, cmap="gray"), plt.axis('off'), plt.title('input')
    plt.subplot(1,3,2), plt.imshow(img_eq, cmap="gray"), plt.axis('off'), plt.title('equalized')
    plt.subplot(1,3,3), plt.imshow(img_cl1, cmap="gray"), plt.axis('off'), plt.title('adaptive equalized')
    plt.show()
    
if __name__ == "__main__":

    demo_histeq_explain()

    demo_global_histeq()
    
    demo_adaptive_histeq()