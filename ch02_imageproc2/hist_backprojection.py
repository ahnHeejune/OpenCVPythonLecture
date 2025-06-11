import numpy as np
import matplotlib.pyplot as plt
import cv2
   
   
   
def demo_hist_backprojection():

    # 1. make a model histogram for hand's H distribution 
    img_model = cv2.imread("handmodel.jpg")
    if img_model is None:
        print('Could not open or find the image:', args.input)
        exit(0)
    img_model_hsv = cv2.cvtColor(img_model, cv2.COLOR_BGR2HSV) 
    # calculating Hue histogram
    hist_model = cv2.calcHist(images = [img_model_hsv], channels = [0], mask = None, histSize =[180], ranges = [0, 180] )
    
    # 2. apply the histogram model to a test images 
    img_target = cv2.imread("manyhands.jpg")
    img_target_hsv = cv2.cvtColor(img_target,cv2.COLOR_BGR2HSV) 
    img_project = cv2.calcBackProject([img_target_hsv], [0], hist_model,[0,180], scale = 1)

    plt.subplot(2,2,1), plt.imshow(img_model_hsv[:,:,0], cmap = "gray"), plt.axis('off'), plt.title('model')
    plt.subplot(2,2,2), plt.imshow(img_target[:,:,::-1]), plt.axis('off'), plt.title('target')
    plt.subplot(2,2,3), plt.imshow(img_target_hsv[:,:,0], cmap = "gray"), plt.axis('off'), plt.title('target hsv')
    plt.subplot(2,2,4), plt.imshow(img_project, cmap = "gray"), plt.axis('off'), plt.title('bp')
    plt.show()

if __name__ == "__main__":

    demo_hist_backprojection()