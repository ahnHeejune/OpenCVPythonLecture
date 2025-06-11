'''  

 Face Identification 

 using ATT dataset  
 
 You can apply to MNIST, Fashion MNIST,  CIFAR 100 etc  
 
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

WIDTH = 92
HEIGHT = 112

def load_face(filename='faces.csv', test_ratio=0.2):

    # read the face list 
    file = open(filename, 'r')
    lines = file.readlines()

    # 2. read all images
    N = len(lines)
    faces = np.empty((N, WIDTH*HEIGHT), dtype=np.uint8 )
    labels = np.empty(N, dtype = np.int32)
    for i, line in enumerate(lines):
        filename, label = line.strip().split(';')
        labels[i] = int(label)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        faces[i, :] = img.flatten()
  
    print(f"{N} images for labels {list(set(labels)).sort()}")
  
    # 3. separate into training and test/validation dataset 
    
    # shuffling and seperate train and test data
    indices = list(range(N))
    random.seed(1) # same random sequences, so the same result
    random.shuffle(indices)
    shuffle_faces = faces[indices]
    shuffle_labels = labels[indices]

    test_size = int(test_ratio*N)

    test_faces = shuffle_faces[:test_size]
    test_labels = shuffle_labels[:test_size]

    train_faces = shuffle_faces[test_size:]
    train_labels = shuffle_labels[test_size:]
 
    return train_faces, train_labels, test_faces, test_labels


if __name__ == "__main__":

    # 1. load dataset 
    train_faces, train_labels, test_faces, test_labels = load_face()
    print('train_faces.shape=',  train_faces.shape)
    print('train_labels.shape=', train_labels.shape)
    print('test_faces.shape=',   test_faces.shape)
    print('test_labels.shape=',  test_labels.shape)

    # 2. create Face recognizer     
    recognizer = cv2.face.EigenFaceRecognizer_create()
    ##recognizer = cv2.face.FisherFaceRecognizer_create()
    
    # 3. train with train dataset 
    print("training....")
    recognizer.train(train_faces.reshape(-1, HEIGHT, WIDTH), train_labels)

    # 3.2 train result: display eigen Face
    eigenFace = recognizer.getEigenVectors()
    eigenFace = eigenFace.T
    print('eigenFace.shape=',  eigenFace.shape)
    dst = np.zeros((8*HEIGHT, 10*WIDTH), dtype=np.uint8)
    ##for i in range(39): # FisherFaceRecognizer
    for i in range(80):
        x = i%10
        y = i//10
        x1 = x*WIDTH
        y1 = y*HEIGHT
        x2 = x1+WIDTH
        y2 = y1+HEIGHT  
        img = eigenFace[i].reshape(HEIGHT, WIDTH)
        dst[y1:y2, x1:x2] = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
    
    #cv2.imshow('eigenFace 80', dst)
    plt.imshow(dst)
    plt.title('eigenFace 80')
    plt.show()
    

    # 4. predict test_faces using recognizer
    print("predicting....")
    correct_count = 0
    for i, face in enumerate(test_faces): 
        predict_label, confidence = recognizer.predict(face)
        if test_labels[i] == predict_label:
            correct_count+= 1
        print(f'{test_labels[i] == predict_label}, gt={test_labels[i]}, pred:{predict_label},confidence={confidence:.2f}')
         
    accuracy = correct_count / float(len(test_faces))
    print(f'accuracy={accuracy:.4f}')
