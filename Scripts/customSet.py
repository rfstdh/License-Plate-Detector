import os
import cv2
import numpy as np

#only need capital letters so i wont use directories over Sample036
digits = ['0','1','2','3','4','5','6','7','8','9']
letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
numeric_representation = [n for n in range(36)] # [0..35]

def customSet(dirStart,dirCounter=0):
    
    IMAGES = []
    LABELS = []
    
    while os.path.isdir(dirStart + str(dirCounter)):        
        imgCounter = 1
        imgPath = dirStart + str(dirCounter) + "/img" + str(imgCounter) +".jpg"
        
        while os.path.isfile(imgPath):       
            img = cv2.imread(imgPath)
            IMAGES.append(img)
            LABELS.append(numeric_representation[dirCounter])

            #update imagePath
            imgCounter+=1
            imgPath = dirStart + str(dirCounter) + "/img" + str(imgCounter) +".jpg"
        
        dirCounter+=1

    #transform lists into numpy arrays so it will be easier to process by NN model
    IMAGES = np.array(IMAGES)
    LABELS = np.array(LABELS)

    #check if everything is correct and we got expected number of samples
    print(IMAGES.shape)
    
    return IMAGES, LABELS