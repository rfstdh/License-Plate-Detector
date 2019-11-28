import os
import cv2

def resizeSet(dirStart,folder,imgMiddle,resize_size,dirCounter=1):
    while os.path.isdir(dirStart + str(dirCounter)):
        imgCounter = 1
        imgStart = "-0000"
        imgPath = dirStart + str(dirCounter) + imgMiddle + str(dirCounter) + imgStart + str(imgCounter) +".png"
        while os.path.isfile(imgPath):
            
            img = cv2.imread(imgPath)
            resized = cv2.resize(img,resize_size)
            cv2.imwrite(imgPath,resized)

            if imgCounter == 9:
                imgStart = "-000"
            if imgCounter == 99:
                imgStart = "-00"

            #update imagePath
            imgCounter+=1
            imgPath = dirStart + str(dirCounter) + imgMiddle + str(dirCounter) + imgStart + str(imgCounter) +".png"
        
        if dirCounter == 9:
            dirStart = f'../DataSource/Data/Img/{folder}/Bmp/Sample0'
            imgMiddle = "/img0"
        
        dirCounter+=1