import cv2
from skimage.measure import label,regionprops

#load image
img = cv2.imread("../img/rej4.jpg",0) # load as gray
cv2.imshow("Gray image",img)
cv2.waitKey(0)

#img size
img_height, img_width = img.shape
print(img_height,img_width)


#make img binary
r, thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)


labelled = label(thresh_img)

#license plate approximate percentage size of an image
license_height_min = 0.2 * img_height
license_height_max = 0.9 * img_height
license_width_min = 0.6 * img_width 
license_width_max = 0.95 * img_width 

#array for possible licenses
plates = []


regions = regionprops(labelled)
for r in regions:
    if r.area > 13000:
        print("Found a region: ")   
        x_top,y_top,x_left,y_left = r.bbox
        h = x_left - x_top
        w = y_left - y_top
        print(r.bbox)
        print((h,w))
        print(r.area)
        if h > license_height_min and h < license_height_max and w > license_width_min and w < license_width_max:
            img = cv2.rectangle(img,(y_top,x_top),(y_left,x_left),(255,0,0),3)
            plate = img[x_top+3:x_left-3,y_top+3:y_left-3]
            plates.append(plate)

cv2.imshow("Gray image",img)
cv2.waitKey(0)
