import cv2
from skimage.measure import label,regionprops
from detectPlate import plates

#load plate
license_plate = plates[0]

cv2.imshow("Plate",license_plate)
cv2.waitKey(0)


#plate size
plate_height, plate_width = license_plate.shape
print(plate_height,plate_width)


r, thresh_img = cv2.threshold(license_plate,157,255,cv2.THRESH_BINARY_INV)
cv2.imshow("Thresh",thresh_img)
cv2.waitKey(0)

labelled = label(thresh_img)

#character approximate percentage size of a plate
character_height_min = 0.6 * plate_height
character_height_max = 0.9 * plate_height
character_width_min = 0.05 * plate_width
character_width_max = 0.15 * plate_width

sortKey = [] #it will be needed in main script in order to order letters
characters = []

regions = regionprops(labelled)
for r in regions:
    if r.area > 130:
        x_top,y_top,x_left,y_left = r.bbox
        h = x_left - x_top
        w = y_left - y_top
        print(r.bbox)
        print((h,w))
        print(r.area)
        if h > character_height_min and h < character_height_max and w > character_width_min and w < character_width_max:
            license_plate = cv2.rectangle(license_plate,(y_top,x_top),(y_left,x_left),(255,0,0),3)
            c = license_plate[x_top+3:x_left-3,y_top+3:y_left-3]
            resized = cv2.resize(c,(28,28))                        
            characters.append(resized)
            sortKey.append(y_top)

cv2.imshow("Plate",license_plate)
cv2.waitKey(0)
