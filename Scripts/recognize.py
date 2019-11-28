import cv2
import numpy as np
from skimage.measure import label,regionprops
from detectPlate import plates
from detectCharacters import characters, sortKey
import keras
from skimage.color import gray2rgb


for i in range(len(characters)):
    characters[i] = gray2rgb(characters[i]) #28,28,1 => 28,28,3 
    characters[i] = np.array(characters[i])
    print(characters[i].shape)
    # cv2.imshow("c",characters[i])
    # cv2.waitKey(0)


model = keras.models.load_model('../Models/best_model.hdf5')

characters = np.array(characters) 
characters = np.reshape(characters,(7,28,28,3))
print(characters.shape)


#dictionary
d = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
predictions = model.predict_classes(characters)
print(predictions)


#letters can be in wrong order so i need to make it right
correct_plate = ""
d2 = {}
for i in range(len(sortKey)):
    d2[sortKey[i]] = predictions[i]

for i in sorted(d2):
    idx = d2[i]
    correct_plate += d[idx]

print("License plate: ", correct_plate)
