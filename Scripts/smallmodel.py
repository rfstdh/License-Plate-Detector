import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalMaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray

import numpy as np

import cv2
from customSet import customSet
from fontImages import prepareFontSet
from sklearn.model_selection import train_test_split
#data
# IMAGES, LABELS = customSet("../DataSource/Letters/Letter")
# IMAGESF, LABELSF = prepareFontSet('../DataSource/English/Fnt/Sample00','/img00')
from makeImageset import prepareSet
IMAGES1, LABELS1 = prepareSet('../DataSource/Data/Img/GoodImg/Bmp/Sample00','GoodImg','/img00')
IMAGES2, LABELS2 = prepareSet('../DataSource/Data/Img/BadImag/Bmp/Sample00','BadImag','/img00')
from resizeImages import resizeSet
resizeSet('../DataSource/Data/Img/GoodImg/Bmp/Sample00','GoodImg','/img00',(48,48))
resizeSet('../DataSource/Data/Img/BadImag/Bmp/Sample00','BadImag','/img00',(48,48))

print(IMAGES1.shape)
print(IMAGES2.shape)
IMAGES = np.concatenate((IMAGES1,IMAGES2))
LABELS = np.concatenate((LABELS1,LABELS2))


X_train, X_test, Y_train, Y_test = train_test_split(IMAGES,LABELS,test_size=0.15)

X_train = X_train / 255.0
X_test = X_test / 255.0

print(IMAGES.shape)
print(LABELS)
train_cat = keras.utils.to_categorical(Y_train, num_classes=36)
test_cat = keras.utils.to_categorical(Y_test, num_classes=36)

print(train_cat.shape)
# print(IMAGES[0])
#making a model
model = Sequential()

model.add(Conv2D(64, (5, 5), padding='same', input_shape=(48,48,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(42, (5, 5)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(GlobalMaxPooling2D())
model.add(BatchNormalization())

model.add(Dense(36))
model.add(BatchNormalization())
model.add(Activation('softmax'))


cp = ModelCheckpoint('small_model.hdf5', save_best_only=True, monitor='val_loss', mode='min')

model.summary()
#compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.save("small_model.hdf5")

model.fit(X_train, train_cat, validation_data = (X_test, test_cat), callbacks = [cp],  epochs=10,batch_size=128)
predictions = model.predict_classes(X_test)
print(predictions)