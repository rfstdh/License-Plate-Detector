import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray

from makeImageset import prepareSet

#preprocess data
IMAGES1, LABELS1 = prepareSet('../DataSource/Data/Img/GoodImg/Bmp/Sample00','GoodImg','/img00')
IMAGES2, LABELS2 = prepareSet('../DataSource/Data/Img/BadImag/Bmp/Sample00','BadImag','/img00')

IMAGES = np.concatenate((IMAGES1,IMAGES2))
LABELS = np.concatenate((LABELS1,LABELS2))

#split the data into training and validation sets
X_train, X_test, Y_train, Y_test = train_test_split(IMAGES,LABELS,test_size=0.15)

# Convert labels to categorical encoding i.e 6 will be [0,0,0,0,0,1,0,0....0]
train_cat = keras.utils.to_categorical(Y_train, num_classes=36)
test_cat = keras.utils.to_categorical(Y_test, num_classes=36)

#making a model
model = Sequential()

model.add(Conv2D(32, (5, 5), padding='same', input_shape=(28,28,3)))
model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.01))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(196, (5, 5)))
model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.01))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(GlobalMaxPooling2D())
model.add(BatchNormalization())

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))

model.add(Dense(36))
model.add(BatchNormalization())
model.add(Activation('softmax'))

#callbacks
cp = ModelCheckpoint('exp_model.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')

#compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, train_cat, validation_data = (X_test,test_cat), callbacks = [cp, reduce_lr_loss], epochs=50,batch_size=256)
predictions = model.predict_classes(X_test)
score = model.evaluate(X_test, test_cat, batch_size=128)

#print results and accuracy
print(predictions)
good = 0
wrong = 0
for i in range(len(X_test)):
   if predictions[i] == Y_test[i]:
      good+=1
   else: wrong+=1

print("Good: {} Wrong: {} with accuracy of {}".format(good,wrong,good/len(X_test)))
