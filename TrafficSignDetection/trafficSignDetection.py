import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.layers import *
from keras.models import Sequential


path = "myData"
labelFile = "labels.csv"
count = 0
images = []
classNo = []
classList = os.listdir(path)
# print(classList, end = '')
noOfClasses = len(classList)
for x in range(0, noOfClasses):
    listOfPictures = os.listdir(path+'/'+str(count))
    for y in listOfPictures:
        currentImg = cv2.imread(path+'/'+str(count)+'/'+y, 0)
        images.append(currentImg)
        classNo.append(count)
    print(count, end = ' ')
    count += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)
images = images/255


X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size = 0.2)
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)


data = pd.read_csv(labelFile, index_col=False)
# print(data.shape)

# plt.imshow(X_train[0])
# plt.show()

# print(y_train[0])
# print(data['Name'][y_train[0]])

X_train = X_train.reshape(-1,32,32,1) #The input to a Conv2D layer must be four-dimensional.
X_test = X_test.reshape(-1,32,32,1)
# print(X_train.shape,X_test.shape)
y_train=np_utils.to_categorical(y_train) #convert array of labeled data(from 0 to nb_classes-1) to one-hot vector.
y_test=np_utils.to_categorical(y_test)

# print(y_train[0])

# print(y_train.shape,y_test.shape,y_validation.shape)

model=Sequential()
model.add(Conv2D(64,activation='relu',kernel_size=3,input_shape=(32,32,1),padding='same')) #input_shape does not include batch_size
model.add(Conv2D(64,activation='relu',kernel_size=3,padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16,activation='relu',kernel_size=3,padding='same'))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(43))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) #Sparse Categorical Crossentropy may also be used.

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=5,batch_size=48,verbose=2)

image_index = 402
plt.imshow(X_test[image_index].reshape(32,32),cmap='gray')
pred = model.predict(X_test[image_index].reshape(1,32,32,1))
# print(pred.argmax())
print(data['Name'][pred.argmax()])
