import pandas as pd
import numpy as np
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
#Load data
X = np.load('data/X.npy')
Y = np.load('data/Y.npy')
X_conv=X.reshape(X.shape[0], X.shape[1], X.shape[2],1)
#Split data
X_train = X_conv[0:1750]
Y_train = Y[0:1750]
X_test = X_conv[1750:]
Y_test = Y[1750:]

#Model
model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(64,64,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=11, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
#Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Train
model.fit(X_train, Y_train, batch_size=64, epochs=3)
#Test
test_loss, test_acc = model.evaluate(X_test, Y_test, batch_size=16)
print('Test Acc: ', test_acc)




print(':)')