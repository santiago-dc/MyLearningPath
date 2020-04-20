import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras import regularizers 
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import cv2
from PIL import Image
import os

def load_train(folders,size):
    X=[]
    Y=[]
    for i in range(folders) :
        path = "D:/santi/Documents/MyLearningPath/traffic_sign/data/Train/{0}/".format(i)
        folder=os.listdir(path)
        for img in folder:
            try:
                image=cv2.imread(path+img)
                image_RGB = Image.fromarray(image, 'RGB')
                resized_image = image_RGB.resize((size[0], size[1]))
                X.append(np.array(resized_image))
                Y.append(i)
            except AttributeError:
                print("Error loading picture ", img)
    return np.asarray(X),np.asarray(Y)

def build_model():
    #Model
    model = Sequential()
    model.add(Conv2D(3, kernel_size=7,padding='same', strides=2, activation='relu', input_shape=(30,30,3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=None))#model.add(BatchNormalization())
    #model.add(Dropout(0.25))

    model.add(Conv2D(3, kernel_size=3,padding='same', strides=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=None))#model.add(BatchNormalization())
    #model.add(Dropout(0.25))

    # model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(rate=0.25))
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(rate=0.25))
    model.add(Flatten())
    #model.add(Dropout(rate=0.5))
    model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(43, activation='softmax'))
    return model
    
X,Y=load_train(43,[30,30])

s=np.arange(X.shape[0])
np.random.seed(43)
np.random.shuffle(s)
X=X[s]
Y=Y[s]


#Spliting the images into train and validation sets
(X_train,X_val)=X[(int)(0.2*len(Y)):],X[:(int)(0.2*len(Y))]
X_train = X_train.astype('float32')/255 
X_val = X_val.astype('float32')/255
(y_train,y_val)=Y[(int)(0.2*len(Y)):],Y[:(int)(0.2*len(Y))]

#Using one hote encoding for the train and validation Y
from keras.utils import to_categorical
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)

# %%
#Compilation of the model
model=build_model()
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)
#using ten epochs for the training and saving the accuracy for each epoch
epochs = 150
model.fit(X_train, y_train, batch_size=32, epochs=epochs,validation_data=(X_val, y_val))

print(':)')