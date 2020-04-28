import numpy as np
import pandas as pd
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras import regularizers 
from keras.backend import clear_session 
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, BatchNormalization, Dropout

np.random.seed(5)
clear_session()

# def read(folder_path,flip):
#     data_path = os.path.join(folder_path,'*jpg')
#     folder = glob.glob(data_path)
#     matrix = []
#     for f in folder:
#         img = np.asarray(Image.open(f).convert("RGB"))
#         matrix.append(img)
#     if flip:
#         for i in range(0,len(matrix)):
#             if i%5==0 and i%2==0:#1 de cada 5 se gira
#                 matrix.append(np.fliplr(matrix[i]))#cuando acaba en 0, de izq a derecha
#             elif i%5==0:
#                 matrix.append(np.flipud(matrix[i]))#caundo acaba en 5, de arriba a abajo
#     matrix = np.asarray(matrix)
#     return matrix

def read(folder_path):
    data_path = os.path.join(folder_path,'*jpg')
    folder = glob.glob(data_path)
    matrix = []
    for f in folder:
        img = np.asarray(Image.open(f).convert("RGB"))
        matrix.append(img)
    matrix = np.asarray(matrix)
    return matrix

def build_model():
    #Model
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3,padding='same', strides=2, activation='relu', input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=None))#model.add(BatchNormalization())#model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3,padding='same', strides=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=None))#model.add(BatchNormalization())#model.add(Dropout(0.25))

    model.add(Conv2D(32, kernel_size=3,padding='same', strides=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=None))#model.add(BatchNormalization())#model.add(Dropout(0.25))

    #Normal layers
    model.add(Flatten())#model.add(Dropout(0.25))
    # model.add(Dense(450, activation='relu',kernel_regularizer=regularizers.l2(0.01)))# model.add(Dropout(0.25))
    # model.add(Dense(200, activation='relu',kernel_regularizer=regularizers.l2(0.01)))#model.add(Dropout(0.25))
    # model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01)))#model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(2, activation='softmax')) 
    return model

#Load data
benign_train_folder = 'data/train/benign'
malignant_train_folder = 'data/train/malignant'

benign_test_folder = 'data/test/benign'
malignant_test_folder = 'data/test/malignant'

#Create data
X_benign_train = read(benign_train_folder)
X_malignant_train = read(malignant_train_folder)
X_benign_test = read(benign_test_folder)
X_malignant_test = read(malignant_test_folder)

#Create labels
Y_benign_train = np.zeros(X_benign_train.shape[0])
Y_malignant_train = np.ones(X_malignant_train.shape[0])
Y_benign_test = np.zeros(X_benign_test.shape[0])
Y_malignant_test = np.ones(X_malignant_test.shape[0])

#Merge and Shuffle data
X_train = np.concatenate((X_benign_train, X_malignant_train), axis = 0)
Y_train = np.concatenate((Y_benign_train, Y_malignant_train), axis = 0)
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

X_test = np.concatenate((X_benign_test, X_malignant_test), axis = 0)
Y_test = np.concatenate((Y_benign_test, Y_malignant_test), axis = 0)
s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]

#Turn labels into one hot encoding (ya veremos si esto hace falta o si lo dejo)
Y_train = to_categorical(Y_train, num_classes= 2)
Y_test = to_categorical(Y_test, num_classes= 2)

# Normalization
X_train = X_train/255.
X_test = X_test/255.

#CNN
model = build_model()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#reduce_lr = ReduceLROnPlateau(monitor='val_loss',verbose=1,factor=0.5,mode='max',patience=5,min_lr=1e-7)

model.fit(X_train,Y_train,validation_data=(X_test, Y_test),batch_size=10,epochs=50)

test_loss, test_acc = model.evaluate(X_test, Y_test,batch_size=10)
print('Test Acc: ',test_acc,' :)')

