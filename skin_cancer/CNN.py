import numpy as np
import pandas as pd
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
np.random.seed(5)

benign_train_folder = 'data/train/benign'
malignant_train_folder = 'data/train/malignant'

benign_test_folder = 'data/test/benign'
malignant_test_folder = 'data/test/malignant'

def read(folder_path):
    data_path = os.path.join(folder_path,'*jpg')
    folder = glob.glob(data_path)
    matrix = []
    for f in folder:
        img = np.asarray(Image.open(f).convert("RGB"))
        matrix.append(img)
    matrix = np.asarray(matrix)
    return matrix

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

# Display first 15 images of moles, and how they are classified
w=40
h=30
fig=plt.figure(figsize=(12, 8))
columns = 5
rows = 3

for i in range(1, columns*rows +1):
    ax = fig.add_subplot(rows, columns, i)
    if Y_train[i] == 0:
        ax.title.set_text('Benign')
    else:
        ax.title.set_text('Malignant')
    plt.imshow(X_train[i],interpolation='nearest')
plt.show()

#Turn labels into one hot encoding (ya veremos si esto hace falta o si lo dejo)
Y_train = to_categorical(Y_train, num_classes= 2)
Y_test = to_categorical(Y_test, num_classes= 2)

# Normalization
X_train = X_train/255.
X_test = X_test/255.

#Model
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam

input_shape = (224,224,3)
lr = 1e-5
epochs = 50
#batch_size = 64

from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# model = Sequential() #Test Acc:  0.7757575511932373
# #add model layers
# model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(224,224,3)))#capa de 64 filtros de 3x3. dims 64*28*28*1
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=None))
# model.add(Conv2D(32, kernel_size=3, activation='relu'))#capa de 32 filtros de 3x3. dims 32*28*28*1
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=None))
# model.add(Flatten())# pasa los resultados de matriz a vector
# model.add(Dense(30, activation='relu'))
# model.add(Dense(2, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=3)

# test_loss, test_acc = model.evaluate(X_test, Y_test)
# print('Test Acc: ', test_acc)

model = Sequential() #Test Acc:  0.7757575511932373
#add model layers
model.add(Conv2D(128, kernel_size=5, activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=None))
model.add(Conv2D(128, kernel_size=5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=None))
model.add(Conv2D(64, kernel_size=5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=None))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=16, epochs=3)

test_loss, test_acc = model.evaluate(X_test, Y_test, batch_size=16)
print('Test Acc: ', test_acc)










print(':)')