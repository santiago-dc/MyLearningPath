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

X =np.concatenate((X_train, X_test), axis = 0)
Y =np.concatenate((Y_train, Y_test), axis = 0)
s = np.arange(X.shape[0])
np.random.shuffle(s)
X = X[s]
Y = Y[s]
X_conv_train = X[0:3000]#2637
X_conv_test = X[3000:]
Y_conv_train = Y[0:3000]
Y_conv_test = Y[3000:]


# Display first 15 images of moles, and how they are classified
# w=40
# h=30
# fig=plt.figure(figsize=(12, 8))
# columns = 5
# rows = 3

# for i in range(1, columns*rows +1):
#     ax = fig.add_subplot(rows, columns, i)
#     if Y_train[i] == 0:
#         ax.title.set_text('Benign')
#     else:
#         ax.title.set_text('Malignant')
#     plt.imshow(X_train[i],interpolation='nearest')
# plt.show()

#Turn labels into one hot encoding (ya veremos si esto hace falta o si lo dejo)
Y_conv_train = to_categorical(Y_conv_train, num_classes= 2)
Y_conv_test = to_categorical(Y_conv_test, num_classes= 2)

# Normalization
X_conv_train = X_conv_train/255.
X_conv_test = X_conv_test/255.

#Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, BatchNormalization, Dropout
model = Sequential()
model.add(Conv2D(64, kernel_size=7, activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=None))
model.add(BatchNormalization())
#model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=None))
model.add(BatchNormalization())
#model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=None))
model.add(BatchNormalization())
#model.add(Dropout(0.25))
from keras import regularizers  
model.add(Flatten())
#model.add(Dropout(0.25))
model.add(Dense(450, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(0.25))
model.add(Dense(250, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(0.25))
model.add(Dense(125, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(0.25))
model.add(Dense(30, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(2, activation='softmax')) 


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_conv_train, Y_conv_train, validation_data=(X_conv_test, Y_conv_test), batch_size=10, epochs=50)
test_loss, test_acc = model.evaluate(X_conv_test, Y_conv_test, batch_size=10)
print('Test Acc: ', test_acc)
print(':)')
