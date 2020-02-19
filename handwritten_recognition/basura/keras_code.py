#https://www.youtube.com/watch?v=cvNtZqphr6A

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras 

#load data
df_train = pd.read_csv('data/mnist_train.csv', delimiter = ',')
df_test = pd.read_csv('data/mnist_test.csv', delimiter = ',')

df_train = df_train.dropna(how='any',axis=0)
df_test = df_test.dropna(how='any',axis=0)

def extract():
    df_train = pd.read_csv('data/mnist_train.csv', delimiter = ',').to_numpy()
    df_test = pd.read_csv('data/mnist_test.csv', delimiter = ',').to_numpy()
    df = pd.DataFrame(np.concatenate((df_train,df_test)))
    return df

def transform(df): 
    y_raw = df.iloc[:, 0].to_numpy()
    x = df.drop(df.columns[0], axis='columns').to_numpy()
    y = []
    for i in range(0,y_raw.shape[0]):
        row=np.zeros(10)
        row[y_raw[i]]=1
        y.append(row)
    y=np.array(y)
    y.reshape(y.shape[0],10)
    return x, y

df = extract()
x_train,y_train = transform(df.iloc[0:60000])
#x_val,y_val = transform(df.iloc[50000:60000])
x_test,y_test = transform(df.iloc[60000:70000])

# Dense default values:
#                 activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs)

model = keras.Sequential([
    keras.layers.Dense(784, input_dim=784),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")#Softmax is kind of Multi Class Sigmoid: https://www.quora.com/Why-is-it-better-to-use-Softmax-function-than-sigmoid-function
])
print('xshape:', x_train.shape, 'y_shape:', y_train.shape)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 50, batch_size=200, verbose=2)

tes_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Acc: ', test_acc)