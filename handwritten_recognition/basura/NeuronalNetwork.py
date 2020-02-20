import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import itertools
import seaborn as sns

def purelim(N):
    return N

def sigmoid(N):
    x = 1/(1+np.exp(-N))
    print(x)
    return x

def dSigmoid(N): 
    a = sigmoid(N)
    d = a * (1-a)
    return d

class NeuronalNetwork:
    def __init__(self, x, y, lr, dims):
        self.X=x
        self.Y=y
        self.Yh=np.zeros((self.Y.shape))
        self.dims= dims
        self.param={}
        self.lr=lr
        self.m = self.Y.shape[0]
        self.threshold=0.5
        self.error=[]
        self.m = self.Y.shape[0]

    #TO DO: esto es una chapuza, modularizarlo
    def nInit(self): #function where we initialize with random values the parameters of our network
        np.random.seed(1)
        #hidden layer 1
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 
        self.param['b1'] = np.zeros((self.dims[1], 1))   
        self.param['a1'] = np.zeros((self.dims[1], 1))        
        self.param['N1'] = np.zeros((self.dims[1], 1))             
        #hidden layer 2
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
        self.param['b2'] = np.zeros((self.dims[2], 1))   
        self.param['a2'] = np.zeros((self.dims[2], 1))        
        self.param['N2'] = np.zeros((self.dims[2], 1))  
        # #hidden layer 3
        # self.param['W3'] = np.random.randn(self.dims[3], self.dims[2]) / np.sqrt(self.dims[2]) 
        # self.param['b3'] = np.zeros((self.dims[3], 1))   
        # self.param['a3'] = np.zeros((self.dims[3], 1))     
        # self.param['N3'] = np.zeros((self.dims[3], 1))  
        # #output layer 
        # self.param['W4'] = np.random.randn(self.dims[4], self.dims[3]) / np.sqrt(self.dims[3]) 
        # self.param['b4'] = np.zeros((self.dims[4], 1))   
        # self.param['a4'] = np.zeros((self.dims[4], 1))     
        # self.param['N4'] = np.zeros((self.dims[4], 1))  
      

    # def costF(self,Yh): #Cross-Entropy Loss Function
    #     loss = (1./self.m) * (-np.dot(self.Y,np.log(Yh).T) - np.dot(1-self.Y, np.log(1-Yh).T)) #store error of the iteration   
    #     return loss


    #TO DO: esto es una chapuza, modularizarlo
    def forward(self):
        N1 = self.param['W1'].dot(self.X) + self.param['b1']
        A1 = purelim(N1)
        N2 = self.param['W2'].dot(A1) + self.param['b2']
        A2 = sigmoid(N2)
        self.Yh,self.param['N1'],self.param['a1'],self.param['N2']=A2,N1,A1,N2
        error = np.square(np.subtract(self.Y,self.Yh.T)).mean()
        #error = mean_squared_error(self.Y,self.Yh)
        #error = self.costF(A2)
        return A2, error

        # N3 = self.param['W3'].dot(self.param['a2']) + self.param['b3']
        # A3 = purelim(N3)
        # self.param['N3'],self.param['a3']=N3,A3

        # N4 = self.param['W4'].dot(self.param['a3']) + self.param['b4']
        # A4 = sigmoid(N4)
        # self.param['N4'],self.Yh=N4,A4

        # self.Yh=self.Yh.transpose()
        # self.param['N2']=self.param['N2'].transpose()
        # error = np.square(np.subtract(self.Y,self.Yh)).mean()
        # return A2, error #A4,error
    
    def backpropagation(self):# según la teoría de sci, no consigo que funcione, no lo uso
        s2 = -2 * np.dot((dSigmoid(self.param['N2'])),(self.Y-self.Yh).T)
        s1 = np.dot(self.param['a1'].T,self.param['W2'].T)*s2
        #Update weights and bias
        self.param["W2"] = self.param["W2"] - self.lr * s2 * self.param['a1'].T
        self.param["b2"] = self.param["b2"] - self.lr * s2 
        self.param["W1"] = self.param["W1"] - self.lr * s1 * self.X.T 
        self.param["b1"] = self.param["b1"] - self.lr * s1 

    def backward(self):
        #derror = - (np.divide(self.Y, self.Yh.T) - np.divide(1 - self.Y, 1 - self.Yh.T)) #derivate of error function, Cross-Entropy, not MSE
        derror = np.subtract(self.Y,self.Yh.T).mean()#RMSE
        s2 = derror * dSigmoid(self.param['N2'].T) #s2 = derivate of error function * derivate of sigmoid function 
        variaton_W2 = 1./self.param['a1'].shape[1] * np.dot(s2.T,self.param['a1'].T) #s2 * a1
        variaton_b2 = 1./self.param['a1'].shape[1] * np.dot(s2, np.ones([s2.shape[1],1])).T #s2 * array of ones

        ws = np.dot(self.param["W2"].T,s2.T) #W2 * s2                    
        s1 = ws * self.param['N1']#s1 = W2 * s2 * derivate of the function       
        variaton_W1 = 1./self.X.shape[1] * np.dot(s1,self.X.T) #s1 * x
        variaton_b1 = 1./self.X.shape[1] * np.dot(s1, np.ones([s1.shape[1],1])) #s1 * array of ones 
        
        self.param["W1"] = self.param["W1"] - self.lr * variaton_W1 #W1 upgrade
        self.param["b1"] = self.param["b1"] - self.lr * variaton_b1 #b1 upgrade
        self.param["W2"] = self.param["W2"] - self.lr * variaton_W2 #W2 upgrade
        self.param["b2"] = self.param["b2"] - self.lr * variaton_b2 #b2 upgrade

    def backward_mal(self):
        #derror = - (np.divide(self.Y, self.Yh.T) - np.divide(1 - self.Y, 1 - self.Yh.T)) #derivate of error function, Cross-Entropy, not MSE
        #derror = -(2/self.Y.shape[0])*(np.subtract(self.Y,self.Yh.T).mean())#sMSE
        derror = np.subtract(self.Y,self.Yh.T).mean()#RMSE
        print('error ',derror)
        print('n4 ', self.param['N4'])
        s4 = derror * dSigmoid(self.param['N4']) #s4 = derivate of error function * derivate of sigmoid function 
        print('s4 ',s4)
        variaton_W4 = 1./self.param['a3'].shape[1] * np.dot(s4,self.param['a3'].T) #s4 * a3
        variaton_b4 = 1./self.param['a3'].shape[1] * np.dot(s4, np.ones([s4.shape[1],1])) #s4 * array of ones
        self.param["W4"] = self.param["W4"] - self.lr * variaton_W4 
        self.param["b4"] = self.param["b4"] - self.lr * variaton_b4 
        print('variation w4', variaton_W4)
        print('new w4 ', self.param["W4"])
        print('a3 ', self.param['a3'])
        ws4 = np.dot(self.param["W4"].T,s4)                
        s3 = ws4 * self.param['N3']     
        variaton_W3 = 1./self.param['a2'].shape[1] * np.dot(s3,self.param['a2'].T) 
        variaton_b3 = 1./self.param['a2'].shape[1] * np.dot(s3, np.ones([s3.shape[1],1]))
        self.param["W3"] = self.param["W3"] - self.lr * variaton_W3 
        self.param["b3"] = self.param["b3"] - self.lr * variaton_b3 

        ws3 = np.dot(self.param["W3"].T,s3)                 
        s2 = ws3 * self.param['N2']       
        variaton_W2 = 1./self.param['a1'].shape[1] * np.dot(s2,self.param['a1'].T) 
        variaton_b2 = 1./self.param['a1'].shape[1] * np.dot(s2, np.ones([s2.shape[1],1])) 
        self.param["W2"] = self.param["W2"] - self.lr * variaton_W2 
        self.param["b2"] = self.param["b2"] - self.lr * variaton_b2 

        ws2 = np.dot(self.param["W2"].T,s2) 
        s1 = ws2 * self.param['N1']     
        variaton_W1 = 1./self.X.shape[1] * np.dot(s1,self.X.T) 
        variaton_b1 = 1./self.X.shape[1] * np.dot(s1, np.ones([s1.shape[1],1]))
        self.param["W1"] = self.param["W1"] - self.lr * variaton_W1 
        self.param["b1"] = self.param["b1"] - self.lr * variaton_b1       

    def gradient_descend(self, epochs):
        np.random.seed(1)                         
        self.nInit()#init weights and bias
        for i in range(0, epochs):#run
            Yh, error = self.forward()
            self.backward()
            if i % 100 == 0:
                #print ("Cost after iteration %i: %f" %(i, error)) #cost value every 100 epochs
                self.error.append(error) #we only updates the error values every 500 epochs to get harder modifications 

        plt.plot(np.squeeze(self.error))
        plt.ylabel('Loss')
        plt.xlabel('Iter')
        plt.title("Lr =" + str(self.lr))
        plt.show()
        
    def predict(self, x, y):#predict after the model training finished  
        self.X=x
        self.Y=y
        comp = np.zeros((1,x.shape[1]))
        pred, error = self.forward()    
    
        for i in range(0, pred.shape[1]):
            if pred[0,i] > self.threshold: comp[0,i] = 1
            else: comp[0,i] = 0
    
        print("Acc: " + str(np.sum((comp == y)/x.shape[1]))) 
        return comp
    

#borrador del notebook no descomentar
df_train = pd.read_csv('data/mnist_train.csv', delimiter = ',')
df_test = pd.read_csv('data/mnist_test.csv', delimiter = ',')

df_train = df_train.dropna(how='any',axis=0)
df_test = df_test.dropna(how='any',axis=0)

y_train_raw = df_train.iloc[:, 0].to_numpy()
y_train = []
for i in range(0,y_train_raw.shape[0]):
    row=np.zeros(10)
    row[y_train_raw[i]]=1
    y_train.append(row)
y_train=np.array(y_train)
y_train.reshape(y_train.shape[0],10)
x_train = df_train.drop(df_train.columns[0], axis='columns').to_numpy()

y_test_raw = df_test.iloc[:, 0].to_numpy()
y_test = []
for i in range(0,y_test_raw.shape[0]):
    row=np.zeros(10)
    row[y_test_raw[i]]=1
    y_test.append(row)
y_test=np.array(y_test)
y_test.reshape(y_test.shape[0],10)
x_test = df_test.drop(df_test.columns[0], axis='columns').to_numpy()

del df_train
del df_test
# frac = 0.99/ 255
# x_train = x_train * frac + 0.01

nn = NeuronalNetwork(x_train.T,y_train,0.01,[784,128,10])
nn.gradient_descend(2000)