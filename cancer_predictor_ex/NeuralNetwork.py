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
    return 1/(1+np.exp(-N))

def dSigmoid(N): 
    a = sigmoid(N)
    d = a * (1-a)
    return d

class NeuralNetwork:
    def __init__(self, x, y, lr):
        self.X=x
        self.Y=y
        self.Yh=np.zeros((1,self.Y.shape[1]))
        self.dims=[9,15,1]
        self.param={}
        self.lr=lr
        self.m = self.Y.shape[1]
        self.threshold=0.9
        self.error=[]
        self.m = self.Y.shape[1]
    
    def nInit(self): #function where we initialize with random values the parameters of our network
        np.random.seed(1)
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 
        self.param['b1'] = np.zeros((self.dims[1], 1))   
        self.param['a1'] = np.zeros((self.dims[1], 1))        
        self.param['N1'] = np.zeros((self.dims[1], 1))             
     
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
        self.param['b2'] = np.zeros((self.dims[2], 1))     
        self.param['a2'] = np.zeros((self.dims[2], 1))   
      

    # def costF(self,Yh): #Cross-Entropy Loss Function
    #     loss = (1./self.m) * (-np.dot(self.Y,np.log(Yh).T) - np.dot(1-self.Y, np.log(1-Yh).T)) #store error of the iteration   
    #     return loss

    def forward(self):
        N1 = self.param['W1'].dot(self.X) + self.param['b1']
        A1 = purelim(N1)
        N2 = self.param['W2'].dot(A1) + self.param['b2']
        A2 = sigmoid(N2)
        self.Yh,self.param['N1'],self.param['a1'],self.param['N2']=A2,N1,A1,N2
        error = np.square(np.subtract(self.Y,self.Yh)).mean()
        #error = mean_squared_error(self.Y,self.Yh)
        #error = self.costF(A2)
        return A2, error
    
    def backpropagation(self):# según la teoría de sci, no consigo que funcione, no lo uso
        s2 = -2 * np.dot((dSigmoid(self.param['N2'])),(self.Y-self.Yh).T)
        s1 = np.dot(self.param['a1'].T,self.param['W2'].T)*s2
        #Update weights and bias
        self.param["W2"] = self.param["W2"] - self.lr * s2 * self.param['a1'].T
        self.param["b2"] = self.param["b2"] - self.lr * s2 
        self.param["W1"] = self.param["W1"] - self.lr * s1 * self.X.T 
        self.param["b1"] = self.param["b1"] - self.lr * s1 

    def backward(self):
        derror = - (np.divide(self.Y, self.Yh ) - np.divide(1 - self.Y, 1 - self.Yh)) #derivate of error function, Cross-Entropy, not MSE
        s2 = derror * dSigmoid(self.param['N2']) #s2 = derivate of error function * derivate of sigmoid function 
        variaton_W2 = 1./self.param['a1'].shape[1] * np.dot(s2,self.param['a1'].T) #s2 * a1
        variaton_b2 = 1./self.param['a1'].shape[1] * np.dot(s2, np.ones([s2.shape[1],1])) #s2 * array of ones

        ws = np.dot(self.param["W2"].T,s2) #W2 * s2                    
        s1 = ws * self.param['N1']#s1 = W2 * s2 * derivate of the function       
        variaton_W1 = 1./self.X.shape[1] * np.dot(s1,self.X.T) #s1 * x
        variaton_b1 = 1./self.X.shape[1] * np.dot(s1, np.ones([s1.shape[1],1])) #s1 * array of ones 
        
        self.param["W1"] = self.param["W1"] - self.lr * variaton_W1 #W1 upgrade
        self.param["b1"] = self.param["b1"] - self.lr * variaton_b1 #b1 upgrade
        self.param["W2"] = self.param["W2"] - self.lr * variaton_W2 #W2 upgrade
        self.param["b2"] = self.param["b2"] - self.lr * variaton_b2 #b2 upgrade
        

    def gradient_descent(self, epochs):
        np.random.seed(1)                         
        self.nInit()#init weights and bias
        for i in range(0, epochs):#run
            Yh, error = self.forward()
            self.backward()
            if i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, error)) #cost value every 500 epochs
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

# def prueba():
#     df = pd.read_csv('wisconsin-cancer-dataset.csv',header=None)
#     df.head(5)
#     df.iloc[:,10].replace(2, 0,inplace=True)
#     df.iloc[:,10].replace(4, 1,inplace=True)
#     df = df[~df[6].isin(['?'])]
#     df = df.astype(float)
#     #Feature Normalization
#     names = df.columns[0:10]
#     scaler = MinMaxScaler() 
#     scaled_df = scaler.fit_transform(df.iloc[:,0:10]) 
#     scaled_df = pd.DataFrame(scaled_df, columns=names)
#     scaled_df[10]= df[10]
#     scaled_df.iloc[0:13,1:11].plot.bar();
#     scaled_df.iloc[0:13,1:11].plot.hist(alpha=0.5)
#     plt.show()
#     #creating train and validation sets
#     x=scaled_df.iloc[0:500,1:10].values.transpose()
#     y=df.iloc[0:500,10:].values.transpose()
#     xval=scaled_df.iloc[501:683,1:10].values.transpose()
#     yval=df.iloc[501:683,10:].values.transpose()
#     #declaring nn
#     nn = NeuralNetwork(x,y,0.02)
#     nn.gradient_descent(50000)#gradient descent algorithm
#     #predict and comparing training acurrancy vs validation acurrancy
#     pred_train = nn.predict(x, y)
#     pred_test = nn.predict(xval, yval)
#     #test with validation set and plot the skewed classes
#     nn.X,nn.Y=xval, yval 
#     yvalh, loss = nn.forward()
#     def plotCf(a,b,t):
#         # cm =confusion_matrix(a,b)
#         # class_labels = ['0','1']
#         # df_cm = pd.DataFrame(cm, index = class_labels, columns = class_labels)
#         # graph_matrix = sns.heatmap(df_cm, cmap = 'Blues', annot = True)
#         # graph_matrix.set(xlabel='Y value',ylabel='Yh value')
#         # plt.show()
#         cf =confusion_matrix(a,b)
#         plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
#         plt.colorbar()
#         plt.title(t)
#         plt.xlabel('Predicted')
#         plt.ylabel('Actual')
#         tick_marks = np.arange(len(set(a))) # length of classes
#         class_labels = ['0','1']
#         plt.xticks(np.ndarray([0,1]))
#         plt.yticks(np.ndarray([0,1]))
#         for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
#             plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] > (cf.max()*0.7) else 'black')
#         plt.show();
#     nn.threshold=0.9
#     nn.X,nn.Y=x, y 
#     target=np.around(np.squeeze(y), decimals=0).astype(np.int)
#     predicted=np.around(np.squeeze(nn.predict(x,y)), decimals=0).astype(np.int)
#     plotCf(target,predicted,'Cf Training Set')

#     nn.X,nn.Y=xval, yval 
#     target=np.around(np.squeeze(yval), decimals=0).astype(np.int)
#     predicted=np.around(np.squeeze(nn.predict(xval,yval)), decimals=0).astype(np.int)
#     plotCf(target,predicted,'Cf Validation Set')
# prueba()
