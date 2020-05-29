import numpy as np
import csv

# May 29 2020
# an implementation of the example in https://www.gormanalysis.com/blog/neural-networks-a-worked-example/
# on testing got 87% correct, roughly
# something is slightly unstable about the results, but they're in that ballpark
# I think it might be more than just the random initial weights, but maybe that's it.

def sigmoid(z):
        return 1/(1+np.exp(-z))

def softmax(z):
        e_z=np.exp(z-np.max(z))
        ezs = e_z.sum(axis=1).reshape(400,1)
        return e_z / ezs 

def softmax100(z):  #should make the reshaping number a function parameter
        e_z=np.exp(z-np.max(z))
        ezs = e_z.sum(axis=1).reshape(100,1)
        return e_z / ezs 

def cost(Yhat, Y):
        return np.average(np.sum(np.multiply(-Y,np.log(Yhat)),axis=1))

eta = 0.1    
from numpy import genfromtxt
P = np.multiply(genfromtxt('stairs_train.csv',delimiter=',',usecols=(1,2,3,4)), 1.0/1.0)
Y1 = genfromtxt('stairs_train.csv',delimiter=',',usecols=(5)).reshape(400,1) 
ones400 = np.ones((400,1))

Y = np.append(Y1,ones400-Y1.reshape(400,1),axis=1) #still figuring out broadcasting
X1 = np.append(ones400,P,axis=1)
W1 = np.random.uniform(low=-0.01,high=0.01,size=(5, 2))
W2 = np.random.uniform(low=-0.01,high=0.01,size=(3, 2))
for i in range(3000):
        # forward propagation
        Z1 = np.dot(X1,W1)
        X2 = np.append(ones400,sigmoid(Z1),axis=1)
        Z2 = np.dot(X2,W2)
        Yhat = softmax(Z2)
        dC = cost(Yhat, Y)
        print(dC) #just to monitor progress
        # forward propagation
        dCdZ2 = Yhat-Y
        dCdX2 = np.dot(dCdZ2,W2.T)
        dCdZ1 = np.multiply(np.delete(dCdX2,0,1) ,np.multiply(np.delete(X2,0,1),1-np.delete(X2,0,1)))
        dCdW2 = np.zeros((3,2))
        for j in range(400):  #there must be a way to vectorize this
                dCdW2 += np.outer(X2.T[:,j],dCdZ2[j,:])/400
        dCdW1 = np.zeros((5,2))
        for k in range(400):  #there must be a way to vectorize this
                dCdW1 += np.outer(X1.T[:,k],dCdZ1[k,:])/400
        W1 = W1 - np.multiply(eta,dCdW1)
        W2 = W2 - np.multiply(eta,dCdW2)
        

########### TESTING
# should make this forward propagation into a function

tP = np.multiply(genfromtxt('stairs_test.csv',delimiter=',',usecols=(1,2,3,4)), 1.0/1.0)
tY1 = genfromtxt('stairs_test.csv',delimiter=',',usecols=(5)).reshape(100,1)
ones100 = np.ones((100,1))
tY = np.append(tY1,ones100-tY1.reshape(100,1),axis=1) #still figuring out broadcasting
tX1 = np.append(ones100,tP,axis=1)
tZ1 = np.dot(tX1,W1)
tX2 = np.append(ones100,sigmoid(tZ1),axis=1)
tZ2 = np.dot(tX2,W2)
tYhat = softmax100(tZ2)
print(np.round(tYhat[:,0],decimals=0)) #to see the predictions
print(tY1.reshape(1,100)) #to see what the results should be
print("Percentage incorrect:")
print(np.sum(np.round(tYhat[:,0],decimals=0)-tY1.reshape(1,100)[0]))
        
