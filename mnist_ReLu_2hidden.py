import numpy as np
import csv

# June 07 2020
# This is 2 hidden ReLu layers with softmax output and momentum for kaggle mnist competition. 
# https://www.kaggle.com/c/digit-recognizer/leaderboard
# Got 0.78114, which is terrible, but which shows that the two layers works (most probably).
# I'm going to prioritize learning a CNN.
# So unless I learn something which tells me about a mistake I made here,
# I'm going to leave this as-is.

def sigmoid(z):
        return 1/(1+np.exp(-z))

def softmax(z):
        e_z=np.exp(z-np.max(z))
        ezs = e_z.sum(axis=1).reshape(6000,1)
        return e_z / ezs 

def softmax100(z):  #should make the reshaping number a function parameter
        e_z=np.exp(z-np.max(z))
        ezs = e_z.sum(axis=1).reshape(28000,1)
        return e_z / ezs 

def cost(Yhat, Y):
        return np.average(np.sum(np.multiply(-Y,np.log(Yhat)),axis=1))

#initialization
W1 = np.random.uniform(low=0.001,high=0.011,size=(785, 300)) #weights
W5 = np.random.uniform(low=0.001,high=0.011,size=(301, 300)) #weights, 5 is short for 1.5
W2 = np.random.uniform(low=0.001,high=0.011,size=(301, 10)) #weights
eta = 0.1
totalsteps=0
from numpy import genfromtxt
skiphnum = 0
skiphnum_arr = [0,6000,12000,18000,24000,30000,36000]
for skiphnum in skiphnum_arr:
#for skiphnum in range(1):
        skipfnum = 42000 - skiphnum - 6000
        #this one works
        #P = np.multiply(genfromtxt('mnist_train_400.csv',delimiter=','), 1.0/1.0)
        P = np.multiply(genfromtxt('train.csv',delimiter=',', skip_header=skiphnum, skip_footer=skipfnum), 1.0/255.0)
        print("P")
        print(np.shape(P))
        #print(P)
        print
        #P = genfromtxt('mnist_train.csv',delimiter=',', skip_header=skiphnum, skip_footer=skipfnum)
        #P = np.delete(np.multiply(genfromtxt('mnist_train.csv',delimiter=',', skip_header=skiphnum, skip_footer=skipfnum), 1.0/1.0),0,axis=1)
        #this one works
        #Y1 = genfromtxt('mnist_train_400.csv',delimiter=',',usecols=(0)).reshape(400,1)
        print(skiphnum)
        print(skipfnum)
        Y1 = genfromtxt('train.csv',delimiter=',',usecols=(0), skip_header=skiphnum, skip_footer=skipfnum).reshape(6000,1) 
        ones400 = np.ones((6000,1))
        print("Y1")
        print(np.shape(Y1))
        #print(Y1)
        print
        
        Y = np.zeros((6000,10))
        for ii in range(6000):
                Y[ii,Y1[ii][0]]=1
        
        #Y = np.append(Y1,ones400-Y1.reshape(400,1),axis=1) #still figuring out broadcasting
        X1 = np.append(ones400,np.delete(P,0,axis=1),axis=1)
        print("X1")
        print(np.shape(X1))
        #print(X1)
        print
        print("Y")
        print(np.shape(Y))
        #print(Y)
        print
        dCdW2old = np.zeros((301,10))
        dCdW5old = np.zeros((301,300))
        dCdW1old = np.zeros((785,300))
        for i in range(150):
                totalsteps=totalsteps+1
                #eta = 0.5/(1.0+totalsteps/105.0)
                #print("eta")
                #print(eta)
                # forward propagation
                Z1 = np.dot(X1,W1)
                #print("Z1")
                #print(np.shape(Z1))
                #print(Z1)
                #print
                #print("Z1 max")
                #print(np.maximum(Z1,np.zeros((6000,300))))
                #print
                X5 = np.append(ones400,np.maximum(Z1,np.zeros((6000,300))),axis=1)
                #X5 = np.append(ones400,sigmoid(Z1),axis=1)
                #print("X5")
                #print(np.shape(X5))
                #print(X5)
                #print
                Z5 = np.dot(X5,W5)
                #print("Z5")
                #print(Z5)
                #print
                X2 = np.append(ones400,np.maximum(Z5,np.zeros((6000,300))),axis=1)
                #X2 = np.append(ones400,sigmoid(Z5),axis=1)
                #print("X2")
                #print(np.shape(X2))
                #print(X2)
                #print
                Z2 = np.dot(X2,W2)
                #print("Z2")
                #print(Z2)
                #print
                Yhat = softmax(Z2)
                #print("Yhat")
                #print(Yhat)
                #print
                dC = cost(Yhat, Y)
                #print("dC") #just to monitor progress
                print(dC) #just to monitor progress
                # backward propagation
                dCdZ2 = Yhat-Y
                dCdX2 = np.dot(dCdZ2,W2.T)
                dCdW2 = np.zeros((301,10))
                for j in range(6000):  #there must be a way to vectorize this
                        dCdW2 += np.outer(X2.T[:,j],dCdZ2[j,:])/6000

                dCdZ5 = np.multiply(np.delete(dCdX2,0,1) ,np.multiply(np.delete(X2,0,1),0.5*(np.sign(np.delete(X2,0,1))+1)))
                #dCdZ5 = np.multiply(np.delete(dCdX2,0,1) ,np.multiply(np.delete(X2,0,1),np.heaviside(np.delete(X2,0,1),0)))
                #dCdZ5 = np.multiply(np.delete(dCdX2,0,1) ,np.multiply(np.delete(X2,0,1),1-np.delete(X2,0,1)))
                dCdX5 = np.dot(dCdZ5,W5.T)
                dCdW5 = np.zeros((301,300))
                for k in range(6000):  #there must be a way to vectorize this
                        dCdW5 += np.outer(X5.T[:,k],dCdZ5[k,:])/6000

                dCdZ1 = np.multiply(np.delete(dCdX5,0,1) ,np.multiply(np.delete(X5,0,1),0.5*(np.sign(np.delete(X5,0,1))+1)))
                #dCdZ1 = np.multiply(np.delete(dCdX5,0,1) ,np.multiply(np.delete(X5,0,1),np.heaviside(np.delete(X5,0,1),0)))
                #dCdZ1 = np.multiply(np.delete(dCdX5,0,1) ,np.multiply(np.delete(X5,0,1),1-np.delete(X5,0,1)))
                dCdW1 = np.zeros((785,300))
                for k in range(6000):  #there must be a way to vectorize this
                        dCdW1 += np.outer(X1.T[:,k],dCdZ1[k,:])/6000

                alpha = 0.9
                W1 = W1 - np.multiply(eta,dCdW1) - np.multiply(eta*alpha,dCdW1old)
                W5 = W5 - np.multiply(eta,dCdW5) - np.multiply(eta*alpha,dCdW5old)
                W2 = W2 - np.multiply(eta,dCdW2) - np.multiply(eta*alpha,dCdW2old)
                dCdW1old = dCdW1
                dCdW5old = dCdW5
                dCdW2old = dCdW2

                #W1 = W1 - np.multiply(eta,dCdW1)
                #W5 = W5 - np.multiply(eta,dCdW5)
                #W2 = W2 - np.multiply(eta,dCdW2)
        

########### TESTING
# should make this forward propagation into a function
'''
                # forward propagation
                Z1 = np.dot(X1,W1)
                X5 = np.append(ones400,sigmoid(Z1),axis=1)
                Z5 = np.dot(X5,W5)
                X2 = np.append(ones400,sigmoid(Z5),axis=1)
                Z2 = np.dot(X2,W2)
                Yhat = softmax(Z2)
'''
#P = np.multiply(genfromtxt('mnist_train.csv',delimiter=',', skip_header=skiphnum, skip_footer=skipfnum), 1.0/1.0)
tP = np.multiply(genfromtxt('test.csv',delimiter=','), 1.0/255.0)
#tY1 = genfromtxt('test.csv',delimiter=',',usecols=(0)).reshape(10000,1)
ones100 = np.ones((28000,1))
#tY = np.append(tY1,ones100-tY1.reshape(10000,1),axis=1) #still figuring out broadcasting
#tY = tY1
#X1 = np.append(ones400,np.delete(P,0,axis=1),axis=1)
tX1 = np.append(ones100,tP,axis=1)
print("tX1")
print(np.shape(tX1))
print(tX1)
print
print("W1")
print(np.shape(W1))
print(W1)
print
tZ1 = np.dot(tX1,W1)
tX5 = np.append(ones100,np.maximum(tZ1,np.zeros((28000,300))),axis=1)
#tX5 = np.append(ones100,sigmoid(tZ1),axis=1)
tZ5 = np.dot(tX5,W5)
tX2 = np.append(ones100,np.maximum(tZ5,np.zeros((28000,300))),axis=1)
#tX2 = np.append(ones100,sigmoid(tZ5),axis=1)
tZ2 = np.dot(tX2,W2)
tYhat = softmax100(tZ2)
#print(tY) #to see the truth
print(np.argmax(tYhat,axis=1)) #to see the predictions
#for i in range(len(tY)):
#        print(i,tY[i,0],np.argmax(tYhat[i,:]))
#diffs = tY[:,0]-np.argmax(tYhat,axis=1)
#print("diffs2")
#print(diffs)
#print(np.count_nonzero(diffs))
np.savetxt("mnist_ReLu2_guess",np.round(np.argmax(tYhat,axis=1),0).astype(int))
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
with open("mnist_ReLu_guess2","w") as f:
        f.write("ImageId,Label\n")
        for jj in range(28000):
                f.write("{idx},{val}\n".format(idx=jj+1,val=np.argmax(tYhat[jj])))
tYhatint = tYhat.astype(int)
with open("mnist_ReLu_guess2_int","w") as f:
        for mm in range(28000):
                f.write(tYhatint[jj])
#print(tY1.reshape(1,10000)) #to see what the results should be
#print("Percentage incorrect:")
#print(np.sum(np.round(tYhat[:,0],decimals=0)-tY1.reshape(1,100)[0]))
       
