import numpy as np
import csv

# June 01 2020
# running this got about 0.946 on this Kaggle competition https://www.kaggle.com/c/digit-recognizer
# Again, this hasn't been cleaned up and made efficient.
# I spent some time working on the input and the output, reading only parts of a file, using only
# some columns, and formatting the output.
# Instead of cleaning this up, I am going to spend my limited time in trying to take the next step
# in compexity and sophistication of the algorithm.

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
W1 = np.random.uniform(low=-0.01,high=0.01,size=(785, 300)) #weights
W2 = np.random.uniform(low=-0.01,high=0.01,size=(301, 10)) #weights
eta = 0.1    
from numpy import genfromtxt
skiphnum = 0
skiphnum_arr = [0,6000,12000,18000,24000,30000,36000]
for skiphnum in skiphnum_arr:
#for skiphnum in range(1):
        skipfnum = 42000 - skiphnum - 6000
        #this one works
        #P = np.multiply(genfromtxt('mnist_train_400.csv',delimiter=','), 1.0/1.0)
        P = np.multiply(genfromtxt('train.csv',delimiter=',', skip_header=skiphnum, skip_footer=skipfnum), 1.0/1.0)
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
        for i in range(60):
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
                dCdW2 = np.zeros((301,10))
                for j in range(6000):  #there must be a way to vectorize this
                        dCdW2 += np.outer(X2.T[:,j],dCdZ2[j,:])/6000
                dCdW1 = np.zeros((785,300))
                for k in range(6000):  #there must be a way to vectorize this
                        dCdW1 += np.outer(X1.T[:,k],dCdZ1[k,:])/6000
                W1 = W1 - np.multiply(eta,dCdW1)
                W2 = W2 - np.multiply(eta,dCdW2)
        

########### TESTING
# should make this forward propagation into a function

#P = np.multiply(genfromtxt('mnist_train.csv',delimiter=',', skip_header=skiphnum, skip_footer=skipfnum), 1.0/1.0)
tP = np.multiply(genfromtxt('test.csv',delimiter=','), 1.0/1.0)
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
tX2 = np.append(ones100,sigmoid(tZ1),axis=1)
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
np.savetxt("mnist_kaggle2_guess",np.round(np.argmax(tYhat,axis=1),0).astype(int))
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
with open("mnist_guess2","w") as f:
        f.write("ImageId,Label\n")
        for jj in range(28000):
                f.write("{idx},{val}\n".format(idx=jj+1,val=np.argmax(tYhat[jj])))
tYhatint = tYhat.astype(int)
with open("mnist_guess2_int","w") as f:
        for mm in range(28000):
                f.write(tYhatint[jj])
#print(tY1.reshape(1,10000)) #to see what the results should be
#print("Percentage incorrect:")
#print(np.sum(np.round(tYhat[:,0],decimals=0)-tY1.reshape(1,100)[0]))
       
