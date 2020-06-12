import numpy as np
import csv

# June 12 2020
# This is my first CNN, written in order to learn how the backpropagation works.
# Input: MNIST picture, 28 by 28 pixels.
# Convolution layer 5 by 5, 32 filters, ReLu activation.
# NO Max Pool Layer; I needed something simpler for my first time making all the backpropagation work.
# Fully Connected layer of 100 neurons, ReLu activation.
# Softmax output layer.
#
# Debugged with the help of gradient 
# submitted to https://www.kaggle.com/c/digit-recognizer/submit, score 0.93
# Not as good as my best, but does demonstrate that I got the convolution and backpropagation right (most probably).

def sigmoid(z):
        return 1/(1+np.exp(-z))

def softmax(z):
        e_z=np.exp(z-np.max(z))
        ezs = e_z.sum(axis=1).reshape(1,1)
        return e_z / ezs 

def softmax100(z):  #should make the reshaping number a function parameter
        e_z=np.exp(z-np.max(z))
        ezs = e_z.sum(axis=1).reshape(28000,1)
        return e_z / ezs 

def cost(Yhat, Y):
        return np.average(np.sum(np.multiply(-Y,np.log(Yhat)),axis=1))


# Random initialization with seeds specified to help with debugging.
# I only tried to initialize weights to be small and positive.
# Separated out biases B1 and B2 to make debugging a bit easier.
np.random.seed(1)
W1 = np.random.uniform(low=0.001,high=0.011,size=(32,5,5))
np.random.seed(1)
B1 = np.random.uniform(low=0.001,high=0.011,size=(32))
np.random.seed(1)
W2 = np.random.uniform(low=0.001,high=0.011,size=(18432,100))
np.random.seed(1)
B2 = np.random.uniform(low=0.001,high=0.011,size=(100))
np.random.seed(1)
W3 = np.random.uniform(low=0.001,high=0.011,size=(101, 10)) #weights


eta = 0.05 #step size

# Since training data is only 42000 lines, read them all in.
from numpy import genfromtxt
aP = np.multiply(genfromtxt('train.csv',delimiter=','), 1.0/255.0)
aY1 = genfromtxt('train.csv',delimiter=',',usecols=(0)).reshape(42000,1)

for skiphnum in range(84000): #cycle through each record in the training data twice
        skipfnum = 42000 - skiphnum - 1 #had been used to read records, now just a counter
        P = aP[skiphnum%42000]
        Y1 = aY1[skiphnum%42000]
        ones400 = np.ones((1,1)) #used to pad a weights matrix with ones if incorporating bias

        # the classification Y needs to be put into vector form
        # the digit it is gets a 1, the rest a 0
        Y = np.zeros((1,10))
        for ii in range(1):
                Y[ii,Y1[0]]=1
        
        X1 = np.delete(P,0,axis=0).reshape(28,28) #delete first entry, which is actually the classification Y

        # convolution layer
        Z1=np.zeros((32,24,24))
        for filters in range(32):
                for h_lo in range(24):
                        h_hi=h_lo+5
                        for v_lo in range(24):
                                v_hi=v_lo+5
                                Z1[filters,h_lo,v_lo]=np.dot(W1[filters,:,:].ravel(),X1[h_lo:h_hi,v_lo:v_hi].ravel())+B1[filters]
        # ReLu activation                
        X2 = np.multiply(Z1,0.5*(np.sign(Z1)+1))
        Z2 = np.dot(X2.ravel(),W2)+B2
        X3 = np.append(ones400,np.maximum(Z2,np.zeros((1,100))),axis=1) # ReLu activation
        # output layer
        Z3 = np.dot(X3,W3)
        Yhat = softmax(Z3)
        dC = cost(Yhat, Y)

        # backpropagation
        # layer 3 from FC to output
        dCdZ3 = Yhat-Y; 
        dCdX3 = np.dot(dCdZ3,W3.T); 
        dCdW3 = np.outer(X3.T,dCdZ3); 
        # layer 2 from Convolution to FC
        dCdZ2 = np.multiply(np.delete(dCdX3,0,1) ,0.5*(np.sign(np.delete(X3,0,1))+1))
        dCdB2 = dCdZ2; 
        dCdX2 = np.dot(dCdZ2,W2.T)
        dCdW2 = np.outer(X2.ravel().T,dCdZ2)
        # layer 1 from Input to Convolution
        X2r = X2.ravel()
        dCdZ1 = np.multiply(dCdX2 ,0.5*(np.sign(X2r)+1))
        dCdB1=np.zeros((32))
        dCdW1=np.zeros((32,5,5))
        dCdZ1r = dCdZ1.reshape(32,24,24)
        for filters in range(32): #need to learn how to vectorize this
                for h_lo in range(24):
                        h_hi=h_lo+5
                        for v_lo in range(24):
                                v_hi=v_lo+5
                                dCdW1[filters,:,:] += np.multiply(X1[h_lo:h_hi,v_lo:v_hi],dCdZ1r[filters,h_lo,v_lo])
                                dCdB1[filters] += np.sum(dCdZ1r[filters,h_lo,v_lo])
        # update weight matrices
        # in later versions used momentum, but not here
        W3 = W3 - np.multiply(0.00001*eta,dCdW3) #this matrix needed to be updated with a smaller step
        W2 = W2 - np.multiply(eta,dCdW2)
        W1 = W1 - np.multiply(eta,dCdW1)
        B1 = B1 - np.multiply(eta,dCdB1)
        B2 = np.add(B2,np.multiply(-1.0*eta,dCdB2))

        # keep track of progress from the terminal
        if skiphnum%100==0:
                print(dC, skiphnum, skipfnum, Y1[0], Y)

print("done with training")

#beginning of testing. Load testing data.
atP = np.multiply(genfromtxt('test.csv',delimiter=','), 1.0/255.0)
atY1 = genfromtxt('test.csv',delimiter=',',usecols=(0)).reshape(28000,1)
print("done with loading test.csv")
# specify output file. Run forward propagation on each record of testing data. Keep weight and bias matrices.
with open("mnist_CNN_02_guess","w") as f:
        f.write("ImageId,Label\n")
        for skiphnum in range(28000):
                skipfnum = 28000 - skiphnum - 1
                tP = atP[skiphnum]
                tY1 = atY1[skiphnum]
                ones400 = np.ones((1,1))
                tY = np.zeros((1,10))
                for ii in range(1):
                        tY[ii,tY1[0]]=1
                tX1 = tP.reshape(28,28)
                tZ1=np.zeros((32,24,24))
                for filters in range(32):
                        for h_lo in range(24):
                                h_hi=h_lo+5
                                for v_lo in range(24):
                                        v_hi=v_lo+5
                                        tZ1[filters,h_lo,v_lo]=np.dot(W1[filters,:,:].ravel(),tX1[h_lo:h_hi,v_lo:v_hi].ravel())+B1[filters]
                tX2 = np.multiply(tZ1,0.5*(np.sign(tZ1)+1))
                tZ2 = np.dot(tX2.ravel(),W2)+B2
                tX3 = np.append(ones400,np.maximum(tZ2,np.zeros((1,100))),axis=1) # ReLu activation
                tZ3 = np.dot(tX3,W3)
                tYhat = softmax(tZ3)
                # output to file for submission to kaggle
                f.write("{idx},{val}\n".format(idx=skiphnum+1,val=np.argmax(tYhat)))
                if skiphnum%100==0:
                        print(skiphnum) #keep track of progress from terminal

