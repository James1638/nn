import numpy as np
import numpy as np

#My first neural net from scratch. 
#This is code to mimic the example at 
#https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
#After 10,000 steps, I get an cost of 0.000037887 while Mazur had 0.000035108.
#I'm still figuring out the most efficient way to handle matrix operations in python / numpy.
#FSain 23 May 2020


eta = 0.5

W1 = np.array([[0.15, 0.20], [0.25, 0.30]])
W5 = np.array([[0.40, 0.45], [0.50, 0.55]])
T = np.array([0.01,0.99]).reshape(2,1)
I = np.array([0.05,0.10]).reshape(2,1)
B = np.array([0.35,0.60]).reshape(2,1)
F = np.array([[1,0], [1,0]])
S = np.array([[0,1], [0,1]])

Hn = np.dot(W1,I)+np.dot(F,B)
Ho = 1/(1+np.exp(-Hn))

On = np.dot(W5,Ho)+np.dot(S,B)
Oo = 1/(1+np.exp(-On))

E = T - Oo
C = np.dot([1,1],0.5*E**2)[0]
print(C)

temp1 = np.multiply(np.multiply(-E,Oo),1-Oo)
B[1]=B[1]-eta*np.dot(np.array([1,1]),temp1)	

for i in range(10000):

	junk = np.transpose(W5)
	#print(np.transpose(W5))
	#print
	#print(np.array([[.4,.5],[.45,.55]]))
	#print
	#print(Oo)

	#print(np.multiply( np.array([1,2]).reshape(2,1) ,np.array([[4,4.5],[5,5.5]])))
	#print(1-Oo)

	W5 = W5 - eta*np.dot(np.multiply(np.multiply(-E,Oo),1-Oo),Ho.reshape(1,2))
	#temp = np.dot(np.multiply(np.multiply(-E,Oo),1-Oo),Ho.reshape(1,2))
	#temp = np.dot(np.multiply(-(E),Oo,1-Oo),Ho.reshape(1,2))

	temp1 = np.multiply(np.multiply(-E,Oo),1-Oo)
	#temp2 = np.multiply( np.multiply( np.multiply(-E,Oo) , 1-Oo ) , np.array([.4, .45]).reshape(2,1) )
	#temp3 = np.multiply(np.multiply(np.multiply(-E,Oo),1-Oo),np.array([.5, .55]).reshape(2,1))
	temp4 = np.dot(np.array([1,1]),np.multiply(np.multiply(np.multiply(-E,Oo),1-Oo),junk)).reshape(2,1)
	#temp5 = np.multiply(np.multiply(np.multiply(-E,Oo),1-Oo),np.transpose(W5))
	#temp5 = np.multiply(np.multiply(np.multiply(-E,Oo),1-Oo),junk)
	temp6 =  np.dot( np.multiply(Ho,1-Ho), I.reshape(1,2) )
	
	B[0]=B[0]-eta*np.dot(np.array([1,1]),np.multiply(temp4,np.multiply(Ho,1-Ho)))	
	B[1]=B[1]-eta*np.dot(np.array([1,1]),temp1)	

	temp7 = np.multiply(temp4,temp6)
	
	W1 = W1 - eta*temp7
	
	Hn = np.dot(W1,I)+np.dot(F,B)
	Ho = 1/(1+np.exp(-Hn))

	On = np.dot(W5,Ho)+np.dot(S,B)
	Oo = 1/(1+np.exp(-On))

	E = T - Oo
C = np.dot([1,1],0.5*E**2)[0]
print(C)

