import numpy as np
import csv

# May 27 2020
# a bit of practice for numpy and logistic regression backpropagation and derivatives
# replicating the calculation from the website
# https://en.wikipedia.org/wiki/Logistic_regression


hours = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
passe = np.array([   0,    0,    0,    0,    0,    0,    1,    0,    1,    0,    1,    0,    1,    0,    1,    1,    1,    1,    1,    1])

eta = 0.1
w = 0
b = 0

for i in range(10000):
        z = w*hours+b
        a = 1/(1+np.exp(-z))
        c = -(np.multiply(passe,np.log(a))+np.multiply(1-passe,np.log(1-a)))
        cost = np.average(c)
        dz = a - passe
        dw = np.multiply(dz, hours)
        dw_ave = np.average(dw)
        db = np.average(dz)
        w = w - eta*dw_ave
        b = b - eta*db
print(w) # should get 1.5046
print(b) # should get -4.0777
