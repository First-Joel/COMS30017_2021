import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_ps = np.loadtxt('train_images.csv', delimiter=',')
test_ps = np.loadtxt('test_images.csv', delimiter=',')

# plt.imshow(np.reshape(train_ps[2],(28,28)))
# plt.show()




print(train_ps)
train_t = train_ps.T
print(train_t[0:4])

print(len(train_ps))
print(len(train_t))


def weight_formula(xis,xjs,n):
    w=0.0
    for a in range(0,n):
        w += xis[a]*xjs[a]
    w = w/n
    return w

def weight_matrix(train_t,n):

    w_matrix = np.empty([len(train_t),len(train_t)])
    for i in range(0,len(train_t)):
        for j in range(0,len(train_t)):
            if(i==j):
                w = 0
            else:
                w = weight_formula(train_t[i],train_t[j],n)
            w_matrix[i,j] = w
    return w_matrix

def evolve_formula(i,xi_t0,xjs,w_matrix,theta):
    for j in xjs:
        ffadjafdhjk += w_matrix[i,j]*xj

    if(ffadjafdhjk > theta):
        return 1
    else:
        return -1
def evolve(steps)

w_matrix = weight_matrix(train_t,3)
print(w_matrix)

plt.imshow(w_matrix)
plt.show()
