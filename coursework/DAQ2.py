import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_ps = np.loadtxt('train_images.csv', delimiter=',')
test_ps = np.loadtxt('test_images.csv', delimiter=',')
train_t = train_ps.T



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

def evolve_formula(i,xi_t0,xs,w_matrix,theta):
    total = 0
    for j in range(0,len(xs)):
        total += w_matrix[i,j]*xs[j]

    if( total > theta):
        return 1
    else:
        return -1

def evolve_random_x(xs,w_matrix,theta):
    i = np.random.randint(0,len(xs))
    xi_t0 = xs[i]
    xs[i] = evolve_formula(i,xi_t0,xs,w_matrix,theta)
    return xs

def evolve_xs(xs,w_matrix,steps,theta):
    for i in range(0,steps):
        xs = evolve_random_x(xs,w_matrix,theta)
    return xs

def energy_formula(xs,w_matrix):
    total =0
    for i in range(0,len(xs)):
        for j in range(0,len(xs)):
            total += w_matrix[i,j]*xs[i]*xs[j]
    e = -0.5*total
    return e

def mix_image(A,B,prob):
    mask = np.random.choice([0,1], size= len(A), p=[prob,1-prob])
    inv_mask = 1-mask

    masked_A = A*mask
    masked_B = B*inv_mask
    image = masked_A+masked_B
    return image

w_matrix = weight_matrix(train_t,3)

mixed_image = mix_image(train_ps[0],train_ps[1],0.95)
plt.imshow(np.reshape(mixed_image,(28,28)))
plt.show()

test1 = test_ps[0]
test2 = test_ps[1]


"""plt.imshow(np.reshape(test1,(28,28)))
plt.show()
print("energy",energy_formula(test1,w_matrix))
test1 = evolve_xs(test1,w_matrix,1000,0)
plt.imshow(np.reshape(test1,(28,28)))
plt.show()
print("energy",energy_formula(test1,w_matrix))
test1 = evolve_xs(test1,w_matrix,2000,0)
plt.imshow(np.reshape(test1,(28,28)))
plt.show()
print("energy",energy_formula(test1,w_matrix))

print("energy",energy_formula(test2,w_matrix))
plt.imshow(np.reshape(test2,(28,28)))
plt.show()
test2 = evolve_xs(test2,w_matrix,1000,0)
print("energy",energy_formula(test2,w_matrix))
plt.imshow(np.reshape(test2,(28,28)))
plt.show()
test2 = evolve_xs(test2,w_matrix,2000,0)
print("energy",energy_formula(test2,w_matrix))
plt.imshow(np.reshape(test2,(28,28)))
plt.show()"""
