import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def l_i_f(Vt,Ie):
    dVt = 0
    C = 0.3 * (10**(-9)) #nF
    Vrest = -70 * (10**(-3)) #mV

    EL = -70 * (10**(-3)) #mV
    gL = 10 * (10**(-9)) #nSiemens
    dVt = ((gL*(EL-Vt))+Ie)/C


    return dVt

def euler_method(y0,t0,t1,h,Vthresh,Ie): #returns values of voltage until a spike occurs
    yValues = []
    t = t0
    y = y0
    spike =0

    while((y < Vthresh)and(t<=t1)):
        y+= h * l_i_f(y,Ie)
        t+= h
        yValues.append(y)
        if(y >= Vthresh):
            spike = 1

    return yValues,t, spike


def I_F(t0,t1,h,Vthresh,Vreset,V0,Ie):
    voltages = []
    spikes = 0
    t = t0

    while(t <= t1):
        yValues,t,spike = euler_method(V0,t,t1,h,Vthresh,Ie)
        spikes += spike
        V0 = Vreset
        voltages = np.concatenate((voltages,yValues))
    spikefreq  = spikes/t1
    return voltages, spikefreq

t0 = 0#ms
t1 = 10#s
V0 = np.random.randint(-70,-50)*10**(-3)#mV
h  = 0.1 * (10**(-3)) #ms
Vthresh = -50 * (10**(-3))#mV
Vreset= -60 * (10**(-3))#mV
Ie = 0.21 * (10**-(9)) #nA

voltages , spikefreq = I_F(t0,t1,h,Vthresh,Vreset,V0,Ie)
# print(spikefreq)

Ies = np.linspace(0, 0.5, num=100)
Ies = Ies*10**(-9)

spikefreqs = []
for I in Ies:
    _ , spikefreq = I_F(t0,t1,h,Vthresh,Vreset,V0,I)
    spikefreqs.append(spikefreq)

plt.plot(Ies,spikefreqs)
plt.show()

# plt.plot(Ies,spikefreqs)
# plt.show()
