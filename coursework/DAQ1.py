
""" This is Question 1.0.1 of the coursework."""
import os
import sys
import math
import numpy as np
import pandas as pd

spikes = np.loadtxt('neuron_A.csv', delimiter=',')
trial_IDs = np.loadtxt('trial_ID.csv', delimiter=',')

def get_fano_factor(sc_var,sc_mean):
    #Variance of spike count divided by the mean of the spike count  NOT INTERSPIKE INTERVALS
    f = sc_var/sc_mean
    return f

def get_cv(interspike_intervals):
    #Standard deviation of interspike interval divided by the mean of the interspike interval
    isi_mean = np.mean(interspike_intervals)
    isi_std = np.std(interspike_intervals)
    cv = isi_std/isi_mean
    return cv

def spike_count_variance():
    return

def spike_count_mean(spike_counts):
    return

def get_interspike_intervals(spikes):
    interspike_intervals =[]
    for i in range(0,len(spikes)-1):
        interval = spikes[i+1] - spikes[i]
        interspike_intervals.append(interval)
    return interspike_intervals

def get_intervals_by_trials(trials):
    total_intervals =[]
    for trial in stimulus_trials:
        intervals = get_interspike_intervals(trial)
        total_intervals = numpy.concatenate((total_intervals,intervals))
    return total_intervals

def get_trial_spikes(spikes,trial_number,offset):
    trial_spikes = []
    min = (trial_number-1)*1000
    max = min + 1000
    if spikes[offset]<min:
        print("Error: first spike is smaller than trial")
    while spikes[offset]<max:
        trial_spikes.append(spikes[offset])
        offset = offset+1
        if (offset>=len(spikes)):
            break

    return trial_spikes , offset

def separate_spikes_by_trial(spikes,trial_IDs):
    offset = 0
    trials = []
    for i in range(1,(1+len(trial_IDs))):
        trial_i, offset = get_trial_spikes(spikes,i,offset)
        trials.append(trial_i)
    return trials

def seperate_trials_by_stimulus(trials,trial_IDs):
    #Make sure the trials line up
    stimulus_trials=[]
    no_stimulus_trials=[]
    for i in range(0,len(trial_IDs)):
        if(trial_IDs[i] == 1):
            stimulus_trials.append(trials[i])
        elif(trial_IDs[i] ==0):
            no_stimulus_trials.append(trials[i])
        else:
            print("ERROR: NOT ZERO OR ONE")

    return stimulus_trials, no_stimulus_trials


def question_1():
    return

trials = separate_spikes_by_trial(spikes,trial_IDs)
stimulus_trials, no_stimulus_trials = seperate_trials_by_stimulus(trials,trial_IDs)
interspike_intervals = get_interspike_intervals(spikes)
cv = get_cv(interspike_intervals)

stimulus_intervals = get_intervals_by_trials(stimulus_trials)
stimulus_cv = get_cv(stimulus_intervals)
no_stimulus_intervals = get_intervals_by_trials(no_stimulus_trials)
no_stimulus_cv = get_cv(no_stimulus_intervals)




print("coefficient_of_variation:",cv)
print("cv with stimulus", stimulus_cv)
print("cv with no stimulus", no_stimulus_cv)
