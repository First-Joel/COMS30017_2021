
""" This is Question 1.0.1 of the coursework."""
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

spikes = np.loadtxt('neuron_A.csv', delimiter=',')
trial_IDs = np.loadtxt('trial_ID.csv', delimiter=',')

def get_fano_factor(spike_counts):
    sc_var = np.var(spike_counts)
    sc_mean = np.mean(spike_counts)
    #Variance of spike count divided by the mean of the spike count  NOT INTERSPIKE INTERVALS
    f = sc_var/sc_mean
    return f

def get_cv(interspike_intervals):
    #Standard deviation of interspike interval divided by the mean of the interspike interval
    isi_mean = np.mean(interspike_intervals)
    print("mean ",isi_mean)
    isi_std = np.std(interspike_intervals)
    cv = isi_std/isi_mean
    return cv

def spike_count_variance():
    return

def spike_count_histogram(stimulus_on_SCs,stimulus_off_SCs):
    bins = 25
    plt.hist(stimulus_on_SCs,bins,alpha=0.5)
    plt.hist(stimulus_off_SCs,bins,alpha=0.5)
    plt.xlabel('Spike count')
    plt.ylabel('Trials')
    plt.show()
    return

def get_spike_counts(trials):
    spike_counts = []
    for trial in trials:
        spike_counts.append(len(trial))
    return spike_counts

def get_interspike_intervals(spikes):
    interspike_intervals =[]
    for i in range(0,len(spikes)-1):
        interval = spikes[i+1] - spikes[i]
        interspike_intervals.append(interval)
    return interspike_intervals

def get_intervals_by_trials(trials):
    total_intervals =[]
    for trial in trials:
        intervals = get_interspike_intervals(trial)
        total_intervals = np.concatenate((total_intervals,intervals))
    return total_intervals

def get_trial_spikes(spikes,trial_number,offset,bin_size):
    trial_spikes = []
    min = (trial_number-1)*bin_size
    max = min + bin_size
    if spikes[offset]<min:
        print("Error: first spike is smaller than trial")
    while spikes[offset]<max:
        trial_spikes.append(spikes[offset])
        offset = offset+1
        if (offset>=len(spikes)):
            break

    return trial_spikes , offset

def separate_spikes_by_bin_size(spikes,bin_size):
    offset = 0
    trials = []
    for i in range(1,1+(math.ceil(spikes[-1]/bin_size))):
        trial_i, offset = get_trial_spikes(spikes,i,offset,bin_size)
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

def decoder(single_spike_count,decision_boundary):
    if(single_spike_count>=decision_boundary):
        return 1
    else:
        return 0

def question_1(spikes,trial_IDs):
    return
def question_2(spikes,trial_IDs):
    return

for i in [100,200,500,1000,2000]:
    f = get_fano_factor(get_spike_counts(separate_spikes_by_bin_size(spikes,i)))
    print("F",i,":  ",f)


trials = separate_spikes_by_bin_size(spikes,1000)
spike_counts= get_spike_counts(trials)
stimulus_on_trials, stimulus_off_trials = seperate_trials_by_stimulus(trials,trial_IDs)
stimulus_on_SCs =get_spike_counts(stimulus_on_trials)
stimulus_off_SCs = get_spike_counts(stimulus_off_trials)

spike_count_histogram(stimulus_on_SCs,stimulus_off_SCs)

interspike_intervals = get_interspike_intervals(spikes)
cv = get_cv(interspike_intervals)

stimulus_on_intervals = get_intervals_by_trials(stimulus_on_trials)
stimulus_on_cv = get_cv(stimulus_on_intervals)
stimulus_off_intervals = get_intervals_by_trials(stimulus_off_trials)
stimulus_off_cv = get_cv(stimulus_off_intervals)



print("coefficient_of_variation:",cv)
print("cv with stimulus", stimulus_on_cv)
print("cv with no stimulus", stimulus_off_cv)
