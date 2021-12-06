
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import enum


spikes_A = np.loadtxt('neuron_A.csv', delimiter=',')
spikes_B = np.loadtxt('neuron_B.csv', delimiter=',')
trial_IDs = np.loadtxt('trial_ID.csv', delimiter=',')

#given an array of spike counts returns the fano factor
def get_fano_factor(spike_counts):
    sc_var = np.var(spike_counts)
    sc_mean = np.mean(spike_counts)
    #Variance of spike count divided by the mean of the spike count  NOT INTERSPIKE INTERVALS
    f = sc_var/sc_mean
    return f
#given an array of intervals returns the coefficient of variation
def get_cv(interspike_intervals):
    #Standard deviation of interspike interval divided by the mean of the interspike interval
    isi_mean = np.mean(interspike_intervals)
    isi_std = np.std(interspike_intervals)
    cv = isi_std/isi_mean
    return cv
#gets the count of a list of trials
def get_spike_counts2(trials):
    spike_counts = []
    for trial in trials:
        spike_counts.append(len(trial))
    return spike_counts

def get_spike_counts(trials):
    length = np.vectorize(len)
    return length(trials)

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

def get_single_trial_spikes(spikes,trial_number,offset,bin_size):
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
        trial_i, offset = get_single_trial_spikes(spikes,i,offset,bin_size)
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

#plots histogram for stimulus and no stimulus spike counts
def plot_SC_hist(stimulus_on_SCs,stimulus_off_SCs):
    bins = 25
    plt.hist(stimulus_on_SCs,bins,alpha=0.5)
    plt.hist(stimulus_off_SCs,bins,alpha=0.5)
    plt.xlabel('Spike count')
    plt.ylabel('Trials')
    plt.show()
    return

#returns prediction based on decision boundary
def decoder(single_spike_count,boundary):
    if(single_spike_count>=boundary):
        return 1
    else:
        return 0

# returns how the decoder faired against the true value
def decoder_vs_true_value(trial,trial_ID,boundary):
        prediction = decoder(len(trial),boundary)
        if (trial_ID ==1 and prediction ==1):
            return "TP"
        elif(trial_ID==0 and prediction ==0):
            return "TN"
        elif (trial_ID ==0 and prediction ==1):
            return "FP"
        elif (trial_ID ==1 and prediction ==0):
            return "FN"
        else:
            return "ERROR not zero or one"

#runs the decoder for different values of decision boundary in range 0-40
def decode_boundaries2(trials,trial_IDs):
    tps = []
    tns = []
    tcs = []
    for boundary in range(0,41):
        true_positives = 0
        true_negatives = 0
        total_correct = 0
        for t in range(0,len(trials)):
            check = decoder_vs_true_value(trials[t],trial_IDs[t],boundary)
            if (check == "TP"):
                true_positives+=1
                total_correct+=1
            elif(check == "TN"):
                true_negatives+=1
                total_correct+=1
        tps.append(true_positives)
        tns.append(true_negatives)
        tcs.append(total_correct)
    return tps,tns,tcs

def decode_boundaries(trials,trial_IDs):
    for boundary in range(0,41):
        vdecode = np.vectorize(decoder)
        predictions = vdecode(trials,boundary)


def plot_decoder(tps,tns,tcs):
    plt.plot(tps)
    plt.plot(tns)
    plt.plot(tcs)
    plt.show()


def question_1(spikes,trial_IDs):
    return
def question_2(spikes,trial_IDs):
    return




trials = np.array(separate_spikes_by_bin_size(spikes_A,1000))
spike_counts = get_spike_counts(trials)
stimulus_on_trials, stimulus_off_trials = seperate_trials_by_stimulus(trials,trial_IDs)
print(stimulus_on_trials)
print("\n")
print(stimulus_off_trials)


cv = get_cv(get_interspike_intervals(spikes_A))
stimulus_on_cv = get_cv(get_intervals_by_trials(stimulus_on_trials))
stimulus_off_cv = get_cv(get_intervals_by_trials(stimulus_off_trials))

print("\n")
print("coefficient_of_variation:",cv)
print("cv with stimulus", stimulus_on_cv)
print("cv with no stimulus", stimulus_off_cv)
print("\n")

for i in [100,200,500,1000]:
    f = get_fano_factor(get_spike_counts(separate_spikes_by_bin_size(spikes_A,i)))
    print("fano factor with binSize",i,":  ",f)

#tps,tns,tcs = decode_boundaries(trials,trial_IDs)
#plot_decoder(tps,tns,tcs)
plot_SC_hist(get_spike_counts(stimulus_on_trials),get_spike_counts(stimulus_off_trials))
