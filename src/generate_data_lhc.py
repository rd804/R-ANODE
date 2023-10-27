#import pandas as pd
import numpy as np
import os
import argparse
import vector
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.utils import shuffle





def deltaR(features_df):
    j1_vec = vector.array({
        "px": np.array(features_df[["pxj1"]]),
        "py": np.array(features_df[["pyj1"]]),
        "pz": np.array(features_df[["pzj1"]]),
    })
    j2_vec = vector.array({
        "px": np.array(features_df[["pxj2"]]),
        "py": np.array(features_df[["pyj2"]]),
        "pz": np.array(features_df[["pzj2"]]),
    })
    return j1_vec.deltaR(j2_vec).flatten()


def separate_SB_SR(data, minmass, maxmass):
    innermask = (data[:, 0] > minmass) & (data[:, 0] < maxmass)
    outermask = ~innermask
    return data[innermask], data[outermask]


# the "data" containing too much signal
def resample_split(data_dir, n_sig = 1000 , resample_seed = 1,\
                   minmass = 3.3, maxmass = 3.7, resample = True,
                   noise = False, noise_seed = 0, n_noise = 5):
    background = np.load(f'{data_dir}/data_bg.npy')
    signal = np.load(f'{data_dir}/data_sig.npy')[:-30_000]

    if noise:
        total_data = len(signal) + len(background)
        np.random.seed(noise_seed)
        noise = np.random.multivariate_normal(np.zeros(n_noise), np.eye(n_noise), total_data)
        signal = np.concatenate((signal, noise[:len(signal)]), axis=1)
        background = np.concatenate((background, noise[len(signal):]), axis=1)

    # choose 1000 signal events
    # random choice of 1000 signal events
    if resample:
        print(f'choosing random {n_sig} signal events for mock_data from {len(signal)} events')
        np.random.seed(resample_seed)
        choice = np.random.choice(len(signal), n_sig, replace=False)
        signal = signal[choice]
    else:
        print(f'using first {n_sig} signal events for mock_data from {len(signal)} events')
        signal = signal[:n_sig]


    # concatenate background and signal
    data = np.concatenate((background, signal),axis=0)
    data = shuffle(data, random_state=resample_seed)

    SR_data, CR_data = separate_SB_SR(data, minmass, maxmass)
    
    S = SR_data[SR_data[:, -1]==1]
    B = SR_data[SR_data[:, -1]==0]

    true_w = len(S)/(len(B)+len(S))
    sigma = len(S)/np.sqrt(len(B))

    print(f'sigma={sigma}')
    print(f'true w: {true_w}')

    return SR_data, CR_data, true_w, sigma


#if __name__ == '__main__':
 #   path = './input_data'

  #  SR, CR, true_w, sigma = resample_split(path, n_sig = 1000,
   #                                 resample_seed = 1)
  #  print('SR: ', SR.shape)
  #  print('CR: ', CR.shape)
