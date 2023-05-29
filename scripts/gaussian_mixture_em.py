import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.mixture import GaussianMixture as GMM
from src.utils import *
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.cluster import KMeans
# import train test split
from sklearn.model_selection import train_test_split
import argparse
import pickle

parser = argparse.ArgumentParser(description='Run EM 10 times and save the results')
parser.add_argument('--n', type=int, default=10, help='number of times to run EM')
parser.add_argument('--data', type = str, default = 'data/data.pkl', help = 'path to data')
parser.add_argument('--sig_train', default = 10, help = 'sigma for training data')

args = parser.parse_args()

# load data
with open(args.data, 'rb') as f:
    data = pickle.load(f)

back_mean = 0
sig_mean = 3
sig_simga = 0.5
back_sigma = 3


with open('results/background_only_fit.pkl', 'rb') as f:
    background_only_fit = pickle.load(f)

[mu_background , sigma_background] = background_only_fit


# fit train data
best_parameters = {}
run = 0

sig_train = args.sig_train

best_parameters[str(sig_train)] = {}
x_train = data[str(sig_train)]['train']['data']

print('#################################')
print('#################################')
print('#################################')
print('---------------------------------')





for run in range(10):


    best_parameters[str(sig_train)][str(run)] = {}

    print('#################################')
    print
    print('Running EM algorithm for sig = {}'.format(sig_train))
    n_optimal = 2
    trial = 0
    best_likelihood = -np.inf
    convergence_ = False

    while convergence_ != True:
        trial += 1
        convergence_ ,mu_data , sigma_data, w_data, like_arr = EM_2_gaussian(x_train, n_optimal, 2000, 
                                                                            1e-32,
                                                        init_params='random',
                                                        mu_back=mu_background,
                                                        sigma_back=sigma_background)
        
        max_likelihood = np.argmax(like_arr)


        mu_ = mu_data[max_likelihood]
        sigma_ = sigma_data[max_likelihood]
        w_ = w_data[max_likelihood]
    

        if like_arr[max_likelihood] > best_likelihood:
            best_likelihood = like_arr[max_likelihood]
            best_mu = mu_
            best_sigma = sigma_
            best_w = w_
            best_trial = trial
            best_like_arr = like_arr


            best_parameters[str(sig_train)][str(run)]['mu'] = best_mu
            best_parameters[str(sig_train)][str(run)]['sigma'] = best_sigma
            best_parameters[str(sig_train)][str(run)]['w'] = best_w
            

        if trial > 100:
            break

    if convergence_ == True:
        print('Converged after {} trials'.format(trial))
        print('Best trial was {}'.format(best_trial))

        best_parameters[str(sig_train)][str(run)]['converged_mu'] = mu_
        best_parameters[str(sig_train)][str(run)]['converged_sigma'] = sigma_
        best_parameters[str(sig_train)][str(run)]['converged_w'] = w_

    else:
        print('Did not converge after {} trials'.format(trial))
        print('Best trial was {}'.format(best_trial))

    
    # save best parameters to pickle file



    with open(f'results/best_parameters_sig_{sig_train}.pkl', 'wb') as f:
        pickle.dump(best_parameters[str(sig_train)], f)

        














