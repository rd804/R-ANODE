import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.mixture import GaussianMixture as GMM
from src.utils import *
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.cluster import KMeans
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


with open('data/background.pkl', 'rb') as f:
    background = pickle.load(f)


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

gmm = GMM(n_components=1, covariance_type='full', tol=1e-3, max_iter=1, init_params='random')

gmmfit = gmm.fit(background.reshape(-1,1))

mu_background = gmmfit.means_[0][0]
sigma_background = gmmfit.covariances_[0][0][0]

print('mu_background: ', mu_background)
print('sigma_background: ', sigma_background)



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
        convergence_ ,mu_data , sigma_data, w_data, like_arr = EM_2_gaussian(x_train, n_optimal, 1000, 
                                                                            1e-32,
                                                        init_params='random')
        w_max = np.argmax(w_data[-1]).flatten()[0]

        w_back = w_data[-1][w_max]

        mu_sig = mu_data[-1][1-w_max]
        sigma_sig = sigma_data[-1][1-w_max]

        print('mu_sig: ', mu_sig)
        print('sigma_sig: ', sigma_sig)
        print('w_back: ', w_back)

        convergence_ ,mu_data , sigma_data, w_data, like_arr = EM_2_gaussian(x_train, n_optimal, 1000, 
                                                                            1e-32,
                                                        init_params='all fixed',
                                                        mu_back=mu_background,
                                                        sigma_back=sigma_background, p_back=w_back,
                                                        mu_sig=mu_sig, sigma_sig=sigma_sig)
                                                        

        
        
        
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



    with open(f'results/best_parameters_modified_with_weights_100_epochs_em_sig_{sig_train}.pkl', 'wb') as f:
        pickle.dump(best_parameters[str(sig_train)], f)

        














