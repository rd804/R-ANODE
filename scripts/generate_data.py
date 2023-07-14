import numpy as np
import argparse
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--gaussian_dim',  default=2)
parser.add_argument('--signal_mean', default=2.0)
parser.add_argument('--signal_sigma', default=0.25)
parser.add_argument('--background_mean', default=0.0)
parser.add_argument('--background_sigma', default=3)
parser.add_argument('--n_back', default=200000)


args = parser.parse_args()


gaussian_dim = args.gaussian_dim
signal_mean = args.signal_mean
signal_sigma = args.signal_sigma
background_mean = args.background_mean
background_sigma = args.background_sigma
n_back = args.n_back


data_dict = {}
true_w_dict = {}

sig_list = [0.1, 0.2, 0.5, 0.8, 0.9, 1, 1.5, 2, 5, 10]
sigma_test = 100
for sig in sig_list:
    data_dict[str(sig)] = {}
    data_dict[str(sig)]['train'] = {}
    data_dict[str(sig)]['val'] = {}

    n_sig = int(sig * np.sqrt(n_back))
    n_test = int(sigma_test * np.sqrt(n_back))

    sig_data = np.random.normal(signal_mean, signal_sigma, (n_sig, gaussian_dim))
    back_data = np.random.normal(background_mean, background_sigma, (n_back, gaussian_dim))

    sig_test = np.random.normal(signal_mean, signal_sigma, (n_test, gaussian_dim))
    back_test = np.random.normal(background_mean, background_sigma, (n_back, gaussian_dim))

    true_w = len(sig_data)/(len(back_data) + len(sig_data))

    data_dict[str(sig)]['train']['data'] = np.concatenate((sig_data, back_data), axis=0)
    data_dict[str(sig)]['train']['label'] = np.concatenate((np.ones(len(sig_data)), np.zeros(len(back_data))), axis=0)

    data_dict[str(sig)]['val']['data'] = np.concatenate((sig_test, back_test), axis=0)
    data_dict[str(sig)]['val']['label'] = np.concatenate((np.ones(len(sig_test)), np.zeros(len(back_test))), axis=0)

    true_w_dict[str(sig)] = [true_w, 1-true_w]

    print(f'for sig_train = {sig}, true_w = {true_w}')
    print(f'amount of signal = {len(sig_data)}')
    print(f'signal shape = {sig_data.shape}')
    print(f'amount of background = {back_data.shape}')
    print(f'amount of background = {len(back_data)}')


with open(f'data/data_{args.gaussian_dim}d.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

with open(f'data/true_w_{args.gaussian_dim}d.pkl', 'wb') as f:
    pickle.dump(true_w_dict, f)

n_background = 100000
background = np.random.normal(background_mean, background_sigma, (n_background, gaussian_dim))

with open(f'data/background_{args.gaussian_dim}d.pkl', 'wb') as f:
    pickle.dump(background, f)

print('background shape: ', background.shape)











