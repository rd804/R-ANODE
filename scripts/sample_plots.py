import pickle
import numpy as np
import matplotlib.pyplot as plt

data_dir = "data/lhc_co"
nsig=1000
x_test = np.load(f'{data_dir}/x_test.npy')
label = x_test[:,-1]
true_signal = x_test[label == 1]
true_background = x_test[label == 0]
bins = np.linspace(0.0,1.0,100)
true_w = 0.006

data_histogram = {}


for try_ in range(10):
    data_histogram[try_] = {}
    
    with open(f'figures/sample_dict_SR_mass_{nsig}_{try_}.pkl', 'rb') as f:
        samples = pickle.load(f)
    

    weights = []

    for key in samples['SR'].keys():
      #  hist = np.array(samples['SR'][key])
    #  print(samples['SR'][key].shape)
        weight_value = float(key.split('_')[0])
        weights.append(weight_value)
        hist = np.array(samples['SR'][key])
        hist_data = [np.histogram(hist[:,i+1], bins = bins, density = True)[0] for i in range(4)]
        hist_data = np.array(hist_data)
        data_histogram[try_][weight_value] = hist_data


# average over the 10 tries
data_histogram_mean = {}
data_histogram_std = {}
for key in data_histogram[0].keys():
    data_histogram_mean[key] = {}
    data_histogram_std[key] = {}

    for i in range(4):
        data_histogram_mean[key][i] = np.mean([data_histogram[try_][key][i] for try_ in range(10)], axis = 0)
        data_histogram_std[key][i] = np.std([data_histogram[try_][key][i] for try_ in range(10)], axis = 0)

#############################
#############################
# true mass P_sig(x|m)
with open(f'figures/sample_dict_SR_mass_true_m_{nsig}_0.pkl', 'rb') as f:
        samples = pickle.load(f)

samples = samples['SR'][f'{true_w}_{nsig}']
hist = np.array(samples)
hist_data = [np.histogram(hist[:,i], bins = bins, density = True)[0] for i in range(4)]
hist_data = np.array(hist_data)
#data_histogram_mean['mass'] = {}
#data_histogram_std['mass'] = {}
#for i in range(4):
 #   data_histogram_mean['mass'][i] = hist_data[i]
  #  data_histogram_std['mass'][i] = np.zeros_like(hist_data[i])

##########################################
##########################################

# plot the histograms
fig, ax = plt.subplots(2,2, figsize = (16,8))
ax = ax.flatten()
for feature in range(4):
  ax[feature].hist(true_signal[:,feature+1], bins = bins, density = True, label = 'signal', color='black', alpha = 0.3)
  ax[feature].hist(true_background[:,feature+1], bins = bins, density = True, label = 'background', color='gray', alpha = 0.3)
  for weight in data_histogram_mean.keys():
    if weight == true_w:
      ax[feature].step(bins[:-1], data_histogram_mean[weight][feature], label = f'true_w={weight}')
    else:
     # ax[feature].step(bins[:-1], data_histogram_mean[weight][feature], label = weight)
      continue
    if weight != 'mass':
      ax[feature].fill_between(bins[:-1], data_histogram_mean[weight][feature] - data_histogram_std[weight][feature], data_histogram_mean[weight][feature] + data_histogram_std[weight][feature], step='pre', alpha = 0.3)
    ax[feature].set_title(f'Feature {feature}')
    ax[feature].legend()
  #  ax[feature].set_xlabel('Feature value')
    ax[feature].set_ylabel('Density')

# save the figure
plt.savefig(f'figures/sample_plots_SR_mass_with_background_true_weight_{nsig}.png')  

