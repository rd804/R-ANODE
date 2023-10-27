import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.cluster import KMeans
from scipy.stats import norm
import torch
from src.density_estimator import DensityEstimator
from scipy.interpolate import interp1d
import numpy.ma as ma
# SIC curve

# some data preprocessing functions
def logit_transform(x, min_vals, max_vals):
    with np.errstate(divide='ignore', invalid='ignore'):
        x_norm = (x - min_vals) / (max_vals - min_vals)
        logit = np.log(x_norm / (1 - x_norm))
    domain_mask = ~(np.isnan(logit).any(axis=1) | np.isinf(logit).any(axis=1))
    return logit, domain_mask

def standardize(x, mean, std):
    return (x - mean) / std

def inverse_logit_transform(x, min_vals, max_vals):
    x_norm = 1 / (1 + torch.exp(-x))
    return x_norm * (max_vals - min_vals) + min_vals

def inverse_standardize(x, mean, std):
    return x * std + mean

def preprocess_params_fit_all(data):
    preprocessing_params = {}
    preprocessing_params["min"] = np.min(data[:,1:-1], axis=0)
    preprocessing_params["max"] = np.max(data[:,1:-1], axis=0)

    preprocessed_data_x, mask = logit_transform(data[:,1:-1], preprocessing_params["min"], preprocessing_params["max"])
    preprocessed_data = np.hstack([data[:,0].reshape(-1,1),preprocessed_data_x, data[:,-1].reshape(-1,1)])[mask]

    preprocessing_params["mean"] = np.mean(preprocessed_data[:,1:-1], axis=0)
    preprocessing_params["std"] = np.std(preprocessed_data[:,1:-1], axis=0)

    return preprocessing_params

def preprocess_params_transform_feature(data, params):
    preprocessed_data_x, mask = logit_transform(data,
                                                params["min"], params["max"])
    preprocessed_data_x = standardize(preprocessed_data_x, params["mean"], params["std"])
    return preprocessed_data_x


def preprocess_params_transform_all(data, params):
    preprocessed_data_x, mask = logit_transform(data[:,:-1],
                                                 params["min"], params["max"])
    preprocessed_data_x = standardize(preprocessed_data_x, params["mean"], params["std"])
    preprocessed_data = np.hstack([preprocessed_data_x, data[:,-1].reshape(-1,1)])[mask]


    return preprocessed_data


def inverse_transform(data, params):
    inverse_data = inverse_standardize(data[:, 1:-1], params["mean"], params["std"])
    inverse_data = inverse_logit_transform(inverse_data, params["min"], params["max"])
    inverse_data = torch.hstack([data[:, 0:1], inverse_data, data[:, -1:]])

    return inverse_data

def preprocess_params_fit(data):
    preprocessing_params = {}
    preprocessing_params["min"] = np.min(data[:, 1:-1], axis=0)
    preprocessing_params["max"] = np.max(data[:, 1:-1], axis=0)

    preprocessed_data_x, mask = logit_transform(data[:, 1:-1], preprocessing_params["min"], preprocessing_params["max"])
    preprocessed_data = np.hstack([data[:, 0:1], preprocessed_data_x, data[:, -1:]])[mask]

    preprocessing_params["mean"] = np.mean(preprocessed_data[:, 1:-1], axis=0)
    preprocessing_params["std"] = np.std(preprocessed_data[:, 1:-1], axis=0)

    return preprocessing_params

def preprocess_params_transform(data, params):
    preprocessed_data_x, mask = logit_transform(data[:, 1:-1],
                                                 params["min"], params["max"])
    preprocessed_data = np.hstack([data[:, 0:1], 
                                   preprocessed_data_x, data[:, -1:]])[mask]
    preprocessed_data[:, 1:-1] = standardize(preprocessed_data[:, 1:-1], 
                                             params["mean"], params["std"])



    return preprocessed_data

def inverse_transform_all(data, params):
    inverse_data = inverse_standardize(data[:, :-1], params["mean"], params["std"])
    inverse_data = inverse_logit_transform(inverse_data, params["min"], params["max"])
    inverse_data = torch.hstack([inverse_data, data[:, -1:]])

    return inverse_data


def generate_transformed_samples_RQS(model, data, preprocessing_params, device, batch=2048):

    for key in preprocessing_params.keys():
        preprocessing_params[key] = preprocessing_params[key].to(device)


    x_samples = model.sample(1, data[:,0].reshape(-1,1), batch_size=batch)
    x_samples = x_samples.reshape(-1,4)
    x_samples = inverse_standardize(x_samples, preprocessing_params["mean"], preprocessing_params["std"])
    x_samples = inverse_logit_transform(x_samples, preprocessing_params["min"], preprocessing_params["max"])
    x_samples = torch.hstack((data[:,0].reshape(-1,1), x_samples))
    # x_samples = np.vstack((x_samples_train, x_samples_val))
    x_samples = x_samples.cpu().detach().numpy()
    x_samples = x_samples[~np.isnan(x_samples).any(axis=1)]

    print('x_samples shape', x_samples.shape)
    print('all_data shape', data.shape)
   # print()
   # phyiscal_samples = torch.hstack([data[:, 0:1], phyiscal_samples])

    return x_samples


def generate_transformed_samples(model, data, preprocessing_params, device, number=10000):

    mass_test = data[:,0].reshape(-1,1).type(torch.FloatTensor).to(device)
    with torch.no_grad():
       # x_samples = model.sample(len(mass_test), cond_inputs=mass_test)
        x_samples = model.sample(len(mass_test), mass_test)
    
    for key in preprocessing_params.keys():
        preprocessing_params[key] = preprocessing_params[key].to(device)

      #  x_samples = model.sample(10000, mass_test[0:10000])
    phyiscal_samples = inverse_standardize(x_samples, preprocessing_params["mean"], preprocessing_params["std"])
    phyiscal_samples = inverse_logit_transform(phyiscal_samples, preprocessing_params["min"], preprocessing_params["max"])
   # print(x_samples.shape)
   # print(phyiscal_samples.shape)
   # print()
   # phyiscal_samples = torch.hstack([data[:, 0:1], phyiscal_samples])

    return phyiscal_samples

def generate_transformed_samples_uncond(model, data, preprocessing_params, device, number=10000):

    mass_test = data[:,0].reshape(-1,1).type(torch.FloatTensor).to(device)
    with torch.no_grad():
       # x_samples = model.sample(len(mass_test), cond_inputs=mass_test)
        x_samples = model.sample(len(mass_test))
    
    for key in preprocessing_params.keys():
        preprocessing_params[key] = preprocessing_params[key].to(device)

      #  x_samples = model.sample(10000, mass_test[0:10000])
    phyiscal_samples = inverse_standardize(x_samples, preprocessing_params["mean"], preprocessing_params["std"])
    phyiscal_samples = inverse_logit_transform(phyiscal_samples, preprocessing_params["min"], preprocessing_params["max"])
   # print(x_samples.shape)
   # print(phyiscal_samples.shape)
   # print()
   # phyiscal_samples = torch.hstack([data[:, 0:1], phyiscal_samples])

    return phyiscal_samples




def SIC_fpr(label, score, fpr_target):
    fpr, tpr, thresholds = roc_curve(label, score)

    index = np.argmin(np.abs(fpr - fpr_target))

    sic = tpr/np.sqrt(fpr)

    sic_target = sic[index]

    return sic_target



def SIC(label, score):
    fpr, tpr, thresholds = roc_curve(label, score)
    auc = roc_auc_score(label, score)

    tpr = tpr[fpr>0]
    fpr = fpr[fpr>0]

    sic = tpr/np.sqrt(fpr)

    return sic, tpr, auc


def roc_interp(label, score, tpr_interp):
    fpr, tpr, _ = roc_curve(label, score)


    fpr_interp = interp1d(tpr, fpr)(tpr_interp)

   # indices = np.argwhere(fpr_interp>1/312858/cut**2).flatten()

   # tpr_cut = np.zeros((len(tpr_interp,)))
   # fpr_cut = np.zeros((len(tpr_interp,)))

   # tpr_cut[indices] = tpr_interp[indices].flatten()
   # fpr_cut[indices] = fpr_interp[indices].flatten()

  #  tpr_cut = tpr_interp[indices].flatten()
  #  fpr_cut = fpr_interp[indices].flatten()
    
    return tpr_interp, fpr_interp

def ensembled_SIC(tpr_list, fpr_list, cut=0.20):

    fpr_list = np.array(fpr_list)
    tpr_list = np.array(tpr_list)
   # eB_list = np.namnp.array(1/fpr_list)
    cuts = (fpr_list > 1/(312858*cut**2)).flatten()*1

    fpr_list_mx = ma.masked_array(fpr_list, mask=1-cuts, fill_value=np.nan).filled()
    tpr_list_mx = ma.masked_array(tpr_list, mask=1-cuts, fill_value=np.nan).filled() 

    fpr_median = np.nanmedian(fpr_list_mx, axis=0)
    fpr_max = np.nanpercentile(fpr_list_mx, 84, axis=0)
    fpr_min = np.nanpercentile(fpr_list_mx, 16, axis=0)


    tpr_median = np.nanmedian(tpr_list_mx, axis=0)
    cuts = np.argwhere(fpr_median > 1/(312858*cut**2)).flatten()

    tpr_median = tpr_median[cuts]
    fpr_median = fpr_median[cuts]
    fpr_max = fpr_max[cuts]
    fpr_min = fpr_min[cuts]

    sic_median = np.nan_to_num(tpr_median/np.sqrt(fpr_median), posinf=0)
    sic_max = np.nan_to_num(tpr_median/np.sqrt(fpr_max), posinf=0)
    sic_min = np.nan_to_num(tpr_median/np.sqrt(fpr_min), posinf=0)
    tpr_median = np.nan_to_num(tpr_median, posinf=0)

    return sic_median, sic_max, sic_min, tpr_median

    




def SIC_cut(label, score, cut=0.2):
    fpr, tpr, thresholds = roc_curve(label, score)
    auc = roc_auc_score(label, score)

    tpr = tpr[fpr>0]
    fpr = fpr[fpr>0]
    indices = np.argwhere(fpr>1/312858/cut**2)

    tpr_cut = tpr[indices]
    fpr_cut = fpr[indices]

    sic_cut = tpr_cut/np.sqrt(fpr_cut)
    max_sic = np.nanmax(np.nan_to_num(sic_cut, posinf=0), initial=0)

    return sic_cut, tpr_cut, max_sic


def max_SIC(label, score, cut=0.2):
    fpr, tpr, _ = roc_curve(label, score)

    tpr = tpr[fpr>0]
    fpr = fpr[fpr>0]

    sic = tpr/np.sqrt(fpr)

    return np.nanmax(np.nan_to_num(sic, posinf=0),where= fpr>1/312858/cut**2, initial=0)

def prior(pcx):
    return pcx.mean()



def p_theta_given_x(data, mu1, sigma1, mu2, sigma2, prior):
    ''' 
    calculate responsibility p(c|x) for each datapoint. where c is the cluster.
                   p(x|c) p(c)              p(x|c) p(c)                  p(x|c) p(c) [prior]
    p(c|x) =  --------------------- = ------------------------- =  -------------------------------
                      p(x)                  sum(c) p(x,c)             p(x|c) p(c) + p(x|c') p(c')  
    '''
    numerator = norm.pdf(data, mu1, np.sqrt(sigma1)) * prior
    denominator = numerator + norm.pdf(data, mu2, np.sqrt(sigma2) ) * (1-prior)
    return numerator / denominator

 

# write EM algorithm with weights
def EM_2_gaussian(data, n_components, max_iter, tol, init_params='kmeans',
                  mu_back=0.5, sigma_back=0.1, p_back = 0.5,
                   mu_sig=0.5, sigma_sig=0.1 ):

    """
    EM algorithm for 2 gaussian mixture model

    Parameters
    ----------
    data : array-like, shape (n_samples,)
        List of n_features-dimensional data points.  Each row
        corresponds to a single data point.
    n_components : int
        Number of gaussian components.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence threshold.
    init_params : {'kmeans', 'random'}
        Method for initialization, defaults to 'kmeans'.

    Returns
    -------
    mu : array-like, shape (n_components,)
        List of means for each gaussian component.
    sigma : array-like, shape (n_components,)
        List of standard deviations for each gaussian component.
    p : array-like, shape (n_components,)
        List of weights for each gaussian component.
    likelihood_arr : array-like, shape (n_iter,)
        List of log likelihoods per iteration.


    """

    n = len(data)
    if init_params == 'kmeans':
        kmeans = KMeans(n_clusters=n_components, random_state=0,
                        tol=10-12).fit(data.reshape(-1, 1))
        mu = kmeans.cluster_centers_.flatten()
        sigma = np.random.uniform(0, 1, n_components)
        pa = 1 / n_components

    elif init_params == 'random':
        mu = np.random.uniform(min(data), max(data), n_components)
        sigma = np.random.uniform(0, 5, n_components)
        #sigma = np.random.uniform(0, 1, n_components)
        pa = 1 / n_components

    elif init_params == 'fixed':

        if n_components == 2:
            mu_sig = np.random.uniform(min(data), max(data), n_components-1)
            sigma_sig = np.random.uniform(0, 5, n_components-1)
            mu = np.array([mu_back, mu_sig[0]])
            sigma = np.array([sigma_back, sigma_sig[0]])
            pa = 1 / n_components
        elif n_components ==1:
            mu = np.array([mu_back])
            sigma = np.array([sigma_back])
            pa = 1
        else:
            print('n_components not supported')

    elif init_params == 'all fixed':

        if n_components == 2:
           # mu_sig = np.random.uniform(mu_back, max(data), n_components-1)
           # sigma_sig = np.random.uniform(0, 5, n_components-1)

            mu = np.array([mu_back, mu_sig])
            sigma = np.array([sigma_back, sigma_sig])
            pa = p_back
        elif n_components ==1:
            mu = np.array([mu_back])
            sigma = np.array([sigma_back])
            pa = 1
        else:
            print('n_components not supported')

    elif init_params == 'low_weights':
        mu = np.random.uniform(min(data), max(data), n_components)
        sigma = np.random.uniform(0, 1, n_components)
        pa = 0.1
    else:
        print('init_params not supported')


    if n_components == 2:
        p = [pa, 1-pa]
    elif n_components == 1:
        p = [pa]
    else:
        print('n_components not supported')

   # mu = np.random.uniform(min(data), max(data), n_components)

    

    likelihood_arr = []
    mu_arr = []
    sigma_arr = []
    w_arr = []


    for i in range(max_iter):

        '''
        Expectation step: calculate "responsibility" of each cluster to each datapoints.
        i.e which 
        p_c1 = P(c1|x) = ...
        p_c2 = P(c2|x) = ...
        '''
        # Compute the likelihood of the data given the current parameters

       # p_c1 = norm.pdf(data, mu[0], np.sqrt(sigma[0])) * p[0]
        #p_c2 = norm.pdf(data, mu[1], np.sqrt(sigma[1])) * p[1]

        p_c = 0
        for k in range(n_components):
            p_c += norm.pdf(data, mu[k], np.sqrt(sigma[k])) * p[k]


        #log_likelihood = np.log(p_c1 + p_c2).sum()
        log_likelihood = np.log(p_c).sum()

        if n_components == 2:
            a = p_theta_given_x( data, mu[0], sigma[0], mu[1], sigma[1], p[0] )
            b = 1 - a

            pa = prior(a)
            pb = 1 - pa

            p = [pa, pb]

        elif n_components == 1:
            a = 1.
        else:
            print('n_components not supported')


        '''
        adjust mu and sigma 
        '''
        if n_components == 2:
            if init_params == 'fixed':          
                mu[1] = np.multiply(b, data).sum() / b.sum()
                sigma[1] = np.multiply(b, (data - mu[1])**2).sum() / b.sum()

                mu[0] = mu_back
                sigma[0] = sigma_back

            elif init_params == 'all fixed':
                mu[1] = np.multiply(b, data).sum() / b.sum()
                sigma[1] = np.multiply(b, (data - mu[1])**2).sum() / b.sum()

                mu[0] = mu_back
                sigma[0] = sigma_back


            else:
                mu[0] = np.multiply(a, data).sum() / a.sum()
                sigma[0] = np.multiply(a, (data - mu[0])**2).sum() / a.sum() 
                
                mu[1] = np.multiply(b, data).sum() / b.sum()
                sigma[1] = np.multiply(b, (data - mu[1])**2).sum() / b.sum()
        elif n_components == 1:
            mu[0] = np.mean(data)
            sigma[0] = np.mean( (data - mu[0])**2)
        else:
            print('n_components not supported')
            


        likelihood_arr.append(log_likelihood)
        mu_arr.append(mu)
        sigma_arr.append(sigma)
        w_arr.append(p)


        # check convergence
        if i > 0:
            if np.abs(log_likelihood - log_likelihood_old) < tol:
                print('Converged after {} iterations.'.format(i))

                return True, np.array(mu_arr), \
np.array(sigma_arr),np.array(w_arr), np.array(likelihood_arr)
                break
            else :
                log_likelihood_old = log_likelihood

                if i == max_iter - 1:
                    print('Did not converge after {} iterations.'.format(i))

                    return False, np.array(mu_arr), \
np.array(sigma_arr),np.array(w_arr), np.array(likelihood_arr) 
                
        else:
            log_likelihood_old = log_likelihood

           # return False, np.array(mu_arr), \
#np.array(sigma_arr),np.array(w_arr), np.array(likelihood_arr)

    
    


     
    
    


def p_data(data,mu,sigma,p,dim=1):

    """ 2 gaussian mixture model"""
    p_c1 = 1.0
    p_c2 = 1.0


    for d in range(dim):
        p_c1 *= norm.pdf(data[:,d], mu[0], sigma[0])

    for d in range(dim):
        p_c2 *= norm.pdf(data[:,d], mu[1], sigma[1])

    return p[0]*(p_c1) + p[1]*(p_c2)

def p_back(data,mu,sigma,dim=1):

    """ background gaussian """
    pc = 1.0

    for d in range(dim):
        pc *= norm.pdf(data[:,d], mu,sigma)

    return pc

def inverse_sigmoid(x):
    return np.log(x/(1-x))

def capped_sigmoid(x, a):
    x = torch.tensor(x)
    return a/(1+torch.exp(-x))

def scaled_sigmoid(x, a):
    x = torch.tensor(x)
    return 1/(1+torch.exp(- a * x))



def jet_centering(data):

    E = data[...,0]
    px = data[...,1]
    py = data[...,2]
    pz = data[...,3]

    pt = np.sqrt(px**2 + py**2)
    eta = 0.5 * np.log((E + pz)/(E - pz))
    phi = np.arctan2(py, px)
    m = np.sqrt(E**2 - px**2 - py**2 - pz**2)

    jet_ptetaphim = np.stack([pt, eta, phi, m], axis=-1)


    total_E = E.sum(axis=1)
    total_px = px.sum(axis=1)
    total_py = py.sum(axis=1)
    total_pz = pz.sum(axis=1)

    jet_vector = np.stack([total_E, total_px, total_py, total_pz], axis=-1)

    total_eta = 0.5 * np.log((total_E + total_pz)/(total_E - total_pz))
    total_phi = np.arctan2(total_py, total_px)
    total_m = np.sqrt(total_E**2 - total_px**2 - total_py**2 - total_pz**2)
    total_pt = np.sqrt(total_px**2 + total_py**2)

    print(total_pt.shape)

    # centering
    jet_ptetaphim[...,0] = jet_ptetaphim[...,0]/total_pt
    jet_ptetaphim[...,1] = jet_ptetaphim[...,1] - total_eta
    jet_ptetaphim[...,2] = jet_ptetaphim[...,2] - total_phi
    jet_ptetaphim[...,3] = jet_ptetaphim[...,3]/total_m


    # convert back to eta pt phi m to E px py pz
    px = jet_ptetaphim[...,0] * np.cos(jet_ptetaphim[...,2])
    py = jet_ptetaphim[...,0] * np.sin(jet_ptetaphim[...,2])

    mt = np.sqrt(jet_ptetaphim[...,0]**2 + jet_ptetaphim[...,3]**2)
    pz = mt * np.sinh(jet_ptetaphim[...,1])
    E = mt * np.cosh(jet_ptetaphim[...,1])

    jet_constituent_vector = np.stack([E, px, py, pz], axis=-1)

    



