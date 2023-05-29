import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.cluster import KMeans
from scipy.stats import norm

# SIC curve

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

    
    


     
    
    


def p_data(data,mu,sigma,p):

    """ 2 gaussian mixture model"""

    p_c1 = norm.pdf(data, mu[0], np.sqrt(sigma[0])) * p[0]
    p_c2 = norm.pdf(data, mu[1], np.sqrt(sigma[1])) * p[1]

    return p_c1 + p_c2

def p_back(data,mu,sigma):

    """ background gaussian """
    pc = norm.pdf(data, mu, np.sqrt(sigma))

    return pc


