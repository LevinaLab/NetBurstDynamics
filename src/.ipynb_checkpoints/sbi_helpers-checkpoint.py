import numpy as np
# from src.helpers import collectBursts
import torch
na = np.array

from sbi.types import Shape
from src.dynamics_Model2 import FPmultiStart,findOsc 

from numba import jit 
from scipy.integrate import solve_ivp as solve_ivp

import matplotlib.pyplot as plt
from sbi.inference import SNPE
from sbi.utils import RestrictionEstimator
import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import numpy as np

from tqdm import tqdm
na =np.array
import os
import numpy as np
from scipy.io import loadmat as loadmat
na =np.array
from sbi.utils.get_nn_models import posterior_nn
from functools import partial
from sbi.inference import SNPE, SNRE, prepare_for_sbi, simulate_for_sbi
from sbi.types import Shape
from torch import Tensor
from multiprocessing import Pool
from sbi.types import Shape
# from sbi.samplers.mcmc import slice_np_parallized
from scipy.stats import gaussian_kde as kde
from scipy.optimize import minimize


def fit_nd_gaussian(samples,bw=1):
    """
    Fit an n-dimensional Gaussian to a set of samples and find the minimum of the distribution.

    Args:
    samples: numpy array of shape (n_samples, n_features)

    Returns:
    A tuple of the form (mu, sigma, min_val), where
    - mu: numpy array of shape (n_features,) representing the mean of the Gaussian distribution.
    - sigma: numpy array of shape (n_features, n_features) representing the covariance matrix of the Gaussian distribution.
    - min_val: the minimum value of the Gaussian distribution.
    """

    # Estimate the mean and covariance matrix of the Gaussian distribution from the samples
    mu = np.mean(samples, axis=0)
    sigma = np.cov(samples, rowvar=False)

    # Define the negative log-likelihood function to be minimized
    kde_object = kde(samples.T,bw_method=bw)

    def neg_log_likelihood(x):
        return -kde_object.logpdf(x)#multivariate_normal.logpdf(x, mean=mu, cov=sigma)

    # Find the minimum of the negative log-likelihood function
    res = minimize(neg_log_likelihood, mu, method='Nelder-Mead')

    # Compute the minimum value of the Gaussian distribution
    min_val = kde_object.evaluate(res.x)#multivariate_normal.pdf(res.x, mean=mu, cov=sigma)

    return (res.x, sigma, min_val)


def state_checker(param_set,keys=['a','b','theta','tau_w','J','sigma']):
    """State checker for the minimal bursting system

    Args:
        param_set (np list): a,b,J,theta,

    Returns:
        [type]: [description]
    """

    param_set= param_set.numpy()
    params = expand_params(param_set,keys=keys)
    # print(params)
    bis = False
    osc= False
    osc= findOsc(params,tmax=4000000,dt=.05)
    if osc==False:
        fp = FPmultiStart(params)
        bis= len(fp)>1
    return bis,osc

def reduce_params(params,
                keys =['a','b','theta','tau_w','J','sigma']):
    """take a list of parameters and return a list
    """
    params_set = [params[k] for k in keys]
    # params_set[keys.index('tau_w')]=np.exp(params_set[keys.index('tau_w')])
    # params_set[keys.index('tau_w')]/=1000 #conver tau_w to s
    return params_set 

def expand_params(params_set,T=500000,keys=['a','b','theta','tau_w','J','sigma'],
                  dt=0.05, torch_ = False, params = None):
    """expand a list of parameters into a dictionary for simulator

    Args:
        param_set (list ot torch tensor): [
        
    """
    params_set_ = []
    if torch_:
        for i,p in enumerate(params_set):
            params_set_.append(p.numpy())
    else:
        params_set_ = params_set

    # [isinstance(p,torch.Tesnor) for p in params_set]
    # mu  =0.
    # if torch_:
    #     mu = torch.tensor(0.0)
    #     tau_w = torch.exp(params_set[3])
    # else:

    #make sure that tau_w is in exp
    w_ind = ['tau_w' == k for k in keys]
    tau_w = np.exp(na(params_set_)[w_ind][0])
    #deafault parameters
    if default_params is None:
        params ={
            'a':5.,#:params_set[0],
            'b':np.nan,#params_set[1],
            'theta':np.nan,#params_set[2],
            'tau':1.,
            'tau_w':np.nan,#tau_w,#*1000,
            'mu':0.0,
            'J':9.,#j,
            'T':T,
            'dt':dt,
            'x0':0.,
            'w0':0.,
            'D':np.nan,#(params_set[5]**2)/(2),
            'sigma':np.nan,#params_set[5],
        }  
    #update
    for i,key in enumerate(keys):
        params[key] = params_set_[i]
    #make sure to set tau_w correctly (TODO: fix this to take fewer lines)
    params['tau_w'] = tau_w
    # print(params)

    return params


def simulatorDurs(params_set,T=900000, #might need to go back to 600000
              torch_=True, minEvents=50,
              original_v=[np.inf],
              keys=None,
              default_params =None,
             core_seed=2234567888):
    """Simualtor that collect N bursts
    using duration as summary
    """
    params = expand_params(params_set,torch_=torch_,keys=keys,params=default_params)
    # try: 
    ibis,durs = collectBursts(params,n_events = minEvents,
                    T = T,main_seed=core_seed,
                    return_durs = True,torch_=torch_)
    ibis = na(ibis)/1000 # convers ms in s
    durs = na(durs)/1000
    mibi = np.mean(ibis)
    cv= np.std(ibis)/mibi
    mdurs = np.mean(durs)
    cv_durs =np.std(durs)/mdurs
    if np.isfinite(original_v[0]):
        mibi,cv = mibi/original_v[0],cv/original_v[1]
        mdurs,cv_durs = mdurs/original_v[2],cv_durs/original_v[3]
    return torch.as_tensor([mibi,cv,mdurs,cv_durs])

def simulatorDursCorr(params_set,T=900000, #might need to go back to 600000
              torch_=True, minEvents=2,
              original_v=[np.inf],
              keys=None,core_seed=2234567888):
    """Simualtor that collect N bursts
    using duration as summary
    """
    params = expand_params(params_set,torch_=torch_,keys=keys)
    # try: 
    ibis,durs = collectBursts(params,n_events = minEvents,
                    T = T,main_seed=core_seed,max_iter=0,
                    return_durs = True,torch_=torch_)
    ibis = na(ibis)/1000 # convers ms in s
    durs = na(durs)/1000
    mibi = np.mean(ibis)
    cv= np.std(ibis)/mibi
    mdurs = np.mean(durs)
    cv_durs =np.std(durs)/mdurs 
    durs = np.hstack(durs)
    dur_ibi_c = np.corrcoef(durs[1:],ibis)[0,1]
    if np.isfinite(original_v[0]):
        mibi,cv = mibi/original_v[0],cv/original_v[1]
        mdurs,cv_durs = mdurs/original_v[2],cv_durs/original_v[3]
    return torch.as_tensor([mibi,cv,mdurs,cv_durs,dur_ibi_c])

def simulatorNB(params_set,T=500000,
              torch_=True, minEvents=30,
              original_v=[np.inf],core_seed=123456789):
    """Simualtor that collect N bursts"""

    params = expand_params(params_set,torch_=torch_)
    # try: 
    ibis = collectBursts(params,n_events = minEvents,T = T,main_seed=core_seed,torch_=torch_)
    ibis = na(ibis)/1000 # convers ms in s
    mibi = np.mean(ibis)
    cv= np.std(ibis)/mibi
    #except:
        # print('a')
        # mibi = np.nan
        # cv = np.nan
    if np.isfinite(original_v[0]):
        # normalize by the values you'd want to fit
        mibi,cv = mibi/original_v[0],cv/original_v[1]
    return torch.as_tensor([mibi,cv])

def simulator(params_set,plotting=False,T=1000000,return_data=False,
              torch_=True, original_v=[np.inf]):#100000
             #I except a comment  
    dt = 0.05
    mu  =0.
    if torch_:
        mu = torch.tensor(0.0)
    params ={
        'a':params_set[0],
        'b':params_set[1],#1.9,#10.1,#
        'theta':params_set[2],
        'tau':1.,
        'tau_w':params_set[3]*1000,
        'mu':mu,
        'J':params_set[4],
        'T':T,
        'dt':dt,
        'x0':0.,
        'w0':0.,
        'D':(params_set[5]**2)/(2),#0.0045125,#D,
        'absXw':False,
        'cores':30,
        'sigma':params_set[5],#0.095,
    }
    bursty = 0
    
    A = Model2(params,torch=torch_)
    tI,xI_,wI_   = A.simulate(T)

    if np.median(xI_)<0.5:#np.std(xI_)>0.15:
        try:
            if plotting or return_data:
                feat= A.getFeatures(xI_[:].copy(),wI_[:].copy(),tI)
            else:
                feat= A.getFeatures(xI_[:],wI_[:],tI)
            if len(feat['bursts'])>2:    
                ibis = (feat['ibis'])/1000
                mdur = np.mean(feat['durs'])
                mibi = np.mean(ibis)
                cv = np.std(ibis)/mibi
                cv_dur =np.std(feat['durs'])/mdur
                bursty = 1
            else:
                mibi=np.nan
                cv=np.nan
                mdur=np.nan
                ibis=[np.nan]
                cv_dur =np.nan
                
        except:
            mibi=np.nan
            cv=np.nan
            mdur=np.nan
            ibis=[np.nan]
            cv_dur =np.nan
    else :
        mibi=np.nan
        cv=np.nan
        mdur=np.nan
        ibis = [np.nan]
        # cv_dur =np.nan

    if return_data:
        return A,ibis,tI,xI_,wI_,mibi,cv,mdur

    if np.isfinite(original_v[0]):
        # normalize by the values you'd want to fit
        mibi,cv = mibi/original_v[0],cv/original_v[1]#,mdur/original_v[2],cv_dur/original_v[3] 

    return torch.as_tensor([mibi,cv])#[a,w,c]@mibi
