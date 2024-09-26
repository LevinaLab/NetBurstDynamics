
from pickle import FALSE
from src.dynamics import StochSim_o
# from scipy import stats
# import sys
import numpy as np
na = np.array
# from scipy import stats
# import sys
import numpy as np
# from scipy.signal import find_peaks
from numba import jit
import sympy as sp
na = np.array
from scipy.stats import kurtosis,skew
import numpy as np
import torch
na = np.array

# from sbi.types import Shape
from src.dynamics import FPmultiStart,findOsc 

from numba import jit 
from scipy.integrate import solve_ivp as solve_ivp

import matplotlib.pyplot as plt
# from sbi.inference import SNPE
# from sbi.utils import RestrictionEstimator
import torch
from sbi import utils as utils
from sbi import analysis as analysis
# from sbi.inference.base import infer
import numpy as np

# from tqdm import tqdm
na =np.array
# import os
import numpy as np
from scipy.io import loadmat as loadmat
na =np.array
# from sbi.utils.get_nn_models import posterior_nn
# from functools import partial
# from sbi.inference import SNPE, SNRE, prepare_for_sbi, simulate_for_sbi
# from sbi.types import Shape
# from torch import Tensor
# from multiprocessing import Pool
# from sbi.types import Shape
# from sbi.samplers.mcmc import slice_np_parallized


import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
def fit_nd_gaussian(samples):
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
    def neg_log_likelihood(x):
        return -multivariate_normal.logpdf(x, mean=mu, cov=sigma)

    # Find the minimum of the negative log-likelihood function
    res = minimize(neg_log_likelihood, mu, method='Nelder-Mead')

    # Compute the minimum value of the Gaussian distribution
    min_val = multivariate_normal.pdf(res.x, mean=mu, cov=sigma)

    return (res.x, sigma, min_val)


def plot_bifurcations(a=5,add_colors=False):
    """plot bifurcation diagram for a given a"""
    
    bif = np.loadtxt('../../results/bifurcations/diagram_a=%s.txt'%a)
    bf = np.arange(10,400000,1000)*0.00002
    # ax_dict['A'].plot(bif[:,1],bif[:,0],'.')
    plt.ylabel('Excitability ($\\theta$)',fontsize = 26)
    plt.xlabel('Adaptation strength (b)',fontsize = 24)
    bs = np.arange(0,5,0.1)
    y1 = 8.1+bs*-8.82
    y2 = .96+bs*-.208
    plt.plot(bs,-y1,'k',linewidth=4)
    plt.plot(bs,-y2,'k',linewidth=4)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # plt.fill_between(bs, y1, y2,color='C0',alpha=0.1)
    # lower_triag = na([[0.81,0.79],[0.9,0.],[4.6,0.]])
    if add_colors:
        plt.fill_between(bs, -y1, -y2,color='darkgray')
        lower_triag = na([[0.88,-0.75],[0.9,0.],[4.6,0.]])
        plt.fill_between(lower_triag[:,0],lower_triag[:,1],
                y2=na([0.88,-0.75,0,]),color='r',alpha=0.5)
    return None

def bimodality_index(x):
    """computes the sample bomodality index based on 
    https://en.wikipedia.org/wiki/Multimodal_distribution """
    gamma = skew(x)
    kappa = kurtosis(x)
    n = len(x)
    return ((gamma**2)+1)/(kappa + (3*(n-1)**2) / ((n-2)*(n-3)))


def getNull(params):
    """get nullclines using sympy
    Args:
        params ([list]): dict of model paramters
    Returns:
        null ([list]): x, w-null, x-null - plotting ready
    """
    x = sp.symbols('x')
    def sym_logit(x,a,theta):
        return (1/a)*(a*theta + sp.log(x/(1-x)))
    b = params['b']
    mu = params['mu']
    J = params['J']
    a = params['a']
    theta = params['theta']
    w01 = x*b
    log_y= sym_logit((x-mu)/J, a, theta)
    w02 = -log_y+x
    return [x,w01,w02]
    


def HMMburst_detection(xI,filt=False,window=801,order=5,minDist=2000,dt=0.05):
    """detect bursts with hidden markov model
    filt(bool): filter the xI or not
    minDist(ms): min distance between two bursts
    """
    startprob = np.array([[0.99],[0.01]])
    transmat = np.array([[0.99, 0.01], [0.02,0.98]
                        ])
    means =np.array([[0],
                     [2.]])
    covars = np.tile(np.identity(1)*0.1, (2, 1, 1))
    model = hmm.GaussianHMM(n_components=2, covariance_type="full")
    if filt:
        xI = savgol_filter(xI,window,order)
    model.fit(na([xI]).T)
    #get the transition
    Z = model.predict(na([xI]).T)
    # make sure that ones are bursts
    #Z = Z==1#make bool
    if np.mean(xI[Z==1])<np.mean(xI[Z==0]):
        Z = Z==0
        Z = np.array(Z,dtype=float)
        
    #Filter artefacts
#     mask_ = (xI[Z==1]>=0.3)
#     mask
    
#     print(len(mask_),len(Z))
    Z[xI<=0.3] = 0.

    onT = np.where(np.diff(Z)==1)[0]
    offT = np.where(np.diff(Z)==-1)[0]
    
    onT = onT[np.hstack([True,np.diff(onT*dt)>minDist])]
    offT = offT[::-1][np.hstack([True,np.diff(offT[::-1]*dt)>minDist])]
    
    #check if bursts  
#     print(np.mean(xI[Z==1]))
    return Z,onT,offT[::-1]

def smallBurstDet(xI,tI,fp_stat,minDist=100):
    """simple fast burst detection based on thresholding around
    the fixed points"""
    #find ON time
#     mask = xI>=fp_stat[0][2]
#     T_onMask = np.diff(tI[mask])>minDist
#     T_on = tI[mask][np.hstack([True,T_onMask])]
#     #find OFF time
#     T_offMask = -np.diff(tI[mask][::-1])>minDist
#     T_off = tI[mask][np.hstack([False,T_offMask])]

    mask = xI>=fp_stat[0][2]-0.4
    mask  = na(mask,dtype='int')
    onT = np.where(np.diff(mask)==1)[0]
    offT = np.where(np.diff(mask)==-1)[0]

    return mask,onT,offT


def wTransitions(bursts,xI,wI,dt):
    """detect the w_down->up and w_up->down
    based on the detected bursts"""
    w_dus, w_uds = [],[]
    for burst in bursts[1:]:
        b = int(burst[0]/dt)
        wI_d = wI[(b-1000):(b+1000)]
        xI_d= xI[(b-1000):(b+1000)]
        wdu_ = np.argmax(np.diff(xI_d)) #max derivative corresponds to the jump 
        w_dus.append(wI_d[wdu_])

        b = int(burst[1]/dt)
        xI_u= xI[(b-1000):(b+1000)]
        wI_u = wI[(b-1000):(b+1000)]
        wud_ = np.argmin(np.diff(xI_u)) #max derivative corresponds to the jump 
        w_uds.append(wI_u[wud_]) 
    return w_dus, w_uds

@jit(nopython=True)
def find_burstlets(spikes,r_spikes,isi_pop,
              maxISIstart=4.5,
              maxISIb=4.5,
              minSburst=100):
    """ 
    Helper to find burstlets
    Args:
        spikes (arr): spike times
        r_spikes(arr): rounded spike times
        isi_pop(arr):isi
    Returns:
            burst_ (list of tuples): Burst start, burst end
    """
    b_spikes = None
    burst_ = []
    sync_b = False
    b_start = 0
    b_size = 0
    for i,s in enumerate(spikes[:-1]):
        
        if isi_pop[i]<maxISIstart and b_start==0:

            b_size = 0
            b_start += 1
            b_spikes=s
            
        elif isi_pop[i]<=maxISIb and b_start>0: #start if two conseq init isi
            b_start+=1
            b_size+=1
            
        elif b_start>=minSburst:
            burst_.append((b_spikes,s))
            b_spikes =None
            b_size = 0
            sync_b = False
            b_start = 0
            
        else:
            b_spike =None
            b_size = 0
            sync_b = False
            b_start = 0   
    return burst_


def MI_bursts(st,
              maxISIstart=4.5,
              maxISIb=4.5,
              minBdur=40,
              minIBI=40,
              minSburst=50):
    """Min Interval method [1,2] for burst detections
    Optimized version from 03.03.20
    OV

    Args:
            [1] spike times (list) (signle neuron or popultation) (ms)
            [2] (float) max ISI at start of burst (ms)
            [3] (float) max ISI in burst (ms)
            [4] (float) min burst duration (ms)
            [5] (float) min inter-burst interval (ms)
            [6] (float) min number of spike in a burst 
    Returns:
            burst (list of tuples): burst start, burst end
    [1] Nex Technologies.NeuroExplorer Manual.  Nex Technologies,2014
    [2] Cotterill, E., and Eglen, S.J. (2018). Burst detection methods. ArXiv:1802.01287.        """
    spikes = np.sort(st)
    r_spikes = np.round_(spikes,-1)
    isi_pop = np.diff(spikes)
    burst_ =find_burstlets(spikes,r_spikes,isi_pop,maxISIstart,maxISIb,minSburst)
#     print(burst_)
    bursts = []
    if burst_:
        bursts.append(burst_[0])
        for i,b in enumerate(burst_[1:]):
            if b[1]-b[0]>=minBdur and b[0]-bursts[-1][1] >= minIBI:
                bursts.append(b)
            elif b[0]-bursts[-1][1]<= minIBI:
                bursts[-1] = (bursts[-1][0],b[1])
        
    return bursts

# @jit(nopython=True)
def x_to_spikes(x,dt,scale = 10):
    """ convert continuous signal 
    to times of discrete events
    """
    x[x<0] = 0.
    spks = na(scale*(x/x.max()),dtype =int)
    st= np.where(spks)[0]
#     st_ = np.repeat(st,spks[st])
    return st*dt

def spike_burst_det(x,dt,burst_detection_params):
    """detecting skipes using MI methods 
    """
    st_ = x_to_spikes(x.copy(),dt= dt,scale=burst_detection_params['scale'])
    bursts = MI_bursts(st_,
        maxISIstart=burst_detection_params['maxISIstart'],#.5,
        maxISIb=burst_detection_params['maxISIb'],
        minBdur=burst_detection_params['minBdur'],#ms
        minIBI=burst_detection_params['minIBI'],#ms
        minSburst=burst_detection_params['minSburst'])#slikes
        
    return bursts




@jit(nopython=True)
def clean_bursts(bursts):
    """filter bursts below 1std of the baseline"""
    x_sum = 0
    x_sq = 0
    n= 0
    for i,burst in enumerate(bursts[1:]):
        mask = (t>bursts[i][0])*(t<burst[0])
        x_sum += np.sum(x[mask])
        n +=len(x[mask]) 
        x_sq += np.sum(x[mask]**2)
    x_mean =  x_sum/n
    std = np.sqrt((x_sq/n) - x_mean**2)
    valid_bursts = []
    for i,burst in enumerate(bursts):
        mask = (t>burst[0])*(t<burst[1])
        x_b_mean= np.mean(x[mask])
        if x_b_mean>x_mean+(std):
            valid_bursts.append(burst)
    return valid_bursts


def simulator_summary(params_set,simfunc=StochSim_o):#StochSimI_mult
    
    burst_detection_params={'maxISIstart':10.,
                    'maxISIb':10.,
                    'minBdur':20.,
                    'minIBI':20.,
                    'minSburst':5.,
                    'scale':10.
                    }
    #heating up
    params = expand_params(params_set,T=10000,torch_=True,keys=['b','theta','tau_w','sigma'])
    t,x,w = simfunc(params,torch=False)
    params['x0']=x[-1]
    params['w0']=w[-1]
    params['T']=2000000
    t,x,w = simfunc(params,torch=False)
    mibi,mdur,cv_ibis,cv_durs= np.nan,np.nan,np.nan,np.nan
    bi = bimodality_index(x)
    if np.any(bi>0.55) and np.any(x>5.):
        dt= np.diff(t[::100])[0]
        bursts = na(spike_burst_det(x[::100],dt,burst_detection_params))
#         if len(bursts)>30:
#             bursts = na(clean_bursts(bursts))
        if len(bursts)>30:
            durs = np.diff(bursts)/1000
            ibis =(bursts[1:,0]-bursts[:-1,1])/1000
            print(len(ibis),len(durs))
            mibi = np.mean(ibis)
            mdur = np.mean(durs)
            cv_ibis = np.std(ibis)/mibi
            cv_durs = np.std(durs)/mdur
    return torch.tensor([mibi,cv_ibis,mdur,cv_durs,bi])



def batch_simulatior_traces(theta: torch.Tensor): 
    """Return a batch of simulations by looping over a batch of parameters."""
    assert theta.ndim > 1, "Theta must have a batch dimension."
    xs = list(map(simulator_summaryN, theta))
    return torch.cat(xs, dim=0).reshape(theta.shape[0], -1)

def simulatorN(params=None,
                  n_events = 30,
                  main_seed= 123456789,
                  max_iter=10,
                  min_init = 10,
                  torch_=True,
                  param_keys=['b','theta','tau_w','sigma'],
                  burst_detection_params={'maxISIstart':.5,
                                        'maxISIb':.5,
                                        'minBdur':5.,
                                        'minIBI':5.,
                                        'minSburst':100.,
                                        'scale':10.
                                        },
                  sim_func = StochSim_o,
                  default_params = None,
                  thr_scale = 7. 
                  ):
    """ 
    wrapper to collect N bursts
    Careful: The code either return requested N bursts or the maximum number of bursts
    collected over the max_iter. 
    Args:
            params (dict): Dictionary of paramters
            n_events (int): number of bursts to collect
            durs (bool): collect burst durations too
            thr_scale (float): threshold the mean activity, depends on the amplitude of the burst 
            (given by J approximately)
    Returns:
            list: inter-burst intevals (ms)
    #TODO authomatically identify what is the acceptable minimum NB given T?
    #
    """
    inverted=False
    bursts_all = []
    #first round and termalization
    np.random.seed(main_seed)
    #Thermolization
    params = expand_params(params,T=10000,torch_=torch_,keys=param_keys,params=default_params)
    t,x,w = sim_func(params,torch=0)
    params['x0']=x[-1]
    params['w0']=w[-1]
    params['T']=3000000
    t,x,w = sim_func(params,torch=0)
    # bi = bimodality_index(x)
    params['x0']=x[-1]
    params['w0']=w[-1]
#     x,inverted = grid_inv(x)
    if np.sum(x>thr_scale)>10000 and np.any(x<=0.):#bs=0.55 is standard
        dt= np.diff(t[::100])[0]
        bursts_all = na(spike_burst_det(x[::100],dt,burst_detection_params))
        t_last =t[-1]
        # print(t_last)
        nb = len(bursts_all)
        if len(bursts_all)>2:
            if nb>=n_events:
                return bursts_all
            else:
                #The main loop starts here
                for i in range(max_iter):#max number of iterations
                    np.random.seed(main_seed+i)
                    t,x,w = sim_func(params,torch=0)
                    params['x0']=x[-1]
                    params['w0']=w[-1]
                    if np.sum(x>7.)>10000 and np.any(x<=0.):
                        bursts = na(spike_burst_det(x[::100],dt,burst_detection_params))
                        if len(bursts)>0:
                            bursts_all = np.vstack([bursts_all,bursts+t_last])
                            nb += len(bursts)
                    t_last +=t[-1] 
                    if nb>=n_events:
                        break
    return bursts_all
    

def simulator_summaryN(params_set,
                        n_events=30,
                        simfunc=StochSim_o,
                        seed=123456789,
                        param_keys=['b','theta','tau_w','sigma'],
                        output=['mibi','cv_ibis','mdur','cv_durs','r1','r2'],
                        burst_detection_params={'maxISIstart':10.,
                                                'maxISIb':10.,
                                                'minBdur':20.,
                                                'minIBI':20.,
                                                'minSburst':27.,
                                                'scale':10.},):
    """simulate for sbi with min n_events set"""
    bursts = simulatorN(params_set,torch_=True,
                         n_events = n_events,
                         param_keys=param_keys,
                        burst_detection_params =burst_detection_params,
                        sim_func = StochSim_o,
                        main_seed = seed

                        )
    # print(len(bursts))
    mibi,mdur,cv_ibis,cv_durs,r1,r2= np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
    summary = {'mibi':mibi,'mdur':mdur,'cv_ibis':cv_ibis,'cv_durs':cv_durs,'r1':r1,'r2':r2}
    if len(bursts)>5:
        bursts= na(bursts)[3:,:]
        durs = np.diff(bursts)/1000
        ibis =(bursts[1:,0]-bursts[:-1,1])/1000
        mibi = np.mean(ibis)
        mdur = np.mean(durs)
        cv_ibis = np.std(ibis)/mibi
        cv_durs = np.std(durs)/mdur
        r1 = np.corrcoef(ibis[:],durs[:-1,0])[0,1]
        r2 =np.corrcoef(ibis[:],durs[1:,0])[0,1]
        summary = {'mibi':mibi,'mdur':mdur,'cv_ibis':cv_ibis,'cv_durs':cv_durs,'r1':r1,'r2':r2}
    output = [summary[o] for o in output]
    return torch.tensor(output)


def batch_simulatior_tracesN(theta: torch.Tensor): 
    """Return a batch of simulations by looping over a batch of parameters."""
    assert theta.ndim > 1, "Theta must have a batch dimension."
    xs = list(map(function, theta))
    return torch.cat(xs, dim=0).reshape(theta.shape[0], -1)


def grid_inv(x,median_thr=8):
    """function to check the mediam and inverse the trace
    to get the bursts duration in the up-excitable state
    """
    
    if np.median(x)>median_thr:
        return (np.median(x)-x, True)
    else:
        return (x, False)
    
        
    
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

def expand_params(params_set,T=500000,
            keys=['a','b','theta','tau_w','J','sigma'],
            params  = None,
            dt=0.05, 
            torch_ = False,
            ):
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

    #deafault parameters
    if params is None:
        params ={
            'a':5.,#:params_set[0],
            'b':np.nan,#params_set[1],
            'theta':0,#params_set[2],
            'tau':1.,#20! carefull here
            'tau_w':500.,#tau_w,#*1000,
            'mu':0.0,
            'c':1., #standard weight
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
    #make sure that tau_w is in exp
    if 'tau_w' in keys:
        w_ind = ['tau_w' == k for k in keys]
        tau_w = np.exp(na(params_set_)[w_ind][0])
        params['tau_w'] = tau_w

    if 'exp_tau_w' in keys:
        # normal tau_w fiiting (not in log space))
        w_ind = ['exp_tau_w' == k for k in keys]
        tau_w = na(params_set_)[w_ind][0]
        params['tau_w'] = tau_w
    # print(params)

    return params