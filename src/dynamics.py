import sys, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['pdf.fonttype'] = 42
from multiprocessing import Pool
from functools import partial
na = np.array

# 1D neuron with Sigmoid nonlineariyt 
# Models 
from scipy.integrate import odeint
params = {}
def F(x,a,theta):
    return 1/(1+np.exp(-a*(x-theta)))
#-(x**3)/3#1/(1+np.exp(-a*(x-theta)))


def energy(x,a,theta,w,integ=False):
    xi = np.exp(a*(theta-x))
    E = (1/2)*(x-2)*x-(1/a)*np.log(xi+1)+(w*x)
#     xi = bf.exp(a*(theta-x))
#     E = (1/2)*(x-2)*x-((1/a)*bf.log(xi+1)+(w*x))
    if np.any(E==-np.inf):
        E =np.inf
    return E

U = energy 


def fiFN(x,w,params):
    a = params['a']
    b = params['b']
    theta =params['theta']
    tau =params['tau']
    tau_w =params['tau_w']
    return (x - ((x**3)/theta)-w)/tau

def fi(x,w,params):
    a = params['a']
    b = params['b']
    theta =params['theta']
    tau =params['tau']
    return (-x + F(x,a,theta)-w)/tau


def StochSim(params,mu,sigma,t):
#     def wx_coupling(x):
#         if x>0.5:
#             return x
#         else:
#             return 0
    n = len(t)#int(T / dt)  # Number of time steps.
    a = params['a']
    b = params['b']
    theta =params['theta']
    tau =params['tau']
    tau_w =params['tau_w']
    dt = params['dt']
    sqrtdt = np.sqrt(dt)
    x = np.zeros(n)
    w = np.zeros(n)
    x[0] = params['x0']
    w[0] = params['w0']
    absXw= params['absXw']#recity the w and x 
    #set the w-x coupling
    if absXw:
        wx = lambda x: max(0,x)#wx_coupling
    else:
        wx = lambda x: x
        
    for i in range(n - 1):
        w[i+1] = w[i] + dt*((-w[i]+b*wx(x[i]))/ tau_w)
        x[i+1] = x[i] + dt*(fi(x[i],w[i]+mu,params))+ \
          (sigma* np.random
           .normal(0,sqrtdt)) ##(-(x[i] - mu)/tau + I - w[i])# #sigma_bis * sqrtdt 
    # plt.plot(t,x)#sqrtdt
#     print(x[-1])
    return t,x,w


from numba import jit

def StochSim_o(params,mu,sigma,t,torch=True):
    """Optimized stochastic sim"""
    n = len(t)#int(T / dt)  # Number of time steps.
    if torch:
        a = params['a'].numpy()
        b = params['b'].numpy()
        theta =params['theta'].numpy()
        tau_w =params['tau_w'].numpy()
        sigma = sigma.numpy()
        mu = mu.numpy()
        
    else:
        a = params['a']
        b = params['b']
        theta =params['theta']
        tau_w =params['tau_w']

    tau =params['tau']

    dt = params['dt']

    sqrtdt = np.sqrt(dt)
    x = np.zeros(n)
    w = np.zeros(n)
    x[0] = params['x0']
    w[0] = params['w0']
    absXw= params['absXw']#recity the w and x 
    #set the w-x coupling
#     def F(x,a,theta):
#         return 1/(1+np.exp(-a*(x-theta)))
    
    def fi(x,w,a,b,theta,tau):
        return (-x + (1/(1+np.exp(-a*(x-theta))))-w)/tau
    
    if absXw:
        wx = lambda x: max(0.5,x)#wx_coupling
    else:
        wx = lambda x: x
        
    noise = np.random.normal(0,sqrtdt,size = (n,))
    x,w = run_sim(n,w,x,b,tau_w,mu,a,theta,tau,sigma,noise,dt)
    
    return t,x,w

@jit(nopython=True)
def run_sim(n,w,x,b,tau_w,mu,a,theta,tau,sigma,noise,dt):
    #print(n,w,x,b,tau_w,mu,a,theta,tau,sigma,noise,dt)
    for i in range(n - 1):
        w[i+1] = w[i] + dt*((-w[i]+(b*x[i]))/ tau_w)
        x[i+1] = x[i] + dt*((-x[i] + (1/(1+np.exp(-a*(x[i]-theta))))-w[i]+mu)/tau)+ \
          (sigma* noise[i])
        
    return x,w





#Get fixed points 


def findFakeFP(params, wmin=0.,wmax=0.5,dw=0.01,par = True):
    """ Find the fake high and low fp 
    assuming the seperation of timescles
    The method might return inaccurate results
    input
        params (dict): model paramters 
        wmin (float): min fixed w 
        wmax (float): max fixed w
        dw(float): step between w for sweep
        
    returns
        fp_fake(2 arrays): x0s, w0s
    """

    ws= np.arange(wmin,wmax,dw)
    #paralellize the detection

    if par:
        pool = Pool(processes=40)
        findRoots=partial(Eroots,params=params)
        roots= pool.map(findRoots, ws)
        pool.close()
    else:
        roots = []
        for w in ws:
            roots.append(Eroots(w,params))
        
    n_roots = [len(r) for r in  roots]
    #remove detection errors
    #false H fixed points
    badVals = [np.any(r>10) for r in roots]
    badInd = np.where(badVals)[0]
    for bI in badInd:
        roots[bI] = na([roots[bI][0]])
        n_roots[bI] = len(roots[bI])
    ind01=np.where(na(n_roots)>1)[0][0]
    ind02 =np.where(na(n_roots)>1)[0][-1]
    x01 =roots[ind01][0]
    x02 =roots[ind02][1]
    w01 = ws[ind01]#ws[na(n_roots)>1][0]
    w02 = ws[ind02]#ws[na(n_roots)>1][::-1][0]
    
    return ([x01,np.nan,x02],[w01,np.nan,w02])



from intersect import intersection
def findFP(params,dx=0.01):
    """Find all fixed posints graphically
    input:
    params(dict): network parameters
    dx(float): resolution
    reutrns
    fixed points (2 arrays): x0s, w0s
    """
#     x_ = np.arange(-10.5,10.5,0.01)
    x_ = np.arange(-10.5,10.5,dx)
    w01 = x_*params['b']
    w02 = F(x_,params['a'],params['theta'])-x_+params['mu']
    fp_stat = intersection(x_,w01,x_,w02)
    return fp_stat


def FPestimate(params,u0):
    """Numberically estimate the FP
    params (dict): paramters of the networks
    u0(list): intial conditions 
    Returns: 
        x0,w0 
    
    """
    
    def psDyn(X,t,a,theta):
        x,w = X
        dx = -x+F(x-,a,theta)
        dw = -w+b*x
        return [dx,dw]
    t =np.arange(1,200,0.01)
    sol= odeint(psDyn,[u0,u0],t,args =(params['a'],params['theta']))
    return sol[-1,0],sol[-1,1]

from scipy.integrate import quad

import sympy as sp
from scipy.optimize import fsolve,root,basinhopping
from scipy.integrate import odeint

# Mean escape time problem 
from scipy.integrate import quad
def Eroots(w,params={'0':0}):
    """ returns 2 local minima and 1 local maximum of the double well 
    of the form (1/2)*(x-2)*x-(1/a)*sp.log(xi+1)+(w*x) 
    
    input:
        w (float): adaptaiton offset
    returns: 
    u_sol (list): 2 local minima+ 1 local maximum"""
    a = params['a']
    theta = params['theta']
    x = sp.symbols('x',real=True)
    xi = sp.exp(a*(theta-x))
    E = (1/2)*(x-2)*x-(1/a)*sp.log(xi+1)+(w*x)
    dU = -sp.diff(E,x)    
    def der(z,t):
#         print(sp.N(dU.subs(x,z).evalf()))
        return na(sp.N(dU.subs({x:z[0]}).evalf()),dtype=float)
    t =np.arange(1,100,0.01)
    conds = [-1,0,1]
    sol = [odeint(der, cond, t) for cond in conds]
    u_sol = (np.unique([np.round(s[-1],10) for s in sol]))
#     u_fp = np.nan
    u_fp = []
    if len(u_sol)>1:
        #find the unstable FP with reverse time
        sol = odeint(der, np.mean(u_sol), -t)
        u_fp = sol[-1]
        
    u_sol = np.hstack([u_sol,u_fp])
    return u_sol#na(xfps,dtype =float)

#     roots = np.unique(roots)

def fullRates(w,a=np.nan,b=np.nan,params=params,up=False):
    """ Estimate the mean escape time in double well potential
    Uses Etoors - function that returns 2 local minima and 1 local maximum of the double well
    input:
        w(flaot): adaptaiton offset
        a(float): low FP
        b(float): high FP
        params(dict): parameters of the network
        up (bool): invert the E (to get the fullRateUD)
    returns: 
        mean escape time (float): time of the mean escape
    """
    def integrand1(y, a,theta,w, D):
        return np.exp(-U(invert*y,a,theta,w)/D)

    def integrandOuter(x,a,theta,w,D):
            return np.exp(U(invert*x,a,theta,w)/D)*quad(integrand1,-np.inf,x, 
                                                      args = (a,theta,w,D))[0]
    def MET(x0,x1,a,theta,w,D): 
        integral= quad(integrandOuter, x0,x1, args=(a,theta,w,D))[0]
        return (1/D)*integral
    if up:
        invert=-1
    else:
        invert=1
        
    if np.isnan(a):
        #automatically get LFP and HFP for every w
        xfps = Eroots(w,params)
        if len(xfps)>1:
            a = np.min(invert*xfps)
            b = np.max(invert*xfps)#xfps[-1]
        else:
            a = xfps[0]
            b = xfps[0]
    else:
        a = invert*a
        b = invert*b
        
# #     print(a,b)
#     print(a,b,params['a'],params['theta'],w,params['D'])
    return MET(a,b,params['a'],params['theta'],w,params['D'])





# def fullRate(w,params=params,up=False):
#     """ Estimate the mean escape time in double well potential
#     Uses Etoors - function that returns 2 local minima and 1 local maximum of the double well
#     input:
#         w(flaot): adaptaiton offset
#         params(dict): parameters of the network
#         up (bool): invert the E (to get the fullRateUD)
#     returns: 
#         mean escape time (float): time of the mean escape
#     """
#     def integrand1(y, a,theta,w, D):
#         return np.exp(-U(invert*y,a,theta,w)/D)

#     def integrandOuter(x,a,theta,w,D):
#             return np.exp(U(invert*x,a,theta,w)/D)*quad(integrand1,-np.inf,x, 
#                                                       args = (a,theta,w,D))[0]
#     def MET(x0,x1,a,theta,w,D): 
#         integral= quad(integrandOuter, x0,x1, args=(a,theta,w,D))[0]
#         return (1/D)*integral
#     if up:
#         invert=-1
#     else:
#         invert=1
        
#     xfps = Eroots(w,params)
#     if len(xfps)>1:
#         a = np.min(invert*xfps)
#         b = np.max(invert*xfps)#xfps[-1]
#     else:
#         a = xfps[0]
#         b = xfps[0]
# #     print(a,b)
#     return MET(a,b,params['a'],params['theta'],w,params['D'])


# def fullRateUD(w,params):
#     """ modified fullRate function to get U->P transition 
#     Estimate the mean escape time in double well potential
#     Uses Etoors - function that returns 2 local minima and 1 local maximum of the double well
#     input:
#         w(flaot): adaptaiton offset
#         params(dict): parameters of the network
#     returns: 
#         mean escape time (float): time of the mean escape
#     """
#     def integrand1(y, a,theta,w, D):
#         return np.exp(-U(-y,a,theta,w)/D)

#     def integrandOuter(x,a,theta,w,D):
#             return np.exp(U(-x,a,theta,w)/D)*quad(integrand1,-np.inf,x, 
#                                                       args = (a,theta,w,D))[0]
#     def MET(x0,x1,a,theta,w,D): 
#         integral= quad(integrandOuter, x0,x1, args=(a,theta,w,D))[0]
#         return (1/D)*integral

#     xfps = Eroots(w,params)
#     if len(xfps)>1:
#         a = np.min(-xfps)
#         b = np.max(-xfps)#xfps[-1]
#     else:
#         a = xfps[0]
#         b = xfps[0]
# #     print(a,b)
#     return MET(a,b,params['a'],params['theta'],w,params['D'])


