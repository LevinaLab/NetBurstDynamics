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
from numba import jit 

# 1D neuron with Sigmoid nonlineariyt 
# Models 
from scipy.integrate import odeint


from sklearn import linear_model
def get_poisson_coef(sc,x_dyn):
    """ get coefficients to predict the fiting rate """
    clf = linear_model.PoissonRegressor(fit_intercept=False)
    X= x_dyn
    X =X[X>4.5]
    y= sc
    y = y[y>50]
    lent = min([len(X),len(y)])
    clf.fit(X[:lent].reshape((lent,1)), y[:lent])
    return clf.coef_


def get_slope(mibi,dur,J):
    """ Compute the approximate excitability/adatation"""
    p_up= dur/(mibi+dur)
    mean_x = p_up *J
    return mean_x



params = {}

@jit(nopython=True)
def F(x,w,a,theta,j):
    return j/(1+np.exp(-a*(x-w-theta)))
#-(x**3)/3#1/(1+np.exp(-a*(x-theta)))

def logit(x,a,theta):
    return (1/a)*(a*theta + np.log(x/(1-x)))

def energy(x,a,theta,J,w,integ=False):
    xi = np.exp(a*(theta+w-x))
    E = -(1/a)*J*np.log(xi+1)+((x**2)/2)-(J*x)
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
    return (-x + F(x-w,a,theta))/tau


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


def StochSim_o(params,torch=True):
    """Optimized stochastic sim
    with inhibition"""
    t = np.arange(0,params['T'],params['dt'])

    n = len(t)#int(T / dt)  # Number of time steps.
    if torch:
        a = params['a'].numpy()
        b = params['b'].numpy()
        theta =params['theta'].numpy()
        tau_w =params['tau_w'].numpy()
        sigma = params['sigma'].numpy()
        mu = params['mu'].numpy()
        c= params['c'].numpy()
        tau = params['tau'].numpy()

        
    else:
        a = params['a']
        b = params['b']
        theta =params['theta']
        tau_w =params['tau_w']
        j = params['J']
        sigma = params['sigma']
        mu = params['mu']
        c= params['c']
        tau = params['tau']

    dt = params['dt']
    sqrtdt = np.sqrt(dt)
    x = np.zeros(n,dtype=np.float32)
    w = np.zeros(n,dtype=np.float32)
    x[0] = params['x0']
    w[0] = params['w0']
    noise = np.random.normal(0,sqrtdt,size = (n,1))
    x,w = run_sim_o(n,w,x,b,tau_w,mu,a,theta,j,c,sigma,noise,tau,dt)
    return t,x,w

@jit(nopython=True)
def run_sim_o(n,w,x,b,tau_w,mu,a,theta,j,c,sigma,noise,tau,dt):
    for i in range(n - 1):
        w[i+1] = w[i] + dt*((-w[i]+b*x[i])/ tau_w)#+(sigma*noise[i])
        x[i+1] = x[i] + dt*((1/tau)*(-x[i] +(j/(1+np.exp(-a*(c*x[i]-w[i]-theta+mu))))))  +  ((sigma)* noise[i,0])
    return x,w


from numba import jit
def StochSimI_mult(params,torch=True):
    """Optimized stochastic sim
    with inhibition"""
    t = np.arange(0,params['T'],params['dt'])

    n = len(t)#int(T / dt)  # Number of time steps.
    if torch:
        a = params['a'].numpy()
        b = params['b'].numpy()
        theta =params['theta'].numpy()
        tau_w =params['tau_w'].numpy()
        sigma = params['sigma'].numpy()
        mu = params['sigma'].numpy()
        
    else:
        a = params['a']
        b = params['b']
        theta =params['theta']
        tau_w =params['tau_w']
        j = params['J']
        sigma = params['sigma']
        mu = params['mu']

    dt = params['dt']
    sqrtdt = np.sqrt(dt)
    x = np.zeros(n,dtype=np.float32)
    w = np.zeros(n,dtype=np.float32)
    x[0] = params['x0']
    w[0] = params['w0']
    noise = np.random.normal(0,sqrtdt,size = (n,2))
    x,w = run_sim(n,w,x,b,tau_w,mu,a,theta,j,sigma,noise,dt)
    return t,x,w

@jit(nopython=True)
def run_sim(n,w,x,b,tau_w,mu,a,theta,j,sigma,noise,dt):
    for i in range(n - 1):
        w[i+1] = w[i] + dt*((-w[i]+b*x[i])/ tau_w)#+(sigma*noise[i])
        x[i+1] = x[i] + dt*((-x[i] +(j/(1+np.exp(-a*(x[i]-w[i]-theta+((sigma/dt)* noise[i,0])))))+mu))
    return x,w





from numba import jit
def StochSim_add(params,torch=True):
    """Optimized stochastic sim
    with inhibition"""
    t = np.arange(0,params['T'],params['dt'])

    n = len(t)#int(T / dt)  # Number of time steps.
    if torch:
        a = params['a'].numpy()
        b = params['b'].numpy()
        theta =params['theta'].numpy()
        tau_w =params['tau_w'].numpy()
        sigma = params['sigma'].numpy()
        mu = params['sigma'].numpy()
        
    else:
        a = params['a']
        b = params['b']
        theta =params['theta']
        tau_w =params['tau_w']
        j = params['J']
        sigma = params['sigma']
        mu = params['mu']

    dt = params['dt']
    sqrtdt = np.sqrt(dt)
    x = np.zeros(n,dtype=np.float32)
    w = np.zeros(n,dtype=np.float32)
    x[0] = params['x0']
    w[0] = params['w0']
    noise = np.random.normal(0,sqrtdt,size = (n,2))
    x,w = run_sim_add(n,w,x,b,tau_w,mu,a,theta,j,sigma,noise,dt)
    return t,x,w

@jit(nopython=True)
def run_sim_add(n,w,x,b,tau_w,mu,a,theta,j,sigma,noise,dt):
    for i in range(n - 1):
        w[i+1] = w[i] + dt*((-w[i]+b*x[i])/tau_w)#+(sigma*noise[i])
        x[i+1] = x[i] + dt*((-x[i] +(j/(1+np.exp(-a*(x[i]-theta)))+mu-w[i])))+((sigma)* noise[i,0])
    return x,w


#Get fixed points 
def get_roots(w,w1=np.nan,x_=np.nan,w2=np.nan):
    w1[:] = w
    fp_stat = intersection(x_,w1,x_,w2)
    return fp_stat

def findFakeFP(params, wmin=0.,wmax=0.5,dw=0.01,par = True):
    """ Find the fake high and low fp graphically
    assuming fixed w
    The method might return inaccurate results
  
    input
        params (dict): model paramters 
        wmin (float): min fixed w 
        wmax (float): max fixed w
        dw(float): step between w for sweep
        
    returns
        fp_fake(2 arrays): x0s, w0s
    """
    x_ = np.arange(-0.1,1.1,0.01)
    w1 = np.zeros_like(x_)
    w2 = -logit(x_/params['J'],params['a'],params['theta'])+x_
    ws = np.arange(wmin,wmax,dw)
    # roots = np.zeros(shape=(3,len(w_),2))
    roots=[]
    pool = Pool(processes=40)
    pRoots=partial(get_roots,x_=x_,w1=w1,w2=w2)
    roots= pool.map(pRoots, ws)
    pool.close()
    n_roots = [len(r[0]) for r in  roots]
    #remove detection errors
    #false H fixed points
    badVals = [np.any(r[0]>10) for r in roots]
    badInd = np.where(badVals)[0]
    for bI in badInd:
        roots[bI] = na([roots[bI][0]])
        n_roots[bI] = len(roots[bI])

    ind01=np.where(na(n_roots)>1)[0][0]
    ind02 =np.where(na(n_roots)>1)[0][-1]
    x01 =roots[ind01][0][0]
    x02 =roots[ind02][0][1]
    w01 = ws[ind01]#ws[na(n_roots)>1][0]
    w02 = ws[ind02]#ws[na(n_roots)>1][::-1][0]
    return ([x01,np.nan,x02],[w01,np.nan,w02])



def findFakeFPv2(params, wmin=0.,wmax=0.5,dw=0.01,par = True):
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


@jit(nopython=True)
def psDyn(X,t,a,theta,b,j,tau_w):
    x,w = X
    dx = -x+F(x,w,a,theta,j) 
    dw = (-w+(b*x))/tau_w
    return [dx,dw]


def findOsc(params,tmax=4000000,dt=0.1):
    """Numberically estimate if the system oscillates
    params (dict): paramters of the networks
    Returns: 
        True or False 
    """
    
    # t =np.arange(0,tmax,dt)
    # sol = odeint(psDyn, [0,0], t,args =(params['a'],params['theta'],
                            # params['b'],params['J'],params['tau_w']))
    params['T'] = 4000000
    params['dt']=dt
    _,_,w= StochSim_o(params,torch=False)
    traj_var = np.var(w[int((tmax/dt)*(1/2)):])
    osc=True
    if np.round(traj_var,3)<10e-6:
        osc = False
    return osc


def FPmultiStart(params):
    """Numberically estimate the FP
    Does not guarantee to find the unstable fixed point
    Sometimes finds it by change 

    params (dict): paramters of the networks
    u0(list): intial conditions 
    Returns: 
        x0,w0 
    """

    t =np.arange(0,40000,0.5)
    init_conds_x = np.linspace(0,params['J']+0.2,20)
    init_conds_w = init_conds_x*params['b']
    init_conds = np.vstack([init_conds_x,init_conds_w]).T
    sol = [odeint(psDyn, cond, t,args =(params['a'],params['theta'],
                    params['b'],params['J'],1)) for cond in init_conds ]
    u_sol = np.unique([np.round(s[-1],4) for s in sol],axis = 0)
    return u_sol




from intersect import intersection
def findFP(params,xmin=0.1,xmax=1.,dx=0.001):
    """Find all fixed posints graphically
    input:
    params(dict): network parameters
    dx(float): resolution
    reutrns
    fixed points (2 arrays): x0s, w0s
    """
#     x_ = np.arange(-10.5,10.5,0.01)
    x_ = np.arange(xmin,xmax,dx)
    w01 = x_*params['b']
    w02 = -logit(x_/params['J'],params['a'],params['theta'])+x_
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
        dx = -x+F(x-w,a,theta) 
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
    J = params['J']
    x = sp.symbols('x',real=True)
    xi = sp.exp(a*(theta+w-x))
    E = -(1/a)*J*sp.log(xi+1)+((x**2)/2)-(J*x)
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

def fullRates(w,a=np.nan,b=np.nan, params=params,up=False):
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
    def integrand1(y, a,theta,J, w, D):
        return np.exp(-U(invert*y,a,theta,J,w)/D)

    def integrandOuter(x,a,theta,J,w,D):
            return np.exp(U(invert*x,a,theta,J,w)/D)*quad(integrand1,-np.inf,x, 
                                                      args = (a,theta,J,w,D))[0]
    def MET(x0,x1,a,theta,J,w,D): 
        integral= quad(integrandOuter, x0,x1, args=(a,theta,J,w,D))[0]
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
    return MET(a,b,params['a'],params['theta'],params['J'],w,params['D'])



