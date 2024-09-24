"""function to calulate bifurcations for the model 2 
x' = -x + /phi(x-w-theta)+noise
tau_w w' = -w + bx
"""

from src.dynamics_Model2 import findFP
import numpy as np

def F_diff(x,w,j,a,theta):
    return j*a*np.exp(-a*(-theta-w+x)) / ((np.exp(-a*(-theta-w+x)) +1 )**2) 
    
def trJ(theta,x0,w0,a,j,tau_w):
    phi_diff = F_diff(x0,w0,j,a,theta)
    return -1-(1/tau_w)+phi_diff

def TraceCheck(pp,params = None,fp_check=True):
    """quick bifurcation calculator
    return the trace of the jacobian for one fixed point solutions (check for Hopf)
    and the number of fixed points

    Args:
        pp ([list, tuple or np array]): [theta and b]
        params ([dict], optional): [description]. Defaults to params.
        fp_check (bool, optional): [description]. Defaults to True.
    #TODO make it work with arbitrary paramters by parsing paramter keys
    """
    params_ = params.copy()
    #parse the paramters
    theta = pp[0]
    b= pp[1]
    tau_w = params['tau_w']
    j = params['J']    
    a = params['a']

    params_['theta']=theta
    params_['b'] = b
#     A = Model2(params)
#     A.getFP()
    fp_stat = findFP(params,xmin=-8.,xmax=8.,dx=0.01)
    tr =np.nan
    n_states = len(fp_stat[0])
    if fp_check==False or n_states==1:
        x0 = fp_stat[0][0]
        w0 = fp_stat[1][0]
        tr = trJ(theta,x0,w0,a,j,tau_w)
    else:
        tr= np.nan
    return tr,n_states
