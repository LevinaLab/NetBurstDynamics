#Calculate Transition probabilite class
import sys

from matplotlib.pyplot import get 
from scipy.integrate.quadpack import quad
import numpy as np
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from scipy.special import expi as expi
from multiprocessing import Pool
from functools import partial
from scipy.optimize import curve_fit
from src.dynamics_Model2 import StochSim_o, findFP, findFakeFP, fullRates, logit
from src.helpers import getNull, spike_burst_det, wTransitions
from scipy.optimize import curve_fit
from src.helpers import getNull
na =np.array

class Model2(object):
    """class to run the numerics vs analytics experiments
    simulates:
    x' = -x + 1/(1+exp(a*(x-w-theta))) +mu + noise
    w' = -w +b*x
    """
    def __init__(self,params,torch=False):
        """ 
        'a':a,#
        'b':b,#adaptation increment
        'theta':theta,#sigmoid shift
        'tau':1., #
        'tau_w':1000.,#adaptation timescale
        'mu':0.,#constant input
        'J':1.0, # 
        'T':T, # max time
        'dt':dt, #integration timescale
        'x0':0., initial x
        'w0':0. # initial w
        'D'(float) diffusion constant sigma**2/2
        'cores'(int): number of cores for parallel processing 
        """
        self.params =params
        self.T = params['T']
        self.dt = params['dt']
        self.n_cores = params['cores']
        self.absXw = params['absXw']# rectify coupling between x and w
        self.torch=torch
        self.d=1/params['tau_w']

        
    def nullclines(self):
        return getNull(self.params)
        
    def simulate(self,T):
        t0 = np.arange(0,T,self.dt)
        tI,xI,wI = StochSim_o(self.params,self.params['mu'],self.params['sigma'],t0,torch=self.torch)
        return tI,xI,wI       
    
    def getFP(self,fake=False,exc=False):
        #find the fixed points graphically 
        if fake:
            self.fp_stat =findFakeFP(self.params,0,0.6,0.0001)
            if exc:
                fp01=findFP(self.params)
                #use the correct low FP for excitable regime
                self.fp_stat[0][0] = fp01[0][0]
                self.fp_stat[1][0] = fp01[1][0]
        else:
            self.fp_stat=findFP(self.params,xmin=0.01,xmax=11.1)
        
        
    def getFeatures(self,xI,wI,tI,HMM=False,filt=True,window=801,order=5):
        """getting features of the bursting 
        #TODO: refactor the boundaries correct"""
        dt = self.params['dt']
        bursts= na(spike_burst_det(xI,dt))[1:,:]
        ibis = bursts[1:,0]-bursts[:-1,0]
        durs = np.diff(bursts)
        w_dus, w_uds = wTransitions(bursts,xI,wI,dt)
        w_dus,w_uds = na(w_dus),na(w_uds)
        w_diff = w_uds[:-1]-w_dus[1:] 
        w_ratio = w_dus[1:]/w_uds[:-1]
        
        features = {'xI':xI,
                    'wI':wI,
                    'w_du':w_dus,
                    'w_ud':w_uds,
                    'bursts':bursts,
                    'w_diff':w_diff,
                    'w_ratio':w_ratio,
                    'ibis':ibis,
                     'durs':durs}
        return features
    
    def parsim(self,seed,T,HMM=False):
        np.random.seed(seed)
        tI,xI,wI = self.simulate(T)
        return self.getFeatures(xI,wI,tI,HMM=HMM)
        
        
    def parallelSim(self,n=1,T=1000,HMM=False):
        """because Irun multiple parallel simulations
        Compute the w at down->up and up-> down, ibis and 
        concatinate the observations"""
        seeds = np.arange(n)*1111
        pool = Pool(processes=n)#self.n_cores
        parsims=partial(self.parsim,T=T,HMM=HMM)
        sol= pool.map(parsims, seeds)
        pool.close()
        
        #concat the results
        IBIs = []
        w_dus= []
        w_uds= []
        w_diffs = []
        w_ratio = []
        for s in sol:
            IBIs.append(s['ibis'])
            w_dus.append(s['w_du'])
            w_uds.append(s['w_ud'])
            w_diffs.append(s['w_diff'])
            w_ratio.append(s['w_ratio'])
            
        IBIs = np.hstack(IBIs)
        w_dus = np.hstack(w_dus)
        w_uds = np.hstack(w_uds)
        w_diffs = np.hstack(w_diffs)
        w_ratio = np.hstack(w_ratio)
        
        return sol,IBIs,w_dus,w_uds,w_diffs,w_ratio
#         W_DU = np.hstack()

    def exp_func(self,x, a,b):
            return np.exp(a+(b*x))
        
    def get_Pdu(self,dw=0.01):
        """return the function for p{d->u}(w)"""
        # #get the nullclines and fixed points
        self.Pdu= {}#collection for Pdu
        x_ = np.arange(-10.5,10.5,0.01)
        #estimate the mean escape as a function of w
        self.fp_stat =findFP(self.params)
                             #,0,0.6,0.001)#find the fixed points graphically 
        if len(self.fp_stat[0])<3:
            print('finding fake fixed point')
            self.fp_stat =findFakeFP(self.params,0,0.6,0.0001)
#             raise ImplementationError('w0 estimate might not be accurate')
        print(self.fp_stat)
#         wmin,wmax=self.fp_stat[1][0],self.fp_stat[1][0]+0.4#fp_stat[1][1]-0.1,fp_stat[1][1]+0.1# 0.,fp_stat[1][0]+0.5#fp_stat[1][0]-0.2

        ws_ = np.arange(self.fp_stat[1][0]-0.1,self.fp_stat[1][0]+0.5,dw) # initial ws to estimate the escapes
        pool = Pool(processes=30)
        MET=partial(fullRates,params=self.params,a=self.fp_stat[0][0],b= self.fp_stat[0][2])
        sol= pool.map(MET, ws_)
        pool.close()
        kr_full=1/na(sol)
        x = ws_[np.isfinite(kr_full)]
        y = kr_full[np.isfinite(kr_full)]   
        #Automatic wmin,wmax
        mask1 = y<10e-10
        mask2 = y>0.01
        x = x[(mask1==0)*(mask2==0)]#new x for fitting
        y = y[(mask1==0)*(mask2==0)]#new y for fitting
#         x = np.arange(np.min(ws),np.max(ws),dw)
    
        #fitting
        self.Pdu['y'] =y#original hazard function
        self.Pdu['ws'] =  x
        
        popt, pcov = curve_fit(self.exp_func, x, y)
        y_est = self.exp_func(x,*popt)
        self.Pdu['y_est']= y_est#exp fit
        self.Pdu['expfit_MSE'] = np.mean((y - y_est)**2) #fit erroe
        popt, pcov = curve_fit(self.exp_func, x, y)
        c1 = popt[0]
        c2= popt[1]
        self.Pdu['c1'] = c1
        self.Pdu['c2'] = c2
        k = self.fp_stat[1][0]#wmin #
        d =1/(self.params['tau_w'])
        return partial(self.Tpdf,c1=c1,c2=c2,d=d,k=k)
    
    def get_Pud(self,dw=0.01):
        """"return the function for p{u->d}(w)
        """
        self.Pud= {}#collection for Pdu
        # #get the nullclines and fixed points
        x_ = np.arange(-10.5,10.5,0.01)
        #estimate the mean escape as a function of w
#         fp_stat =findFP(self.params)#find the fixed points graphically 
#         if len(fp_stat[0])<3:
#             print('finding fake fixed point')
#             fp_stat =findFakeFP(self.params)
#             raise ImplementationError('w0 estimate might not be accurate')
        print(self.fp_stat)
#         wmin,wmax=self.fp_stat[1][2]+0.05,self.fp_stat[1][2]+0.2
        #fp_stat[1][2]-0.15,fp_stat[1][2]-0.05 #fp_stat[1][2]-0.15,fp_stat[1][2]-0.05#fp_stat[1][-1]#(2*params['D'])
        # print(wmin,wmax)
        
        ws_ = np.arange(self.fp_stat[1][2]-0.5,self.fp_stat[1][2]+0.1,dw) # initial ws to estimate the escapes
        pool = Pool(processes=30)
        MET=partial(fullRates,params=self.params,a=self.fp_stat[0][2],b=self.fp_stat[0][0],up=True)
        sol= pool.map(MET, ws_)
        pool.close()
        kr_full=1/na(sol)
        x = ws_[np.isfinite(kr_full)]
        y = kr_full[np.isfinite(kr_full)]
        self.Pdu['y'] =y#original hazard function
        self.Pdu['ws'] =  x
        #Automatic wmin,wmax
        mask1 = y<10e-5
        mask2 = y>0.01
        x = x[(mask1==0)*(mask2==0)]#new x for fitting
        y = y[(mask1==0)*(mask2==0)]#new y for fitting
        popt, pcov = curve_fit(self.exp_func, x, y)
        y_est = self.exp_func(x,*popt)
        self.Pud['y_est']= y_est
        self.Pud['expfit_MSE'] = np.mean((y - y_est)**2)
        popt, pcov = curve_fit(self.exp_func, x, y)
        c1 = popt[0]
        c2= popt[1]
        self.Pud['c1'] = c1 #exp coeff 1
        self.Pud['c2'] = c2 # exp coeff 2
        k =self.fp_stat[1][2]# max#
        d =1/(self.params['tau_w'])
        return partial(self.Tpdf,c1=c1,c2=c2,d=d,k=k,minus=-1)

    
    def Tpdf(self,x,c1=0,c2=0,d=0,k=0,minus=1):
        """transition probability for a single exponent hazard
        and slow dynamics of -w+w0 """
#         t2 = c1/(d*(k-x))
#         t1 = np.exp(-(c2*k)+ (c2*(k-x))+ 
#                     ((c1*np.exp(-c2*k)*expi(c2*(k-x)))/d))
#         res = minus*(-t2*t1)
#         iint_appx = (np.exp(c1+c2*k) * expi(c2*(x-k)) )/d
        #-(c1/d) * np.exp(-c2*k) * expi(c2*(k-x))
#         rho = -((np.exp(c1+c2*x))/((d*(k-x))))*np.exp(-iint_appx)

        iint_appx = -(1/d)*np.exp(c1+(c2*k)) * expi(c2*(x-k))
        rho = -((np.exp(c1+(c2*x)))/((d*(k-x))))*np.exp(-iint_appx)
        return killnans(minus*rho)

    def logpd(self,x, pdf):
        """log pho
        input:
           x(float or array): values of x
           pdf(func): pdf for the transitions
            
        """
        res = pdf(np.exp(x))*np.abs(np.exp(x))
        return killnans(res)
    
    def getTransitions(self):
        """save the transition probabilities helper """
        self.pdfdu = self.get_Pdu()
        self.pdfud = self.get_Pud()
        

    def convF(self,x):
        """estimates the transition distributions and 
        convolve w and w0"""   
        pdf1 = partial(self.logpd, pdf =self.pdfdu)
        pdf2 = partial(self.logpd ,pdf =self.pdfud)
        def tau_pdf(x, pdf = pdf1,d = self.d):
            return killnans(d*pdf(x*d))
        def integral(x_,x=0,d = self.d):
            rr = killnans(tau_pdf(x_,pdf=pdf1,d=d)*tau_pdf(x+x_,pdf=pdf2,d=self.d))*10e20
            return rr
        #carefull with the boundaries of integration
        ee = 1/10e20
        return ee*quad(integral,-100000,1000,args = (x,self.d))[0]

    def pIBI(self,x):
        """return the p of IBI"""
        if len(x)>0:
        # xs =np.arange(0,20000,100.)#np.log(np.arange(0.00015,2.26,0.1))
            yys = na([self.convF(x_) for x_ in x])
        else:
            yys = self.convF(x)
        return yys
    
def killnans(res):            
    try:
        res[np.isnan(res)] = 0
        res[res<0] = 0
        res[np.isinf(res)] = 0
        return res
    except:
        if res ==-np.inf:
            return 0. 
        elif res==np.nan:
            return 0.
        elif res<0.:
            return 0
        else:
            return res

    
    
#-----------Functions to solve the rho(down_up) fully numerically (no exp approx)
#-----------

def getMETk(w,invert=1,params=None):
    """get the mean escape time function and the w fixed point
    
    Input:
        w(float): value of adaptation
        invert ([-1,1]): 1 for estimating down-up transitions, 
                        -1 for up-down
    Returns: 
        MET (function): escape times between a and b, where a and b are 
                        x* fixed points
        w* (float): fixed point of w
    """
    xfps = Eroots(w,params)
    if len(xfps)>1:
        a = np.min(invert*xfps)
        b = np.max(invert*xfps)#xfps[-1]
    else:
        a = xfps[0]
        b = xfps[0]
    return partial(fullRates,a=a,b=b,params=params), params['b']*a

def nu_rate(w,MET=None,params=None):
    """convert escape times into rate"""
    MET,_=  getMETk(w)
    return 1/MET(w)

def fullIntegrand(w,invert=False,params=None):
    """ integrand nu(w)/ d(k-w)"""
    MET,k=  getMETk(w)
    return nu_rate(w,MET)/((d*(k-w)))

def compFullInteg(x):
    """ solving the integral 
    TODO: upper bundary +inf doesn't work
    function goes to zero rather fast, it's meaningful to 
    set to to the upper fixed point of w, but it
    has to change depending on the problem
    """
    return quad(fullIntegrand,x,0.3)[0]

def getRho(x,params):
    pool = Pool(processes=60)
    fullIntegrand =partial(fullIntegrand,params=params)
    nus_d= pool.map(fullIntegrand, x)
    pool.close()

    pool = Pool(processes=60)
    # rates =partial(nu_rate,MET=MET)
    iintfull= pool.map(compFullInteg, x)
    pool.close()

    down_upPdf = -na(nus_d)* np.exp(na(iintfull))    
    return down_upPdf

