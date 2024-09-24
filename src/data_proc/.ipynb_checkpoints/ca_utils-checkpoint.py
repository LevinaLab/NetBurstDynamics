"""Utils for processing Ca traces"""

import scipy.signal
from scipy.signal import butter,filtfilt# Filter requirements.
import numpy as np

date_format = "%d-%m-%y"



import scipy.signal
from scipy.signal import butter,filtfilt# Filter requirements.

def x_to_spikes(x,dt,scale = 500):
    """_summary_

    Args:
        x (_type_): _description_
        dt (_type_): _description_
        scale (int, optional): _description_. Defaults to 500.

    Returns:
        _type_: _description_
    """
    x[x<0] = 0.
    spks = na(scale*(x/x.max()),dtype =int)
    st= np.where(spks)[0]
    st_ = np.repeat(st,spks[st])
    return st_*dt
    
def butter_lowpass_filter(data, cutoff=0.01, fs=0.05, order=2):
    """low-pass filter"""
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def data_in_parts(data1):
    """
    Split the recordings into individual 10min parts 
    (WIS fluorescent microscope recordings)
    """
    breaks_inds = list(np.where(np.diff(data1[:,1])>1)[0]+1)
    if len(breaks_inds)<1:
        breaks_inds = [0,len(data1[:,1])]
    else:
        breaks_inds.append(len(data1[:,1]))
        breaks_inds = [0]+breaks_inds
    #Split the recordings into parts for processing 
    time_parts = []
    trace_parts = []
    for i,break_ind in enumerate(breaks_inds[1:]):
#         print(break_ind)
        nmax = break_ind#23000
        trace = data1[breaks_inds[i]:nmax,2]
        time =data1[breaks_inds[i]:nmax,1]-np.min(data1[breaks_inds[i]:nmax,1])
        #Remove the duplicates
        duplicate_ind = np.where(np.diff(time)==0)[0]+1
        trace = np.delete(trace,duplicate_ind)
        time = np.delete(time,duplicate_ind)
        #remove small traces
        if len(trace)>1000:
            time_parts.append(time)
            trace_parts.append(trace)
    return time_parts,trace_parts

def detrend_parts(time_parts,trace_parts):
    """Detrent the Fluo-4 signal by low-pass filtering"""
    dt = np.diff(time_parts[0])[0]
    detrended_parts = []
    for i,trace in enumerate(trace_parts):
        time = time_parts[i]
        T = np.max(time)         # Sample Period
        fs = 1/dt       # sample rate, Hz
        cutoff = 0.005      # desired cutoff frequency of the filter, Hz ,  
        n = int(T * fs) # total number of samples
        y = butter_lowpass_filter(trace,cutoff,fs,3)
#         y = butter_lowpass_filter(trace-y,.5,fs,3)
        detrended_parts.append(trace-y)
#     plt.figure()
#     plt.plot(trace)
#     plt.plot(trace-y)
    
    return detrended_parts

def deconvolve_parts(trace_parts,parts =[0,2]):
    """deconvolve with OASIS algorithm"""
    rates = []
    rec_signal = [] #reconstructed signal
    for trace in trace_parts[parts[0]:parts[1]]:
        c, s, b, g, lam = deconvolve(trace,penalty=0)#
        rates.append(s)
        rec_signal.append(c)
    return rates,rec_signal
        
        
def get_summaries(data1,parts=[0,2],MI_params = {'maxISIstart':70,
                                                 'maxISIb':70,
                                                 'minBdur':100,
                                                 'minIBI':500,
                                                 'minSburst':10
                                                 }):  
    """preprocess and get summaries of the fluorescnet trace"""
    # preprocess    
    time_parts,trace_parts = data_in_parts(data1)
    dt = np.diff(time_parts[0])[0]
    #detrend
    detrended_parts = detrend_parts(time_parts,trace_parts)
#     rates,rec_signal = deconvolve_parts(detrended_parts)
#     detrended_parts = rec_signal;
    #find bursts
    all_bursts = []
    selected_parts = [detrended_parts[i] for i in parts]
    for detrended_trace in selected_parts:
        st = x_to_spikes(detrended_trace.copy(),dt=dt,scale=100)*1000
        bursts = na(MI_bursts(st,
                  maxISIstart=MI_params['maxISIstart'],#70,#250,
                  maxISIb=MI_params['maxISIb'],#70,#250,
                  minBdur=MI_params['minBdur'],#100,
                  minIBI=MI_params['minIBI'],#500,
                  minSburst=MI_params['minSburst']))
        all_bursts.append(bursts)

    # get summaries
    ibis = []
    durs = []
    for bursts in all_bursts:
        ibis_ = (bursts[1:,0]-bursts[:-1,1])/1000
        durs_= np.diff(bursts)/1000
        ibis.extend(ibis_)
        durs.extend(durs_)
    mibi = np.round(np.mean(ibis),3)
    sem_mibi = np.std(ibis)/np.sqrt(len(ibis))
    mdur = np.round(np.mean(durs),3)
    cv_ibi = np.round(np.std(ibis)/np.mean(ibis),3)
    sem_mdur = np.std(durs)/np.sqrt(len(durs))
    cv_dur = np.round(np.std(durs)/np.mean(durs),3)
    summary = [mibi,cv_ibi,mdur,cv_dur,len(ibis)]
    errors = [sem_mibi,0,sem_mdur,0]
    return time_parts,detrended_parts,all_bursts,summary,errors