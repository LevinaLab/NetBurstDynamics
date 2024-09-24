import numpy as np
from scipy.io import loadmat as loadmat
na =np.array
import pandas as pd
from src.helpers import MI_bursts, bimodality_index
from numba import jit
import h5py
import os 
import matplotlib.pyplot as plt
    

import re


def gid_to_numbers(gid):
    """Convert the gid to numbers"""
    for i,u_id in enumerate(np.unique(gid)):
        gid[gid==u_id]=i
    return gid


import os
import re
def load_hyv():
    
    """for Figure 4 """
    path = '/home/ovinogradov/Projects/ReducedBursting/data/iPSCcsCTX/'
    res = list(os.walk(path,topdown=True))
    files =res[0][2] # all file names 
    div_days = [f.split('_')[3] for f in files if 'DIV' in f]
    types = [f.split('_')[0] for f in files if 'DIV' in f]
    mea_n = [f.split('_')[2] for f in files if 'DIV' in f]
    div_days =[re.findall(r'\d+', div) for div in div_days]
    div_days = na(div_days,dtype=int).flatten()
    indis = np.argsort(div_days)
    div_days = div_days[indis]
    types = na(types)[indis]
    mea_n = na(mea_n)[indis]
    files = na(files)[indis]
    divs= []
    summaries = []
    well_id= []
    culture_type= []
    mea_number = []
    sts= []
    gids= []
    dates = []
    for i,file_ in enumerate(files):
        if 'MEA' in file_:
            div = div_days[i]
            type_ = types[i]
            mea_ = mea_n[i]
            spikes =pd.read_csv(path+file_)
            channels = spikes['Channel']
            wells = [ch.split('_')[0] for ch in channels]
            ch_n = [ch.split('_')[1] for ch in channels]
            date = file_.split('_')[1]
            spikes['well'] = wells
            spikes['ch_n'] = ch_n
            # Extract spikes for different wells 
            # well_spikes= []
            for well in np.unique(wells):
                sts.append(na(spikes['Time'][spikes['well']==well]))
                gids.append(na(spikes['ch_n'][spikes['well']==well]))
                divs.append(div)
                dates.append(date)
                well_id.append(well)
                culture_type.append(type_)
                mea_number.append(mea_)

    spikes = pd.DataFrame({'st':sts,
                  'gid':gids,
                  'date':dates,
                  'DIV':divs,
                  'well':well_id,
                  'culture type':culture_type,
                  'MEA':mea_number})
    spikes_filt = spikes[(spikes['DIV']==38 )*(spikes['culture type']=='hPSC')]
    return spikes_filt


def read_ctx_psc_data(path = '/home/ovinogradov/Projects/ReducedBursting/data/iPSCcsCTX/'):
    
    res = list(os.walk(path,topdown=True))
    files =res[0][2] # all file names 
    div_days = [f.split('_')[3] for f in files if 'DIV' in f]
    types = [f.split('_')[0] for f in files if 'DIV' in f]
    mea_n = [f.split('_')[2] for f in files if 'DIV' in f]
    div_days =[re.findall(r'\d+', div) for div in div_days]
    div_days = na(div_days,dtype=int).flatten()
    indis = np.argsort(div_days)
    div_days = div_days[indis]
    types = na(types)[indis]
    mea_n = na(mea_n)[indis]
    files = na(files)[indis]
    divs= []
    summaries = []
    well_id= []
    culture_type= []
    mea_number = []
    sts= []
    gids= []
    dates = []
    for i,file_ in enumerate(files):
        if 'MEA' in file_:
            div = div_days[i]
            type_ = types[i]
            mea_ = mea_n[i]
            spikes =pd.read_csv(path+file_)
            channels = spikes['Channel']
            wells = [ch.split('_')[0] for ch in channels]
            ch_n = [ch.split('_')[1] for ch in channels]
            date = file_.split('_')[1]
            spikes['well'] = wells
            spikes['ch_n'] = ch_n
            # Extract spikes for different wells 
            # well_spikes= []
            for well in np.unique(wells):
                sts.append(na(spikes['Time'][spikes['well']==well]))
                gids.append(na(spikes['ch_n'][spikes['well']==well]))
                divs.append(div)
                dates.append(date)
                well_id.append(well)
                culture_type.append(type_)
                mea_number.append(mea_)
                
    return pd.DataFrame({'st':sts,
                'gid':gids,
                'date':dates,
                'DIV':divs,
                'well':well_id,
                'culture type':culture_type,
                'MEA':mea_number})



def read_ctx_hpc_data(path ='/home/ovinogradov/Projects/ReducedBursting/data/CtxHipp/g2chvcdata-master/inst/extdata/'):
    """Read the ctx hpc data from the g2chvcdata-master folder"""

    res = list(os.walk(path,topdown=True))
    files =res[0][2] # all file names 
    spikes  = []
    sources = []
    info = []
    file_names = []
    spike_count_per_channel = []
    gids = []
    for filename in files:
        with h5py.File(path+filename, "r") as f:
            # List all groups
        #     print("Keys: %s" % f.keys())
        #     a_group_key = list(f.keys())[0]

            # Get the data
            age = list(f['meta']['age'])
            if age[0] in [7,10,11,14,17,18,21,24,25,28]:
                # print(age)
                spikes.append(na(f['spikes']))
                sc_ch = list(f['sCount'])
                spike_count_per_channel.append(sc_ch)
                gid = np.repeat(np.arange(len(sc_ch)),sc_ch)
                gids.append(gid)
                names = list(f['names'])
                info.extend(age) # DIV
                reg= list(f['meta']['region'])
                sources.extend(reg)
                file_names.append([filename]*len(reg))
    spikes_data = pd.DataFrame(data={'DIV':info,
                                    'region':sources,
                                    'file':file_names,
                                    'spikes':spikes,
                                    'gids':gids},)#.to_csv('CtxHipp/ctx_hipp_info.csv')
    return spikes_data
                        


# def pop_burst_detection(st,gid,        
#                 maxISIstart=100,#100.0,
#                 maxISIb=250,
#                 minBdur=50,
#                 minIBI=500,
#                 minSburst=6,
#                 onset_threshold=3,#3,# numer of sim active electrodes to consider a burst
#                 peak_threshold=10,#5# numer of sim active electrodes to consider a burst
#                         ):
#     """Detect network burst based on the detection of bursts in
#     each electrode and further thresholding

#     Args:
#         st (list): spike times (is ms )
#         gid (list): neuron ids
#     Returns:
#         pop_bursts_final (list): list of tuples with the start and end of each burst
#     """
#     st = na(st)
#     gid = na(gid)
#     gid_ids = np.unique(gid)
#     # main vector to store the number of bursts per electrode in each time bin
#     burst_vectors = np.ones(shape=(int(np.max(st)),1))
#     for i,u_id in enumerate(gid_ids):
#         st_ =st[gid==u_id]
#         # Parameters of the MI from Charlesworth et al. 2016
#         bursts = MI_bursts(st_,   
#                 maxISIstart=maxISIstart,#100.0,
#                 maxISIb=maxISIb,
#                 minBdur=minBdur,
#                 minIBI=minIBI,
#                 minSburst=minSburst)
#         for burst in bursts:
#             # accumulate the number of bursts per electrode in each time bin
#             # burst_vectors[int(burst[0]):int(burst[1]),0]+=1
#             burst_vectors[int(burst[0])-5:int(burst[1])+5,0]+=1
#     # add the population detection
#     isi_thr = min(max(np.mean(np.diff(st)),1),500)
#     bursts = MI_bursts(st,   
#             maxISIstart=isi_thr,#100.0,
#             maxISIb=isi_thr,
#             minBdur=50,
#             minIBI=800,
#             minSburst=200)
#     for burst in bursts:
#         burst_vectors[int(burst[0]):int(burst[1]),0]+=10

#     # onset_threshold = np.int(np.mean(burst_vectors))# numer of sim active electrodes to consider a burst
#     # peak_threshold =np.int(np.mean(burst_vectors)+np.std(burst_vectors))


#     # print(np.mean(burst_vectors),np.std(burst_vectors))
#     pop_bursts_final = detect_onests(burst_vectors,na(burst_vectors[:,0]>onset_threshold,dtype=int),peak_threshold) 
#     return pop_bursts_final
    
# @jit(nopython=True)
# def detect_onests(burst_vectors, thr_burst_vector,peak_threshold=5):
#     """Detect the start and end of the network burst"""
#     # detect the start and end of the network burst
# #     peak_threshold = 5 # number of max simult active electrodes to consider a peak
#     pop_bursts= []
#     df_vec = np.diff(thr_burst_vector)

#     for i,k in enumerate(df_vec):
#         if k>0:
#             start = i
#         elif k<0:
#             end = i
#             pop_bursts.append([start,end])
#     pop_bursts_final =[]
#     for burst in pop_bursts:
#         if np.max(burst_vectors[burst[0]:burst[1],0])>peak_threshold:
#         # if np.sum(burst_vectors[burst[0]:burst[1],0])>peak_threshold:
#             pop_bursts_final.append(burst)
#     return pop_bursts_final



# def fixed_bursted_detection(st,sc,
#                             min_isi = 10,
#                             max_isi = 500,
#                             minBdur = 100,
#                             minIBI = 1000,
#                             minSburst = 50,
#                             maxBdur = 20000,
#                             median_thr = 3,
#                             BI_thr = 0.5,
#                             min_n_bursts = 10
#                             ):
#         """ Wrapper for the MI_bursts function,
#         adds 

#         Args:
#             st (_type_): _description_
#             sc (_type_): _description_
#             min_isi (int, optional): _description_. Defaults to 10.
#             max_isi (int, optional): _description_. Defaults to 500.
#             minBdur (int, optional): _description_. Defaults to 100.
#             minIBI (int, optional): _description_. Defaults to 1000.
#             minSburst (int, optional): _description_. Defaults to 50.
#             maxBdur (int, optional): _description_. Defaults to 20000.
#             median_thr (int, optional): _description_. Defaults to 3.
#             BI_thr (float, optional): _description_. Defaults to 0.5.
#             min_n_bursts (int, optional): _description_. Defaults to 10.

#         Returns:
#             list: detected bursts
#         """
#         isi_thr = min(max(np.mean(np.diff(st)),min_isi),max_isi)
#         bi = bimodality_index(sc)

#         bursts = MI_bursts(st,   
#                 maxISIstart=isi_thr,#100.0,
#                 maxISIb=isi_thr,#min(np.median(np.diff(st)),100),#100.0,
#                 minBdur=minBdur,
#                 minIBI=minIBI,
#                 minSburst=minSburst)
#         bursts = na(bursts)
#         # print(len(bursts))
#         # check size 
#         final_bursts = []
#         # Extra conditions 
#         # max duration 

#         for burst in bursts:
#                 size = len(st[(st>burst[0])*(st<burst[1])])>minSburst
#                 duration = burst[1]-burst[0]<maxBdur
#                 if size and duration:
#                         final_bursts.append(burst)
#         final_bursts = na(final_bursts)/1000
#         if len(final_bursts)>min_n_bursts and np.median(sc)<=median_thr and bi>BI_thr:
#                 return final_bursts
#         else:
#                 return []
    

def read_burst_stat(path):
    """ read the burst stats detected by default 
    Args:
        path (string): path to the csv file with network bursts stats
    """
    #Labels set
    bic_label = ['A2','A3','A4','A5','A6',
            'B2','B3','B4','B5','B6']
    control_label =['A1','B1','C1','D1']
    standard_label = ['C2','C3','C4','C5','C6',
            'D2','D3','D4','D5','D6']

    summary = pd.read_csv(path)
    ibis = []
    durs = []
    for i,labels in enumerate([control_label,bic_label,standard_label]):
        for j,label in enumerate(labels):
                try:
                        burst_times =(summary.filter(items = ['Start timestamp [µs]'])[summary['Well Label']==label])*1e-6
                        burst_durs =(summary.filter(items = ['Duration [µs]'])[summary['Well Label']==label])*1e-6
                        burst_times = na(np.squeeze(burst_times))
                        burst_durs = na(np.squeeze(burst_durs))
                        ends = burst_times+burst_durs
                        ibi = burst_times[1:] - ends[:-1]
                        ibis.append(ibi)#[:50])
                        durs.append(burst_durs)#[:50])
                except:
                        ibis.append(np.nan)
                        durs.append(np.nan)


    return ibis,durs
