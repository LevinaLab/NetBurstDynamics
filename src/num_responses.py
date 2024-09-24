""" code to get the response curves numerically 


"""
from src.dynamics_Model2 import StochSim_o
import matplotlib.pyplot as plt
import numpy as np
na = np.array

from src.helpers import expand_params,spike_burst_det

burst_detection_params={'maxISIstart':10.,
                    'maxISIb':10.,
                    'minBdur':17.,
                    'minIBI':20.,
                    'minSburst':17.,
                    'scale':10.
                    }
def print_summary(x_dyn,t_dyn):
    cut = 5000000
    dt= np.diff(t_dyn[cut::100])[0]    
    bursts= na(spike_burst_det(x_dyn[cut::100],dt,burst_detection_params))

    # bursts = bursts+0.05*cut

    # res.append([t_dyn[cut:],x_dyn[cut:],w_dyn[cut:]])
    # print(len(bursts))
    if len(bursts)>2:
        durs = np.diff(bursts)/1000
        ibis =(bursts[1:,0]-bursts[:-1,1])/1000
        mibi = np.mean(ibis)
        mdur = np.mean(durs)
        cv_ibis = np.std(ibis)/mibi
        print('----Bursts-----')
        print(mibi,cv_ibis,mdur)

        print('----Ratio-----')
        print(mdur/mibi)
    else:
        print('No bursts')

    return None


import numpy as np
import multiprocessing as mp

def simulate_for_initial_conditions(params, w0, x0, stimuli, n_rep,stim_len=10):
    responses = []
    for mu in stimuli:
        response = []
        for n in range(n_rep):
            params['w0'] = w0
            params['x0'] = x0
            np.random.seed(123456789 + n)
            params['T'] = stim_len  # 200
            params['mu'] = mu
            t_dyn, x_dyn, w_dyn = StochSim_o(params, torch=0)
            response.append(np.max(x_dyn[:]) > params['J'])  # 2000
        responses.append(np.sum(response) / len(response))
    return responses
    
def get_reponse_curve(params, n_ws=50, stimuli=None, n_rep=50):
    """Collect the responses at different steady levels of adaptation.
    The stimulus is applied as a step function.

    Args:
        n_ws (int): Number of adaptation values to check.
        simuli (array, optional): Stimuli values. If None, defaults to np.linspace(0, 5, 100).
        n_rep (int): Number of repetitions.

    Returns:
        np.ndarray: Response curve.
    """
    t = np.arange(0, params['T'], params['dt'])
    if stimuli is None:
        stimuli = np.linspace(0, 5, 100)
    if params['mu'] != 0:
        raise Warning('Input is non-zero')

    response_curve = np.zeros((n_ws, len(stimuli)))  # *np.nan
    # Preheat the simulations
    np.random.seed(123456789)
    t_dyn, x_dyn, w_dyn = StochSim_o(params, torch=0)
    x_dyn = x_dyn[len(t)//2:]
    t_dyn = t_dyn[len(t)//2:]
    w_dyn = w_dyn[len(t)//2:]

    print('---Burst Summary---')
    print_summary(x_dyn, t_dyn)
    print('------')

    # Values in the down state
    x_mask = x_dyn < ((params['J']/2)-0.1)

    # Take n_ws random values for adaptation
    indis = np.random.choice(np.arange(np.sum(x_mask)), n_ws)
    # Get the values of xs and ws to use them as initial conditions later
    w_sel = w_dyn[x_mask][indis]
    x_sel = x_dyn[x_mask][indis]

    # Prepare for parallel execution
    pool = mp.Pool(mp.cpu_count())

    # Parallelize the outer loop
    results = pool.starmap(simulate_for_initial_conditions, [(params.copy(), w_sel[i], x_sel[i], stimuli, n_rep) for i in range(len(w_sel))])

    pool.close()
    pool.join()

    # Fill the response_curve with the results
    for i, result in enumerate(results):
        response_curve[i, :] = result
    
    return response_curve,w_sel

