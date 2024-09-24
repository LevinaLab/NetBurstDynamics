import matplotlib.pyplot as plt
plt.style.use('default')  
import numpy as np
na =np.array
from scipy.io import loadmat as loadmat
na =np.array
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_bifurcations(a=5,add_colors=False,ax=None):
    """Plot bifurcation diagram for a given value of a"""
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(6,6))
    bif = np.loadtxt('../../results/bifurcations/diagram_a=%s.txt'%a)
    bf = np.arange(10,400000,1000)*0.00002
    # ax_dict['A'].plot(bif[:,1],bif[:,0],'.')
    ax.set_ylabel('Excitability ($\\theta$)')
    ax.set_xlabel('Adaptation strength (b)')
    bs = np.arange(0,5,0.1)
    y1 = 8.1+bs*-8.82
    y2 = .96+bs*-.208
    ax.plot(bs,-y1,'k',linewidth=4)
    ax.plot(bs,-y2,'k',linewidth=4)
    # ax.set_xticks(fontsize=24)
    # ax.set_yticks(fontsize=24)
    # plt.fill_between(bs, y1, y2,color='C0',alpha=0.1)
    # lower_triag = na([[0.81,0.79],[0.9,0.],[4.6,0.]])
    if add_colors:
        plt.fill_between(bs, -y1, -y2,color='darkgray')
        lower_triag = na([[0.88,-0.75],[0.9,0.],[4.6,0.]])
        plt.fill_between(lower_triag[:,0],lower_triag[:,1],
                y2=na([0.88,-0.75,0,]),color='r',alpha=0.5)
    return None