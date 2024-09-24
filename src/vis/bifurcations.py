import matplotlib.pyplot as plt
import numpy as np
na = np.array

# def plot_bifurcations(a=5,
#                       add_colors=False,
#                       ax=None,
#                       inverse=False,
#                       path = '../results/bifurcations/diagram_a=%s.txt',
#                       lw = 1,
#                       ):
#     #===============Bifuraction diagram=============
#     # for c=0 
    
#     # if ax is None:
#         # fig,ax = plt.subplots(1,1,figsize=(6,6))
#     bif = np.loadtxt(path%a)
#     bf = np.arange(10,400000,1000)*0.00002
#     # ax.plot(bif[:,1],-bif[:,0],'.')
#     bs = np.arange(0,5,0.1)
#     y1 = 8.1+bs*-8.82
#     y2 = .96+bs*-.208
#     if inverse:
#         plt.plot(-y1,bs,'k',linewidth=lw)
#         plt.plot(-y2,bs,'k',linewidth=lw)
#         plt.xlabel('Excitability ($\\theta$)')
#         plt.ylabel('Adaptation strength (b)')
#     else:
#         plt.plot(bs,-y1,'k',linewidth=lw)
#         plt.plot(bs,-y2,'k',linewidth=lw)
#         plt.ylabel('Excitability ($\\theta$)')
#         plt.xlabel('Adaptation strength (b)')
#     # ax.set_xticks(fontsize=24)
#     # ax.set_yticks(fontsize=24)
#     # plt.fill_between(bs, y1, y2,color='C0',alpha=0.1)
#     # lower_triag = na([[0.81,0.79],[0.9,0.],[4.6,0.]])
#     if add_colors:
#         plt.fill_between(bs, -y1, -y2,color='darkgray')
#         lower_triag = na([[0.88,-0.75],[0.9,0.],[4.6,0.]])
#         plt.fill_between(lower_triag[:,0],lower_triag[:,1],
#                 y2=na([0.88,-0.75,0,]),color='r',alpha=0.5)
#     return None


def plot_bifurcations(inverse =False,
                      a=5,
                      path='/home/ovinogradov/Projects/ReducedBursting/results/bifurcations/diagram_a=5.txt',
                      add_colors=False,ax=None,
                     color = 'gray'
                     
                     ):
    #===============Bifuraction diagram=============

    bif = np.loadtxt(path)
    # bf = np.arange(10,400000,1000)*0.00002
    if ax is None:
        ax =plt.gca()
        
    inds = np.where((np.diff(bif[:,0])<-3) + (np.diff(bif[:,0])>3))[0]
    if inverse:
        ax.plot(bif[:inds[0],1],-bif[:inds[0],0],'-',color=color)
        for i in range(len(inds[:])-1):
            ax.plot(bif[inds[i]+1:inds[i+1],1],-bif[inds[i]+1:inds[i+1],0],color=color)
        ax.plot(bif[inds[-1]+1:,1],-bif[inds[-1]+1:,0],color=color)
    
    else:
        ax.plot(-bif[:inds[0],0],bif[:inds[0],1],color=color)
        for i in range(len(inds[:])-1):
            ax.plot(-bif[inds[i]+1:inds[i+1],0],bif[inds[i]+1:inds[i+1],1],color=color)
        ax.plot(-bif[inds[-1]+1:,0],bif[inds[-1]+1:,1],color=color)
    return None
