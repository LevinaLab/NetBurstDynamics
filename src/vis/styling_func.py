

#Stores some constants used for measuring
class Measures:
    fig_w_1col = 11.0 #cm
    fig_w_2col = 21.0 #cm

# -------------------------------------- #
#      Adjusted by O.V from 

#       Victor Buendia's master file      #
#    for formatting matplotlib graphs in #
#       a cool way. GPLv3 licensed       #
# Thanks Victor!                         #
# -------------------------------------- #

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
na = np.array


colours = {
        #types
        'ctx':na([234,138,119])/255,
        'hpc':na([135,170,205])/255,
        'iPSC':na([189,157,75])/255,
        # States
        'osc':na([223 ,193, 234])/255,
        'exc':na([144 ,192. , 173])/255,
        'bis':na([169, 169,169])/255
          }



def color_violin(v,color):
    """ 
    Change the violin colors
    """
    # first just the bodies 
    for pc in v['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)

    for partname in v:#,'cmeans','cmedians'):
        if partname!='bodies':
            vp = v[partname]
            vp.set_edgecolor(color)
            vp.set_linewidth(1)
            
    return None

def to_inches(cm):
    """
    Convert cm to inches
    """
    return cm/2.54

def one_col_size(ratio=1.618, height=None):
    """
    Returns a tuple (w,h), where w is the width if a single-column graph.
    Height by default is the golden ratio of the width, but can be chosen (in cm)
    
    Parameters:
    - ratio: float
        How large width is with respect to height. This parameter is ignored if height is not None
    - height: float, cm
        Height of the figure. If different from None (default), ratio parameter is ignored.
    """
    width = to_inches(Measures.fig_w_1col)

    if height == None:
        height = width/ratio
    else:
        height = to_inches(height)
    return (width, height)


def two_col_size(ratio=1.618, height=None):
    """
    Returns a tuple (w,h), where w is the width if a double-column graph.
    Height by default is the golden ratio of the width, but can be chosen (in cm)
    
    Parameters:
    - ratio: float
        How large width is with respect to height. This parameter is ignored if height is not None
    - height: float, cm
        Height of the figure. If different from None (default), ratio parameter is ignored.
    """

    width = to_inches(Measures.fig_w_2col)
    if height == None:
        height = width/ratio
    else:
        height = to_inches(height)
    
    return (width, height)
