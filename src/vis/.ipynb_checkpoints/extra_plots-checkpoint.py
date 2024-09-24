import matplotlib.pyplot as plt

def shaded_error(positions,mean_x,error,color,label=None):
    plt.plot(positions, mean_x, '-',color=color,label=label)
    plt.fill_between(positions, mean_x-error, mean_x+error,color=color,alpha=0.2)
    return None