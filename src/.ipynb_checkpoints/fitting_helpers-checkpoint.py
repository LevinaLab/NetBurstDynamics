import numpy as np
na = np.array
def shift(arr, num, fill_value=np.nan):
    """https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array

    Args:
        arr ([type]): [description]
        num ([type]): [description]
        fill_value ([type], optional): [description]. Defaults to np.nan.

    Returns:
        [type]: [description]
    """
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

# y = y.detach().numpy()
def nplusmap(y,n_next =1,type= 'hist',bins = None):
    """return unique y(t) and y(t+n)

    Args:
        y ([type]): dynamics 
        n_next (int, optional): number of steps in the future or past Defaults to 1.
    Returns:
        [tuple]: [description]
    """
    y_next= []
    if type='all_values':
        y_unique = np.sort(np.unique(y))
        for y_n in y_unique:
            mask = shift(y==y_n,n_next,False)
            y_next.append(y[mask])
        return y_unique, y_next
    elif type =='hist':
        _,bin_edges= np.histogram(y,bins)
        for i,edge in enumerate(bin_edges[:-1]):
            mask = (y>edge)*(y<bin_edges[i+1])
            y_next.append(y[mask])
        

