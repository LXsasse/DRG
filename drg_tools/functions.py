import numpy as np

def relative_absolute(ypred, ytarget, axis = None):
    ynorm = np.copy(ytarget)
    ynorm[ynorm ==0] = 1
    dist = np.absolute(ytarget - ypred)/ytarget
    dist = np.mean(dist, axis = axis)
    return dist


# mean squared error between two arrays q and p
def mse(q,p, axis = None, sqrt = True):
    ms = np.mean((q-p)**2, axis = axis)
    if sqrt:
        ms = np.sqrt(ms)
    return ms


def correlation(y1, y2, axis = 1, ctype = 'pearson', distance = True):
    if ctype == 'spearman':
        y1, y2 = np.argsort(np.argsort(y1, axis = axis), axis = axis), np.argsort(np.argsort(y2,axis = axis), axis = axis)
    if ctype != 'cosine':
        mean1, mean2 = np.mean(y1, axis = axis), np.mean(y2, axis = axis)
        y1mean, y2mean = y1-np.expand_dims(mean1,axis = axis), y2-np.expand_dims(mean2,axis = axis)
    else:
        y1mean, y2mean = y1, y2
    n1, n2 = np.sqrt(np.sum(y1mean*y1mean, axis = axis)), np.sqrt(np.sum(y2mean*y2mean, axis = axis))
    n12 = n1*n2
    y12 = np.sum(y1mean*y2mean, axis = axis)
    if isinstance(y12, float):
        if n12/max(n1,n2) < 1e-16:
            n12, y12 = 1., -1.
    else:
        y12[n12/np.amax(np.array([n1,n2]),axis = 0) < 1e-16] = -1.
        n12[n12/np.amax(np.array([n1,n2]),axis = 0) < 1e-16] = 1
    corout = y12/n12
    if distance:
        corout = 1.-corout
    return np.around(corout,4) 

def dist_measures(x1, x2, distance, similarity = False, axis = 1, summary = None):
    if distance == 'spearman':
        x1 = np.argsort(np.argsort(x1, axis = axis), axis = axis)
        x2 = np.argsort(np.argsort(x2, axis = axis), axis = axis)
        
    if distance == 'pearson':
        x1 -= np.expand_dims(np.mean(x1, axis = axis), axis)
        x2 -= np.expand_dims(np.mean(x2, axis = axis), axis)
    
    if distance == 'cosine' or distance == 'pearson':
        x1 /= np.expand_dims(np.sqrt(np.sum(x1**2, axis = axis)), axis)
        x2 /= np.expand_dims(np.sqrt(np.sum(x2**2, axis = axis)), axis)
    
    
    if distance == 'absolute':
        sim = np.sum(x1 - x2, axis = axis)
    else:
        sim = np.sum(x1*x2, axis = axis)
    
    if distance == 'euclidean':
        np.sum(x1**2, axis = axis)+np.sum(x2**2, axis = axis) - 2*sim
    elif not similarity:
        sim = 1. - sim
    
    if summary is not None:
        if summary == 'mean':
            sim = np.mean(sim)
        if summary == 'sum':
            sim = np.sum(sim)
    return sim
