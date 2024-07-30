# stats_functions.py

'''
Contains some functions to compute metrics that are often needed for model evaluations. 
Most functions handle np.ndarrays

'''


import numpy as np
import torch
from sklearn import metrics
from scipy.stats import ranksums
from scipy.stats import t
import scipy.special as special

def info_content(pwm, background = 0.25, eps = 1e-16):
    '''
    Parameters
    ---------
    pwm: np.ndarray
        position frequency matrix that sums to one at each position of shape
        (positions, channels)
    background: float, np.ndarray
        background frequency
    '''
    
    ic = pwm + eps
    ic = np.mean(np.sum(-pwm*np.log2(pwm/background) , axis = 1))
    return ic

def relative_absolute(ypred, ytarget, axis = None):
    '''
    computes the absolute difference between ypred and ytarget in relation to ytarget
    ytarget should be > 0
    '''
    ynorm = np.copy(ytarget)
    ynorm[ynorm ==0] = 1
    dist = np.absolute(ytarget - ypred)/ytarget
    dist = np.mean(dist, axis = axis)
    return dist


def RankSum(yp, yt, axis = 0, fmt = 0, log = True):
    rs = ranksums(yp, yt, axis = axis)[fmt]
    if log:
        rs = -np.log10(rs)
    return rs

def AuROC(x, y, axis=0):
    if axis != 0:
        x, y = x.T, y.T
    rs = -np.ones(np.shape(x)[1])
    for i in range(np.shape(x)[1]):
        rs[i] = metrics.roc_auc_score(x[:,i],y[:,i])
    return rs

def AuPR(x,y, axis = 0):
    if axis != 0:
        x, y = x.T, y.T
    rs = -np.ones(np.shape(x)[1])
    for i in range(np.shape(x)[1]):
        rs[i] = metrics.average_precision_score(x[:,i],y[:,i])
    return rs

# mean squared error between two arrays q and p
def mse(q,p, axis = None, sqrt = True):
    ms = np.mean((q-p)**2, axis = axis)
    if sqrt:
        ms = np.sqrt(ms)
    return ms

def correlation(y1, y2, axis = 1, ctype = 'pearson', distance = True, nan_eps = 1e-16):
    '''
    Computes the pearson, or spearman correlation, or cosine along a given axis
    cdist or pearsonr don't provide the option of pairwise computation
    Can be used for 1D, 2D or even multi dimensional arrays.
    '''
    if ctype == 'spearman':
        # instead of values, use the ranks of each entry along the given axis
        y1, y2 = np.argsort(np.argsort(y1, axis = axis), axis = axis), np.argsort(np.argsort(y2,axis = axis), axis = axis)
    if ctype != 'cosine':
        # cosine distance does not subtract the mean
        mean1, mean2 = np.mean(y1, axis = axis), np.mean(y2, axis = axis)
        y1mean, y2mean = y1-np.expand_dims(mean1,axis = axis), y2-np.expand_dims(mean2,axis = axis)
    else:
        y1mean, y2mean = y1, y2
    # denominator of the pearson correlation
    n1, n2 = np.sqrt(np.sum(y1mean*y1mean, axis = axis)), np.sqrt(np.sum(y2mean*y2mean, axis = axis))
    n12 = n1*n2
    # nominator
    y12 = np.sum(y1mean*y2mean, axis = axis)
    
    # denominator corrections in case of 0 denominator
    if isinstance(y12, float):
        if n12/max(n1,n2) < nan_eps:
            n12, y12 = 1., -1.
    else:
        y12[n12/np.amax(np.array([n1,n2]),axis = 0) < nan_eps] = -1.
        n12[n12/np.amax(np.array([n1,n2]),axis = 0) < nan_eps] = 1
    
    corout = y12/n12
    
    # Return distance measure instead of correlatoin or cosine
    if distance:
        corout = 1.-corout
    return np.around(corout,4) 


def dist_measures(x1, x2, distance, similarity = False, axis = 1, summary = None):
    '''
    Unifies different distance metrics
    Can return any of the given distance metrics between two arrays
    Does not deal with nan
    If provided, can return summary statistic of the metrics for all data
    points. 
    '''
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



def correlation2pvalue(r, n):
    
    '''
    
    Under the assumption that x and y are drawn from
    independent normal distributions (so the population correlation coefficient is 0), the probability density function of the sample correlation
    coefficient r is ([1]_, [2]_):
    .. math::
    
    f(r) = "\frac{{(1-r^2)}^{n/2-2}}{\athrm{B}(\frac{1}{2},\frac{n}{2}-1)}"
    
    where n is the number of samples, and B is the beta function.  This
    is sometimes referred to as the exact distribution of r.  This is
    the distribution that is used in `pearsonr` to compute the p-value.
    The distribution is a beta distribution on the interval [-1, 1],
    with equal shape parameters a = b = n/2 - 1.  In terms of SciPy's
    implementation of the beta distribution, the distribution of r is::
        dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)
    The p-value returned by `pearsonr` is a two-sided p-value. The p-value
    roughly indicates the probability of an uncorrelated system
    producing datasets that have a Pearson correlation at least as extreme
    as the one computed from these datasets. More precisely, for a
    given sample with correlation coefficient r, the p-value is
    the probability that abs(r') of a random sample x' and y' drawn from
    the population with zero correlation would be greater than or equal
    to abs(r). In terms of the object ``dist`` shown above, the p-value
    for a given r and length n can be computed as::
        p = 2*dist.cdf(-abs(r))
    When n is 2, the above continuous distribution is not well-defined.
    One can interpret the limit of the beta distribution as the shape
    parameters a and b approach a = b = 0 as a discrete distribution with
    equal probability masses at r = 1 and r = -1.  More directly, one
    can observe that, given the data x = [x1, x2] and y = [y1, y2], and
    assuming x1 != x2 and y1 != y2, the only possible values for r are 1
    and -1.  Because abs(r') for any sample x' and y' with length 2 will
    be 1, the two-sided p-value for a sample of length 2 is always 1.
    
    '''
    
    # As explained in the docstring, the p-value can be computed as
    #     p = 2*dist.cdf(-abs(r))
    # where dist is the beta distribution on [-1, 1] with shape parameters
    # a = b = n/2 - 1.  `special.btdtr` is the CDF for the beta distribution
    # on [0, 1].  To use it, we make the transformation  x = (r + 1)/2; the
    # shape parameters do not change.  Then -abs(r) used in `cdf(-abs(r))`
    # becomes x = (-abs(r) + 1)/2 = 0.5*(1 - abs(r)).  (r is cast to float64
    # to avoid a TypeError raised by btdtr when r is higher precision.)
        
    ab = n/2 - 1
    prob = 2*special.btdtr(ab, ab, 0.5*(1 - abs(np.float64(r))))
    return prob


# add function pvalue2correlation to compute correlation tresholds for pvalues
def correlation_to_pvalue(r,n):
    tt = r* np.sqrt(n-2)/np.sqrt(1-r**2)
    pval = t.sf(np.abs(tt), n-1)*2
    return pval

# t.sf is 1-t.cdf
def pvalue_to_correlation(pval,n):
    tt = t.isf(pval/2, n-1)
    r = np.sqrt(tt**2/(n-2+tt**2))
    return r


