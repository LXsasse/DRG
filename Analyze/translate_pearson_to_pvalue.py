import scipy.special as special
import numpy as np
import sys, os
from statsmodels.stats.multitest import multipletests
from scipy.stats import t

'''
Under the assumption that x and y are drawn from
independent normal distributions (so the population correlation coefficient
is 0), the probability density function of the sample correlation
coefficient r is ([1]_, [2]_):
.. math::
    f(r) = \frac{{(1-r^2)}^{n/2-2}}{\mathrm{B}(\frac{1}{2},\frac{n}{2}-1)}
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


def correlation2pvalue(r, n):
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


if __name__ == '__main__':
    
    '''
    r = 0.6
    n = 20
    print(correlation2pvalue(r,n))
    print(correlation_to_pvalue(r,n))
    print(pvalue_to_correlation(correlation_to_pvalue(r,n),n))
    '''
    
    corrfile = np.genfromtxt(sys.argv[1], dtype = str)
    numsamp = int(sys.argv[2])
    
    pvals = correlation2pvalue(1. - corrfile[:,1].astype(float), numsamp)
    
    if '--correct_multiple' in sys.argv:
        reject, pvals, alpha, alhpab = multipletests(pvals, alpha = 0.01, method = 'fdr_bh')
        
    obj = open(os.path.splitext(sys.argv[1])[0]+'pv.txt', 'w')
    for p, pv in enumerate(pvals):
        obj.write(corrfile[p,0] +' '+str(pv)+'\n')
        if '--printout' in sys.argv:
            print(corrfile[p,0],corrfile[p,1], str(pv))
    print(os.path.splitext(sys.argv[1])[0]+'pv.txt')
    
    




