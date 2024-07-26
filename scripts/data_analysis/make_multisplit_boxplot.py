'''
Create boxplot of different columns in different data matrices
TODO make it more general
'''


import numpy as np
import sys, os
from matplotlib import cm
from functools import reduce 
import matplotlib.pyplot as plt
from drg_tools.plotlib import plot_distribution

from drg_tools.io_utils import sortnames, isint

def read(fi):
    d = np.load(fi)
    e = d['experiments']
    v = d['values']
    n = d['names']
    return n,e,v


if __name__ == '__main__':
    
    files = sys.argv[1].split(',')
    
    names, exp, values = [],[],[]
    for f, fi in enumerate(files):
        n, e, v = read(fi)
        names.append(n)
        exp.append(e)
        values.append(v)
    
    sort = sortnames(names)
    for s, so in enumerate(sort):
        names[s] = names[s][so]
        values[s] = values[s][so]
        
    sort = sortnames(exp)
    for s, so in enumerate(sort):
        exp[s] = exp[s][so]
        values[s] = values[s][:, so]
        
    if '--selectexp' in sys.argv:
        elect = sys.argv[sys.argv.index('--selectexp')+1]
        if ',' in elect:
            elect = elect.split(',')
        else:
            elect = [elect]
        for e, el in enumerate(elect):
            if isint(el):
                elect[e] = int(el)
            else:
                elect[e] = list(exp[0]).index(el)
        nvalues = []
        for e, ex in enumerate(exp):
            exp[e] = ex[elect]
            nvalues.append(values[e][:,elect].T)
        split = len(elect)
        values = np.transpose(nvalues, axes = (1,0,2))
        print(split, exp[0], np.shape(values))
    else:
        for v, val in enumerate(values):
            values[v] = val.flatten()
        split = 1
        print(np.shape(values))
    
    if '--names' in sys.argv:
        pos=sys.argv[sys.argv.index('--names')+1].split(',')
    else:
        # potentially replace these with the performance or the fraction of data sets
        pos = np.arange(len(files))
    
    outname = None
    if '--savefig' in sys.argv:
        outname = sys.argv[sys.argv.index('--savefig')+1]
    
    if '--sort' in sys.argv:
        sorting = np.array( sys.argv[sys.argv.index('--sort')+1].split(',') ,dtype = int)
        values = [values[i] for i in sorting]
        pos = np.array(pos)[sorting]
    
    
    for p, po in enumerate(pos):
        print(po, np.mean(values[p]), np.median(values[p]))

    plot_distribution(values, pos, split = split, outname = outname, ylim = [0,1], xwidth = 0.4, height = 2.5, facecolor='darkgoldenrod', mediancolor = 'k', ylabel = 'R ISM to TISM', savedpi = 350)
    
    
    
    
    
    
    
    
    

