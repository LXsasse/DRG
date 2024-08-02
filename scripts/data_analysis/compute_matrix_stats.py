import numpy as np
import sys, os 
from scipy.stats import skew
from drg_tools.stats_functions import gini

'''
computes statistics for data matrix
TODO 
Enable selection of matrix axis 
'''


from drg_tools.io_utils import readalign_matrix_files

if __name__ == '__main__':
    
    delimiter = None
    if '--delimiter' in sys.argv:
        delimiter = sys.argv[sys.argv.index('--delimiter')+1]
    
    genes, exp, values = readalign_matrix_files(sys.argv[1], row_name_replace = '.FC.', delimiter = delimiter)
    axis = int(sys.argv[2])
    
    outname = os.path.splitext(sys.argv[1])[0] + '.ax'+str(axis)
    if '--filterclass' in sys.argv:
        classfilter = sys.argv[sys.argv.index('--filterclass')+1]
        if ',' in classfilter:
            classft = classfilter.split(',')
            mask = np.sum([[cf in e for e in exp] for cf in classft], axis = 0) > 0
        else:
            mask = [classfilter in e for e in exp]
        exp, values = exp[mask], values[:,mask]
        print('\n'.join(exp))
        outname += classfilter.replace(',', '-')
        
    
    
    r = 3
    if '--round' in sys.argv:
        r = int(sys.argv[sys.argv.index('--round')+1])
        
    
    std = np.around(np.std(values, axis = axis),r)
    skew = np.around(skew(values, axis = axis),r)
    mean = np.around(np.mean(values, axis = axis),r)
    minx, perc5, perc10, perc25, median, perc75, perc90, perc95, maxx = np.around(np.percentile(values, [0,5,10,25,50,75,90,95,100], axis = axis),r)
    gindex = np.around(gini(values, axis = axis),r)
    coefvar = np.around(std/mean, r)
    inperc90 = np.around(perc95-perc5,r)
    
    if axis == 0:
        outnames = exp
    else:
        outnames = genes
        
    data = np.array([outnames, minx, perc5, perc10, perc25, median, perc75, perc90, perc95, maxx, mean, inperc90, std, skew, coefvar, gindex]).T
    
    header = 'ID\tMin\t5perc\t10perc\t25perc\tMedian\t75perc\t90perc\t95perc\tMax\tMean\t90percDiff\tStd\tSkew\tCoefofVar\tGini'
    np.savetxt(outname+'.stats.tsv', data, header = header, fmt = '%s', delimiter = '\t')
    print('Exp stats mean')
    print(header.split('\t', 1)[1])
    print('\t'.join(np.around(np.mean(data[:,1:].astype(float), axis = 0), r).astype(str)))
    







