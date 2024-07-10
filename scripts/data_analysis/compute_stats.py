import numpy as np
import sys, os 
from scipy.stats import skew

def read_file(f, expdelim = None, replace = '_', delimiter = None):
    f = open(f, 'r').readlines()
    
    experiments = None
    genes = []
    values = []
    for l, line in enumerate(f):
        if l == 0 and line[0] == '#':
            experiments = line.strip('#').strip().split(delimiter)
        else:
            line = line.strip().split(delimiter)
            genes.append(line[0].strip('"'))
            values.append(line[1:])
            
    if experiments is not None:
        for e, exp in enumerate(experiments):
            experiments[e] = exp.strip('"')
            if expdelim is not None:
                experiments[e] = exp.replace(expdelim, replace)
    
    return np.array(experiments), np.array(genes), np.array(values, dtype = float)


if __name__ == '__main__':
    
    delimiter = None
    if '--delimiter' in sys.argv:
        delimiter = sys.argv[sys.argv.index('--delimiter')+1]
    
    exp, genes, values = read_file(sys.argv[1], expdelim = None, replace = '.FC.', delimiter = delimiter)
    print(exp, genes, np.shape(values))
    
    outname = os.path.splitext(sys.argv[1])[0]
    if '--filterclass' in sys.argv:
        classfilter = sys.argv[sys.argv.index('--filterclass')+1]
        mask = [classfilter in e for e in exp]
        exp, values = exp[mask], values[:,mask]
        print(exp)
        outname += classfilter
    

    stdgenes = np.around(np.std(values, axis = 1),3)
    skewgenes = np.around(skew(values, axis = 1 ),3)
    maxgenes = np.around(np.amax(values, axis = 1 ),3)
    mingenes = np.around(np.amin(values, axis = 1 ),3)
    meangenes = np.around(np.mean(values, axis = 1 ),3)
    mediangenes = np.around(np.median(values, axis = 1 ),3)
    median75genes = np.around(np.percentile(values, 75, axis = 1 ),3)
    median90genes = np.around(np.percentile(values, 90, axis = 1 ),3)
    median95genes = np.around(np.percentile(values, 95, axis = 1 ),3)
    
    stdclass = np.around(np.std(values, axis = 0),3)
    skewclass = np.around(skew(values, axis = 0 ),3)
    maxclass = np.around(np.amax(values, axis = 0 ),3)
    minclass = np.around(np.amin(values, axis = 0 ),3)
    meanclass = np.around(np.mean(values, axis = 0 ),3)
    medianclass = np.around(np.median(values, axis = 0 ),3)
    median75class = np.around(np.percentile(values, 75, axis = 0 ),3)
    median90class = np.around(np.percentile(values, 90, axis = 0 ),3)
    median95class = np.around(np.percentile(values, 95, axis = 0 ),3)
    
    data = np.array([exp, medianclass, median75class, median90class, median95class, meanclass, minclass, maxclass, maxclass-minclass, stdclass, skewclass]).T
    header = 'Exp\tMedian\t75perc\t90perc\t95perc\tMean\tMin\tMax\tMaxDiff\tStd\tSkew'
    np.savetxt(outname+'expstats.tsv', data, header = header, fmt = '%s', delimiter = '\t')
    print('Exp stats mean')
    print(header.split('\t', 1)[1])
    print('\t'.join(np.around(np.mean(data[:,1:].astype(float), axis = 0), 3).astype(str)))
    
    data = np.array([genes, mediangenes, median75genes, median90genes, median95genes, meangenes, mingenes, maxgenes, maxgenes-mingenes, stdgenes, skewgenes]).T
    np.savetxt(outname+'genestats.tsv', data, header = header, fmt = '%s', delimiter = '\t')
    print('Exp gene mean')
    print(header.split('\t', 1)[1])
    print('\t'.join(np.around(np.mean(data[:,1:].astype(float), axis = 0), 3).astype(str)))

    







