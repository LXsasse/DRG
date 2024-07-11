import numpy as np
import sys, os

def readtomtom(f):
    obj = open(f,'r').readlines()
    names = []
    pvals =[]
    qvals = []
    target = []
    for l, line in enumerate(obj):
        if l > 0 and line[0] != '#':
            line = line.strip().split('\t')
            if len(line) > 5:
                names.append(line[0])
                target.append(line[1])
                pvals.append(line[3])
                qvals.append(line[5])
        
    names = np.array(names)
    target = np.array(target)
    pvals = np.array(pvals, dtype = float)
    qvals = np.array(qvals, dtype = float)
    return names, target, pvals, qvals
    
if __name__ == '__main__':
    
    tomtom = sys.argv[1] # output tsv from tomtom
    tnames, target, pvals, qvals = readtomtom(tomtom)
    
    qvalcut = 0.05
    if '--qval' in sys.argv:
        qvalcut = float(sys.argv[sys.argv.index('--qval')+1])
    
    mask = qvals <= qvalcut
    tnames, target, pvals, qvals = tnames[mask], target[mask], pvals[mask], qvals[mask]
    
    utnames, i_ = np.unique(tnames, return_index = True)
    print('Unique filters', len(utnames))
    
    print('Unique best targets', len(np.unique(target[i_])))
    
    percent = np.percentile(qvals[i_], [5,10,50,90,95])
    print('qvalue distribution 5, 10, 50, 90, 95%', '{0:.2E} {1:.2E} {2:.2E} {3:.2E} {4:.2E}'.format(percent[0], percent[1],percent[2], percent[3], percent[4]))
    
    print('Unique targets', len(np.unique(target)))
    
