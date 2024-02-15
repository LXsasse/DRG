import numpy as np
import sys, os
import glob
from functools import reduce

if __name__ == '__main__':
    results = sys.argv[1]
    outname = sys.argv[2]
    if ',' in results:
        results = results.split(',')
    elif '^' in results:
        results = np.sort(glob.glob(results.replace('^', '*')))
    else:
        print('Cannot create list of files with this input, either use , or ^')
        sys.exit()

    print(len(results))
    if outname in results:
        results = np.array(results)
        results = results[results != outname]

    # read in result files
    data = []
    names = []
    for r, re in enumerate(results):
        print(r, 'Read', re)
        dat = np.genfromtxt(re, dtype = object)
        dat = dat[np.argsort(dat[:,0])]
        names.append(dat[:,0].astype(str))
        data.append(dat[:,1].astype(float))
    
    common = reduce(np.intersect1d, names)
    for r, re in enumerate(results):
        mask = np.isin(names[r], common)
        data[r] = data[r][mask]
    
    data = np.array(data)
    print(np.shape(data))
    
    avg = np.mean(data, axis = 0)
    print(len(avg))
    
    np.savetxt(outname, np.array([common, np.around(avg,4).astype(str)]).T, fmt = '%s')
    
        
    
