import numpy as np
import sys, os
import matplotlib.pyplot as plt
from functools import reduce

def read_counts(df, count_column=1):
    '''
    Reads count files
    '''
    df = open(df, 'r').readlines()
    names = []
    counts = []
    for l, line in enumerate(df):
        if line[0]!= '_':
            line = line.strip().split()
            names.append(line[0])
            counts.append(line[count_column])
    names, counts = np.array(names), np.array(counts)
    counts[counts == 'NA'] = 'nan'
    counts = counts.astype(float)
    sort = np.argsort(names)
    names, counts = names[sort], counts[sort]
    return names, counts

if __name__ == '__main__':
    # Design file
    design = np.genfromtxt(sys.argv[1], dtype = str, skip_header = 1)
    outname = os.path.splitext(sys.argv[1])[0]
    
    fnames = design[:,0]
    dnames = design[:,-1]
    cells = np.array([b.split('_')[1] for b in dnames])

    dtypes = design[:,-3]
    
    if '--datatypes' in sys.argv:
        datatypes = sys.argv[sys.argv.index('--datatypes')+1].split(',')
    else:
        datatype = np.unique(dtypes)
    
    fils = design[:,1]
    
    # Read counts for datatypes, i.e. introns and exons
    names = [[] for i in range(len(datatypes))]
    counts = [[] for i in range(len(datatypes))]
    for r, reg in enumerate(datatypes):
        rfils = fils[dtypes == reg]
        for f, fi in enumerate(rfils):
            print(reg, fi)
            gnames, gcounts = read_counts(fi)
            names[r].append(gnames)
            counts[r].append(gcounts)
    
    cnames = []
    for n, name in enumerate(names):
        # find common names for all exon or all intron files
        cnames.append(reduce(np.intersect1d, name))
        for c, count in enumerate(counts[n]):
            mask = np.isin(name[c], cnames[-1])
            counts[n][c] = count[mask]
        counts[n] = np.array(counts[n]).T
        print(np.shape(counts[n]))
        # exon and intron matrix are generated

    # Just some clearning of the column names
    if '--cleancolumns' in sys.argv:
        dnames = []
        for f, fname in enumerate(ftnames):
            dname = []
            for fn in fname:
                if 'IL' in fn:
                    fn = fn.split('.IL')
                    fn = 'IL'+fn[1].split('.')[0]+'_'+fn[0].replace('.', '')+'_'+fn[1].split('_')[-1]
                else:
                    fn = fn.split('.PBS')
                    fn = 'PBS_'+fn[0].replace('.', '')+'_'+fn[1].split('_')[-1]
                dname.append(fn)
            dnames.append(np.array(dname))
        ftnames = np.concatenate(dnames)

    for d, dtt in enumerate(datatypes):
        ftnames = fnames[dtypes == dtt]
        np.savetxt(outname+'.'+dtt+'.mat.tsv', np.append(cnames[d].reshape(-1,1), np.around(counts[d],3).astype(str), axis = 1), delimiter = '\t', fmt = '%s', header = '\t'.join(ftnames))
        
    
