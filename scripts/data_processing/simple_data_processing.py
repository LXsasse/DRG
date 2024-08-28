import numpy as np
import sys, os
import matplotlib.pyplot as plt
from functools import reduce



def quantile_norm(x, axis = 0):
    '''
    Performs quantile normalization along axis
    '''
    meanx = np.mean(np.sort(x, axis = axis), axis = axis-1)
    xqtrl = meanx[np.argsort(np.argsort(meanx,axis = axis), axis = axis)]
    return xqtrl

def median_fraction_norm(x, pseudo = 0):
    '''
    Performs median fraction norm
    '''
    # use geometric mean
    d = data + pseudo
    mj = np.prod(d**(1/np.shape(d)[1]), axis=1)
    r = d/mj[:,None]
    r = np.nan_to_num(r,nan=1)
    si = np.median(r,axis = 0)
    print('Median fractions', si)
    data = data/si
    return data


def readgtf(f, add_to_type = 'ic', name_id = 'gene_id'):
    '''
    Read gtf file for normalization
    IMPORTANT: Use only gtf with canonical set of exons and introns for each transcript
    TODO
    Search for canonical exons and introns in gtf file if not given
    Returns
    -------
    gene_id, 
    type : exon, intron
    totlen : sum over all end-start
    '''
    obj = open(f, 'r').readlines()
    names = []
    types = []
    length = []
    for l, line in enumerate(obj):
        if line[0] != '#':
            line = line.strip().split('\t')
            names.append(line[8].replace(name_id+' "', '').split('";')[0])
            types.append(line[2]+add_to_type)
            length.append(int(line[4]) - int(line[3]))
    names, types, length = np.array(names), np.array(types), np.array(length)
    
    # Join the length of each gene
    unames, utypes = np.unique(names), np.unique(types)
    if (length > 0).all():
        print('lengths okay')
    else:
        print(names[length <= 0], types[length <= 0], length[length <= 0])
    
    totlen = np.zeros((len(utypes), len(unames)), dtype = int)
    for n, na in enumerate(unames):
        for t, ut in enumerate(utypes):
            mask = (types == ut) * (names == na)
            totlen[t,n] = np.sum(length[mask])
    # names will be sorted alphabetically 
    return unames, utypes, totlen


def mean_counts(counts, columns, rep_suffix = '_', geometric = False):
    columns = np.array([c.rsplit(rep_suffix,1)[0] for c in columns])
    newcolumns = np.unique(columns)
    newcounts = -np.ones((len(counts), len(newcolumns)))
    for c, nc in enumerate(newcolumns):
        mask = columns == nc
        if geometric:
            newcounts[:,c] = np.sqrt(np.prod(counts[:, mask]**1/np.sum(mask), axis = 1))
        else:
            newcounts[:,c] = np.mean(counts[:, mask], axis = 1)
    return newcolumns, newcounts
    

'''
This script reads combined count files and normalizes, takes the mean, and 
transforms them for subsequent usage in model training 
# Importantly, it performs these steps on a concatenated matrix of introns
and exons so that entries form the same experiment are normalized equally
That means that introns are treated as their own transcipts.

'''


if __name__ == '__main__':

    intronic = sys.argv[1]
    exonic = sys.argv[2]
    
    intoutname = os.path.splitext(intronic)[0]
    exoutname = os.path.splitext(exonic)[0]
    
    introns = np.genfromtxt(intronic, dtype = str)
    exons = np.genfromtxt(exonic, dtype = str)
    
    intnames = introns[:,0]
    exnames = exons[:,0]
    
    intcounts = introns[:,1:].astype(float)
    excounts = exons[:,1:].astype(float)
    
    intcolumns = np.array(open(intronic, 'r').readline().strip('#').strip().split())[-np.shape(intcounts)[-1]:]
    excolumns = np.array(open(exonic, 'r').readline().strip('#').strip().split())[-np.shape(excounts)[-1]:]
    
    # make sure that columns are aligned
    if not np.array_equal(intcolumns, excolumns):
        comcolumns = np.intersect1d(intcomumns, excolumns)
        sorti = np.argsort(intcolumns)[np.isin(np.sort(intcolumns),comcolumns)]
        sorte = np.argsort(excolumns)[np.isin(np.sort(excolumns),comcolumns)]
        intcolumns, excolumns = intcolumns[sorti], excolumns[sorte]
        intcounts, excounts = intcounts[:, sorti], excounts[:, sorte]
    
    # sort names to use masks more easily
    sorti = np.argsort(intnames)
    sorte = np.argsort(exnames)
    intnames, exnames = intnames[sorti], exnames[sorte]
    intcounts, excounts = intcounts[sorti], excounts[sorte]
    
    if '--datatypes' in sys.argv:
        datatypes = sys.argv[sys.argv.index('--datatypes')+1].split(',')
    else:
        datatypes = ['intronic', 'exonic']
    
    dtypes = np.concatenate([[datatypes[0]]*len(intnames), [datatypes[1]]*len(exnames)])
    
    
    add = ''
    if '--TPM' in sys.argv:
        add = '.tpm'
        nnames, ntypes, nlen = readgtf(sys.argv[sys.argv.index('--TPM')+1])
        print(nnames, ntypes, nlen)
        
        exnamemask = np.isin(nnames, exnames)
        intnamemask = np.isin(nnames, intnames)
        
        if np.array_equal(nnames[exnamemask], exnames) and np.array_equal(nnames[intnamemask], intnames):
            divex = nlen[list(ntypes).index('exonic')][exnamemask]
            divint = nlen[list(ntypes).index('intronic')][intnamemask]
            excounts /= divex[:,None]
            intcounts /= divint[:,None]
            excounts *= 1e3
            intcounts *= 1e3
            totdiv = (np.sum(excounts, axis = 0) + np.sum(intcounts, axis = 0)) / 1e6
            excounts /= totdiv[:,None]
            intcounts /= totdiv[:,None]
        else:
            print('Different names in gtf and count matrix', len(nnames[intnamemask]), len(intnames), len(nnames[exnamemask]), len(exnames), 'tpm not possible')
            sys.exit()
            

    # normalize the count matrices
    elif '--median_fraction' in sys.argv:
        add = '.medianfrac'# median fraction norm uses geometric mean
        counts = np.concatenate([intcounts, excounts], axis=0)
        counts = median_fraction_norm(counts)
        intcounts, excounts = counts[:len(intnames)], counts[len(intnames):]
    
    if '--transform' in sys.argv:
        # Provide all transformations in order: 
        # i.e mean, geomean, log2, add=X, quantile
        # --> currently we use mean,add=1,log2
        transformations = sys.argv[sys.argv.index('--transform')+1]
        if ',' in transformations:
            transformations = transformations.split(',')
        else:
            transformations = [transformations]
            
        for trfm in transformations:
            outname += '.'+trfm.replace('=','')
            if trfm == 'mean':
                # replicates should be in in name as _repX or #_repX
                excolumns, excounts = mean_counts(excounts, excolumns)
                intcolumns, intcounts = mean_counts(intcounts, intcolumns)
            elif trfm == 'geomean':
                # replicates should be in in name as _repX or #_repX
                excolumns, excounts = mean_counts(excounts, excolumns, geometric = True)
                intcolumns, intcounts = mean_counts(intcounts, intcolumns, geometric = True)
            elif 'add' in trfm:
                offset = float(trfm.split('=')[-1])
                excounts += offset
                intcounts += offset
            elif trfm == 'log2':
                excounts[excounts <=0]=1e-8
                excounts = np.log2(excounts)
                intcounts[intcounts <=0]=1e-8
                intcounts = np.log2(intcounts)
            elif trfm == 'quantile':
                excounts = quantile_norm(excounts)
                intcounts = quantile_norm(intcounts)
        
    np.savetxt(intoutname + add + '.tsv', np.append(intnames[c].reshape(-1,1), np.around(intcounts,3).astype(str), axis = 1), delimiter = '\t', fmt = '%s', header = '\t'.join(intcolumns))
    np.savetxt(exoutname + add + '.tsv', np.append(exnames[c].reshape(-1,1), np.around(excounts,3).astype(str), axis = 1), delimiter = '\t', fmt = '%s', header = '\t'.join(excolumns))

















