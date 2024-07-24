# Script to read attributions from two models and compute their correlation on 
# a cell type specific basis.

import numpy as np
import sys, os
from scipy.stats import pearsonr
import glob
from functools import reduce
from functions import correlation, mse

def isint(x):
    try: 
        int(x)
        return True
    except:
        return False

if __name__ == '__main__':

    filea = sys.argv[1]
    fileb = sys.argv[2]

    outname = os.path.splitext(filea)[0]+'_vs_'+os.path.splitext(os.path.split(fileb)[1])[0]
    if '--outname' in sys.argv:
        outname = sys.argv[sys.argv.index('--outname')+1]

    att1 = np.load(filea, allow_pickle = True)
    att2 = np.load(fileb, allow_pickle = True)

    names1, values1, exp1 = att1['names'], att1['values'], att1['experiments']
    names2, values2, exp2 = att2['names'], att2['values'], att2['experiments']

    common = reduce(np.intersect1d, [names1, names2])
    nsort1 = np.argsort(names1)[np.isin(np.sort(names1), common)]
    nsort2 = np.argsort(names2)[np.isin(np.sort(names2), common)]
    values1, values2 = values1[nsort1], values2[nsort2]

    expcom = reduce(np.intersect1d, [exp1, exp2])
    esort1, esort2 = np.argsort(exp1)[np.isin(np.sort(exp1), expcom)], np.argsort(exp2)[np.isin(np.sort(exp2), expcom)]
    values1, values2 = values1[:,esort1], values2[:,esort2]

    values1, values2 = np.swapaxes(values1, -1,-2), np.swapaxes(values2, -1,-2)
    print(np.shape(values1), np.shape(values2))


    if '--meanrefA' in sys.argv:
        values1 = values1 - np.mean(values1, axis = -1)[...,None]
        print('meanrefA')
    if '--meanrefB' in sys.argv:
        values2 = values2 - np.mean(values2, axis = -1)[...,None]
        print('meanrefB')
        
    if '--meanaltrefA' in sys.argv:
        values1 = values1 - (np.sum(values1, axis = -1)/3)[...,None]
        print('meanaltrefA')
    if '--meanaltrefB' in sys.argv:
        values2 = values2 - (np.sum(values2, axis = -1)/3)[...,None]
        print('meanaltrefB')


    if '--atreference' in sys.argv:
        ref = np.load(sys.argv[sys.argv.index('--atreference')+1], allow_pickle = True)
        rnames, seq = ref['genenames'], ref['seqfeatures']
        if len(seq) != len(rnames):
            seq = seq[0]
        rsort = np.argsort(rnames)[np.isin(np.sort(rnames), common)]
        rnames, seq = rnames[rsort], seq[rsort]
        print(np.shape(seq))
        if np.array_equal(rnames, common):
            values1 = np.sum(values1 * seq[:,None], axis = -1)
            values2 = np.sum(values2 * seq[:,None], axis = -1)
            print(np.shape(values1), np.shape(values2))
            
        else:
            print('No ref for all sequences')
        


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
                elect[e] = list(exp).index(el)
        values1 = values1[:,elect]
        values2 = values2[:,elect]
        expcom = expcom[elect]

    ctype = 'pearson'
    distance = False
    if '--cosine' in sys.argv:
        ctype = 'cosine'
        distance = True

    msqe = False
    if '--mse' in sys.argv:
        msqe = True
        
    allcorr = []
    for e, exp in enumerate(expcom):
        
        if msqe:
            corr = np.around(mse(values1[:,e].reshape(values1.shape[0],-1), values2[:,e].reshape(values2.shape[0],-1), axis = -1, sqrt = False), 8)
        else:
            corr = np.around(correlation(values1[:,e].reshape(values1.shape[0],-1), values2[:,e].reshape(values2.shape[0],-1), axis = -1, ctype = ctype, distance = distance), 4)
        if not '--compute_summary' in sys.argv:
            print(exp, round(np.mean(corr),3))
        if '--savenpz' in sys.argv or '--compute_summary' in sys.argv:
            allcorr.append(corr)
        else:
            np.savetxt(outname+'_'+exp+'.txt', np.array([common, np.around(corr,3)]).T, fmt = '%s')
    if '--savenpz' in sys.argv:
        np.savez_compressed(outname+'.npz', names = common, values = np.array(allcorr).T , experiments = expcom)
    if '--compute_summary' in sys.argv:
        print(filea, fileb, np.mean(allcorr), np.std(allcorr))
