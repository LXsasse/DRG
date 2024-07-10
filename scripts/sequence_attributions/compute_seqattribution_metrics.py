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

    
    

filea = sys.argv[1]

# bottom75
# top10/bottom75
# std of min per position

outname = os.path.splitext(filea)[0]

if '--outname' in sys.argv:
    outname = sys.argv[sys.argv.index('--outname')+1]

att1 = np.load(filea, allow_pickle = True)

names1, values1, exp1 = att1['names'], att1['values'], att1['experiments']


if '--meanref' in sys.argv:
    values1 = values1 - np.mean(values1, axis = -1)[...,None]
    print('meanrefA')

if '--atreference' in sys.argv:
    ref = np.load(sys.argv[sys.argv.index('--atreference')+1], allow_pickle = True)
    rnames, seq = ref['genenames'], ref['seqfeatures']
    if len(seq) != len(rnames):
        seq = seq[0]
    rsort = np.argsort(rnames)[np.isin(np.sort(rnames), names1)]
    rnames, seq = rnames[rsort], seq[rsort]
    if np.array_equal(rnames, common):
        values1 = np.sum(values1 * seq[:,None], axis = -1)
        print(np.shape(values1))
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
            elect[e] = list(exp1).index(el)
    values1 = values1[:,elect]
    
values1 = np.transpose(values1, axes = (0,1,-1,-2))

minvals, botvals, topvals = [],[],[]
for e, exp in enumerate(exp1):
    # sort after the max of four bases
    val = values1[:,e]
    maxval = np.amax(np.absolute(val), axis = -1)
    minval = np.amin(np.absolute(val), axis = -1)
    varval = np.mean(val**2,axis = -1)
    
    minstd = np.sqrt(np.mean(minval**2, axis = -1))
    
    argsort = np.argsort(maxval, axis = -1)
    for s, so in enumerate(argsort):
        varval[s] = varval[s,so]
        
    botstd = np.mean(varval[:, :int(np.shape(varval)[1]*0.75)]**2, axis = -1)
    topstd = np.mean(varval[:, int(np.shape(varval)[1]*0.9):]**2, axis = -1)
    
    minvals.append(minstd)
    botvals.append(botstd)
    topvals.append(topstd/botstd)
    
np.savez_compressed(outname+'botstd.npz', names = names1 , values = np.array(botvals).T , experiments = exp1)
np.savez_compressed(outname+'topstd.npz', names = names1 , values = np.array(topvals).T , experiments = exp1)
np.savez_compressed(outname+'minstd.npz', names = names1 , values = np.array(minvals).T , experiments = exp1)

