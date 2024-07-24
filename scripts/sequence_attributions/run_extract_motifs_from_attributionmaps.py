import numpy as np
import sys, os
import glob

from drg_tools.io_utils import write_meme_file
from drg_tools.motif_analysis import find_motifs

if __name__ == '__main__': 
    
    npzfile = sys.argv[1] # 
    npz = np.load(npzfile, allow_pickle = True)
    names, values, experiments = npz['names'], npz['values'], npz['experiments']
    outname = os.path.splitext(npzfile)[0]
    
    values = np.transpose(values, axes = (0,1,3,2))
    
    seq = np.load(sys.argv[2], allow_pickle = True)
    seqs, snames = seq['seqfeatures'], seq['genenames']
    
    sort = np.argsort(names)[np.isin(np.sort(names), snames)]
    names, values = names[sort], values[sort]
    sort = np.argsort(snames)[np.isin(np.sort(snames), names)]
    snames, seqs = snames[sort], seqs[sort]
    
    cut = float(sys.argv[3])
    maxgap = int(sys.argv[4])
    minsig = int(sys.argv[5])
    norm = sys.argv[6]
    
    outname += '_'+norm+'motifs'+str(cut)+'_'+str(maxgap)+'_'+str(minsig)
    print(outname)
    std = np.sqrt(np.mean(values**2, axis = (-1,-2)))
    if norm == 'condition':
        std = np.mean(std, axis = 0)[None,:, None]
    elif norm == 'seq':
        std = np.mean(std, axis = 1)[:,None, None]
    elif norm == 'global':
        std = np.mean(std)
    else:
        std = np.array(1.)
    
    refatt = np.sum(values*seqs[:,None,:,:], axis = -1)
    stats = refatt/std
    if '--normpwms' in sys.argv and norm not in ['global', 'seq', 'condition']:
        print('normpwms')
        std = np.mean(np.sqrt(np.mean(values**2, axis = (-1,-2))))
        
    
    values = values/std[...,None]
    
    obj = open(outname+'.txt', 'w')
    pwms, pwmnames = [], []
    for n, name in enumerate(names):
        for e, exp in enumerate(experiments):
            motiflocs = find_motifs(stats[n,e], cut, maxgap, minsig)
            for m, ml in enumerate(motiflocs):
                ml = np.array(ml)
                seqname = name+'_'+str(exp)+'_'+str(ml[0])+'-'+str(ml[-1])
                # compute mean, max, loc
                mean = np.mean(refatt[n,e,ml])
                maxs = np.amax(refatt[n,e,ml])
                obj.write(seqname+' '+str(round(mean,3))+' '+str(round(maxs,3))+' '+','.join(ml.astype(str))+'\n')
                pwmnames.append(seqname)
                pwms.append(values[n,e,ml[0]:ml[-1]+1]*np.sign(mean))
    
    np.savez_compressed(outname+'.npz', pwms = pwms, pwmnames = pwmnames)
            
    
    
    
    






    
