import numpy as np
import sys, os

from drg_tools.io_utils import numbertype, readtomtom, read_meme

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
    
    if '--tfassignment_file' in sys.argv:
        
        meme = sys.argv[sys.argv.index('--tfassignment_file')+1]
        pwms, pwmnames, nts = read_meme(meme)
        pwmnames = np.array(pwmnames, dtype = utnames.dtype)
        hastf=np.isin(pwmnames, utnames)
        np.savetxt(os.path.splitext(tomtom)[0]+'.assign.txt', np.array([np.arange(len(pwmnames)), hastf.astype(int)]).T.astype(str), fmt = '%s')
        
        
            
            
            
        
        
