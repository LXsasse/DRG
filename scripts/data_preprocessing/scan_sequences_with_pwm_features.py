import sys, os 
import numpy as np
import ast

from drg_tools.io_utils import numbertype, check, read_pwm, readin_sequence_return_onehot
from drg_tools.sequence_utils import pwm_scan
    
    
if __name__ is '__main__':

    inputfile = sys.argv[1]
    X, names, features = readin_sequence_return_onehot(inputfile)
    pwmfile = sys.argv[2]

    infcont = False
    if '--infocont' in sys.argv:
        infcont = True

    psam = False
    if '--psam' in sys.argv:
        psam = True

    motcut = None
    motscale = False
    if '--motif_cutoff' in sys.argv:
        motscale = check(sys.argv[sys.argv.index('--motif_cutoff')+1])
        motcut = float(sys.argv[sys.argv.index('--motif_cutoff')+2])
        


    targetlen = None
    if '--targetlen' in sys.argv:
        targetlen = int(sys.argv[sys.argv.index('--targetlen')+1])
        
    pooling_size = None
    pooling_steps = None
    pooling_type = 'Mean'
    if '--pooling' in sys.argv:
        pooling_size = check(sys.argv[sys.argv.index('--pooling')+1])
        pooling_steps = check(sys.argv[sys.argv.index('--pooling')+2])
        pooling_type = sys.argv[sys.argv.index('--pooling')+3]


    pwms, rbpnames = read_pwm(pwmfile, psam = psam, infcont = infcont)

    X = pwm_scan(X, pwms, targetlen = targetlen, motif_cutoff = motcut, set_to = 0., verbose = False, maxscale = motscale, pooling_size = pooling_size, pooling_steps = pooling_steps, pooling_type = pooling_type)

    if pooling_size is not None:
        nrbpnames = []
        for rbpname in rbpnames:
            for i in range(np.shape(X)[-1]):
                nrbpnames.append(rbpname+'_'+str(i*pooling_steps)+'bp')
    else:
        nrbpnames = rbpnames

    X = X.reshape(np.shape(X)[0], -1)
    outname = os.path.splitext(inputfile)[0]+ '_on_'+ os.path.splitext(os.path.split(pwmfile)[1])[0] + '-psam'+str(psam)[0]+'-ic'+str(infcont)[0]+'_motcut'+str(motscale)[0]+str(motcut)+ '_pooling'+pooling_type+str(pooling_size)+'-'+str(pooling_steps)

    print('Saved as', outname)
    np.savez_compressed(outname, seqfeatures = (X, nrbpnames), genenames = names)
    
    
    
    
