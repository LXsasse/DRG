import numpy as np
import sys, os
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                    prog='combine_motifeffects_from_models',
                    description='Combines motifs effect files from different models with different prefixes for their names')
    parser.add_argument('effectfiles', type=str, 
                        help='Names of seqlet effect files, separated by , ')
    parser.add_argument('--prefixes', type=str, 
                        help='Prefixes for seqlets from different files separated by ","', default = None)
    parser.add_argument('--outname', type = str, default = None)
    
    args = parser.parse_args()
    
    efffiles = args.effectfiles.split(',')
    
    if args.prefixes is not None:
        prefixes = args.prefixes.split(',')
    else:
        prefixes = np.arange(len(efffiles), dtype = int).astype(str)
    
    if args.outname is None:
        args.outname = '-'.join(np.array(prefixes))+'_'+os.path.splitext(os.path.split(efffiles[0])[1])[0]
    
    data = []
    for m, mef in enumerate(efffiles):
        inf = np.genfromtxt(mef, dtype = str)
        names = np.array([prefixes[m] + '_' +mn for mn in inf[:,0]])
        inf[:,0] = names
        data.append(inf)
        
    print(args.outname)
    np.savetxt(args.outname+'.txt', np.concatenate(data, axis = 0), fmt = '%s')
    
        
    
