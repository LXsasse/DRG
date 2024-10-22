import numpy as np
import argparse
import os

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                    prog='split_motif_cluster_files',
                    description='Combines motifs from different motif files with different prefixes for their names')
    parser.add_argument('motifclusters', type=str, 
                        help='Motifclusters')
    
    args = parser.parse_args()
    
    motifclusters = np.genfromtxt(args.motifclusters, dtype = str)
    
    models  = np.array([mn.split('_',1)[0] for mn in motifclusters[:,0]])
    names = np.array([mn.split('_',1)[1] for mn in motifclusters[:,0]])
    motifclusters[:,0] = names
    
    preout = os.path.split(args.motifclusters)[0]
    if preout != '':
        preout = preout + '/'
    outname = os.path.split(args.motifclusters)[1]
    pre, outname = outname.split('_',1)
    
    umodel = np.unique(models)
    
    for m, mod in enumerate(umodel):
        print(preout+mod+'_'+outname)
        np.savetxt(preout+mod+'_'+outname, motifclusters[models == mod], fmt = '%s')
    
