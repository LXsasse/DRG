import numpy as np
import argparse
from scipy.stats import fisher_exact
import os

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                    prog='enrichtment_motifs_in_models',
                    description='Computes the enrichment of motifs in models with fishers exact')
    parser.add_argument('clusterfile', type=str, 
                        help='Joint cluster file from joint clustering after merging seqlets')
    
    parser.add_argument('--outname', type = str, default = None)
    parser.add_argument('--select_set_less', type = float, default = None)
    parser.add_argument('--select_set_greater', type = float, default = None)
    
    args = parser.parse_args()
    
    if args.outname is None:
        args.outname = os.path.splitext(args.clusterfile)[0]
    
    clusters = np.genfromtxt(args.clusterfile, dtype = str)
    models = np.array([c.split('_')[0] for c in clusters[:,0]])
    clusters = clusters[:,1]
    
    uclusters, Nclusters = np.unique(clusters, return_counts = True)
    umodels, Nmodels = np.unique(models, return_counts = True)
    
    stats = []
    for c, cl in enumerate(uclusters):
        mask = clusters == cl
        cmodel = models[mask]
        is0 = np.sum(cmodel == umodels[0])
        is1 = np.sum(cmodel == umodels[1])
        contmat = [[is0, is1], [Nmodels[0]-is0, Nmodels[1]-is1]]
        stat, p = fisher_exact(contmat, alternative = 'less')
        statg, pg = fisher_exact(contmat, alternative = 'greater')
        stats.append([cl, is0, is1, pg, p])
    
    np.savetxt(args.outname+'_pstats.txt', np.array(stats), fmt = '%s')
    
    if args.select_set_less is not None:
        ps = np.array(stats, dtype = object)
        the_set = uclusters[ps[:,-1].astype(float) <= args.select_set_less]
        print(f'{len(the_set)} clusters are enriched for  {args.select_set_less} with model {umodels[1]}')
        np.savetxt(args.outname+'_lessthan'+str(args.select_set_less)+'.txt', the_set, fmt = '%s')
   
    if args.select_set_greater is not None:
        ps = np.array(stats, dtype = object)
        the_set = uclusters[ps[:,-2].astype(float) <= args.select_set_greater]
        print(f'{len(the_set)} clusters are enriched with {args.select_set_greater} with model {umodels[0]}')
        np.savetxt(args.outname+'_greaterthan'+str(args.select_set_greater)+'.txt', the_set, fmt = '%s')
    
    
