import numpy as np
import argparse
import os
from drg_tools.plotlib import plot_distribution
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                    prog='compare_attributions_for_seqlet_clusters',
                    description='Compare the attributions of seqlets of two models for the same sequences')
    parser.add_argument('seqclusters', type=str, 
                        help='Cluster assignment')
    parser.add_argument('seqeffects', type = str)
    parser.add_argument('--values', type = str, default = 'mean')
    parser.add_argument('--cluster', type = str, default = None)
    args = parser.parse_args()
    
    outname = os.path.splitext(args.seqeffects)[0] + '_clustered_hist'
    
    clusters = np.genfromtxt(args.seqclusters, dtype =str)
    stats = np.genfromtxt(args.seqeffects, dtype = str)
    
    sort = np.argsort(clusters[:, 0])[np.isin(np.sort(clusters[:,0]), stats[:,0])]
    clusters = clusters[sort]
    
    sort = np.argsort(stats[:,0])[np.isin(np.sort(stats[:,0]), clusters[:,0])]
    stats = stats[sort] 
    
    if np.array_equal(stats[:,0], clusters[:,0]):
        clusters = clusters[:,1]
        
        if args.values == 'mean':
            effects = stats[:, -3].astype(float)
        elif args.values == 'max':
            effects = stats[:, -2].astype(float)
        
        locations = [np.array(l.split(',')).astype(int) for l in stats[:,-1]]
        models = np.array([mod.split('_')[0] for mod in stats[:,0]])
        cells = np.array([mod.split('_')[-2] for mod in stats[:,0]])
        seqlets = np.array([mod.split('_',1)[1].rsplit('_', 1)[0] for mod in stats[:,0]])
        
        unmodels = np.unique(models)
        unclusters = np.unique(clusters)
        
        if args.cluster is None:
            cset = unclusters
        else:
            cset = args.cluster.split(',')
        
        for c, unc in enumerate(cset):
            seqset = clusters == unc
            seqs = seqlets[seqset]
            unseqs = np.unique(seqs)
            data = []
            for u, us in enumerate(unseqs):
                mask = seqset * (seqlets == us)
                mask0 = np.where(mask * (models == unmodels[0]))[0]
                mask1 = np.where(mask * (models == unmodels[1]))[0]
                for m in mask0:
                    mloc = locations[m]
                    haspartner = False
                    if len(mask1) > 0:
                        otherloc = np.concatenate([locations[m1] for m1 in mask1])
                        otherseq = np.concatenate([[m1] * len(locations[m1]) for m1 in mask1])
                        for ml in mloc:
                            if ml in otherloc:
                                haspartner = otherseq[otherloc == ml][0]
                                break
                    if haspartner:
                        data.append([effects[m], effects[haspartner], cells[m]])
                        mask1 = mask1[mask1 != haspartner]
                    else:
                        data.append([effects[m], 0., cells[m]])
                    mask0 = mask0[mask0 != m]
                    # put effects with celltype, into list and remove both from mask
                        
                # then iterate over other mask if there is anything left
                if len(mask1) > 0:
                    for m in mask1:
                        data.append([0., effects[m], cells[m]])
            data = np.array(data, object)
            ccells = data[:,-1].astype(str)
            data = data[:, :-1]
            uncells = np.unique(ccells)
            x = [[],[]]
            for i, uce in enumerate(uncells):
                x[0].append(data[ccells == uce,0])
                x[1].append(data[ccells == uce,1])
            
            plot_distribution(x, uncells, split = 2, swarm = True, facecolor = ['grey', 'indigo'], outname = outname+unc)
            
        
            
                    
                
                
                
                
                
            
            
    
