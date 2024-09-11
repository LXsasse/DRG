# create_sequence_motif_matrices.py
import numpy as np
import sys, os
import argparse

from drg_tools.io_utils import read_matrix_file, get_indx

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Uses motif cluster file and motif effect file to create sequence to motif matrices')
    parser.add_argument('motif_clusters', type=str, 
                        help='File with motif clusters')
    parser.add_argument('motif_effects', type=str, 
                        help='File with motif effects')
    
    parser.add_argument('--seqname_delimiter', type=str, default = '_',
                        help='Delimiter to split motif name on')
    parser.add_argument('--seqname_inclusion', type=int, default = 2,
                        help='Number of strings that belong to sequences name after split by delimiter')
    
    parser.add_argument('--minimum_size', type=int, 
                        help='Minimum size that clusters need to have to be included', default = 10)
    parser.add_argument('--N_largest', type=int, 
                        help='Number of largest clusters that should be looked at. Default uses minimum_size', default = None)
    parser.add_argument('--minimum_fraction', type=float, 
                        help='Minimum fraction of sequences that should be in cluster. Default uses minimum_size', default = None)
    
    parser.add_argument('--select_conditions', type=str, 
                        help='Selected conditions for statistic', default = None)
    
    parser.add_argument('--motif_values', type=int, 
                        help='Column to choose from the effect file, i.e. either mean=0 or max=1', default = 0)
    parser.add_argument('--motif_statistic', type=str, 
                        help='Which statistic to compute for the matrix, i.e. max, mean, maxdiff across conditions', default = 'max')
    parser.add_argument('--sequence_clusters', type=str, 
                        help='File with sequence clusters to average over sequences', default = None)
    parser.add_argument('--outname', type=str, default = None)
    parser.add_argument('--outfmt', type=str, default = '.npz')
    
    args = parser.parse_args()
    
    
    # Motif clusters
    motif_clusters = np.genfromtxt(args.motif_clusters, dtype = str)
    # Motif effects
    motif_effects = np.genfromtxt(args.motif_effects, dtype = str)
    
    # Sort motif clusters to motif effects
    sortcluster = np.argsort(motif_clusters[:,0])[np.isin(np.sort(motif_clusters[:,0]), motif_effects[:,0])]
    sortmotif = np.argsort(motif_effects[:,0])[np.isin(np.sort(motif_effects[:,0]), motif_clusters[:,0])]
    motif_clusters = motif_clusters[sortcluster]
    motif_names = motif_clusters[:,0]
    motif_clusters = motif_clusters[:,1].astype(int)
    # Select the motif value that should be used
    motif_effects = motif_effects[sortmotif][:,args.motif_values + 1].astype(float)
    
    # Determine unique motif clusters
    unMotclusters, nMotclusters = np.unique(motif_clusters, return_counts = True)
    
    
    # Determine sequences
    seq_names = np.array([args.seqname_delimiter.join(np.array(mn.split(args.seqname_delimiter, args.seqname_inclusion)[:args.seqname_inclusion])) for mn in motif_names])
    conditions = np.array([mn.split(args.seqname_delimiter, args.seqname_inclusion)[args.seqname_inclusion].rsplit('_',1)[0] for mn in motif_names])
    
    unSeqs = np.unique(seq_names)
    unCond = np.unique(conditions)
    
    
    # Determine number of sequences with cluster
    NseqMot = np.zeros(len(unMotclusters))
    for u, um in enumerate(unMotclusters):
        NseqMot[u] = len(np.unique(seq_names[motif_clusters == um]))
    
    if args.outname is None:
        outname = os.path.splitext(args.motif_clusters)[0]
    else:
        outname = args.outname
        
    if args.sequence_clusters is not None:
        outname +='_seqclustmat'+args.motif_statistic
    else:
        outname +='_seqmat'+args.motif_statistic
    
    # Size cut-off for motif clusters
    if args.N_largest is not None:
        arg.minimum_size = -np.sort(-nMotclusters)[args.N_largest]
    elif args.minimum_fraction is not None:
        arg.minimum_size = int(len(unSeqs)*args.minimum_fraction)
    
    if args.select_conditions is not None:
        outname += str(args.select_conditions).replace('musthave=','').replace(',','-')
        unCondindx = get_indx(args.select_conditions, unCond , islist = True)
        fullmask = np.isin(conditions, unCond[unCondindx])
        motif_clusters, motif_effects, seq_names, conditions, motif_names = motif_clusters[fullmask], motif_effects[fullmask], seq_names[fullmask], conditions[fullmask], motif_names[fullmask]
        unSeqs = np.unique(seq_names)
        unCond = np.unique(conditions)
        ucmask = np.isin(unMotclusters, motif_clusters)
        unMotclusters, nMotclusters, NseqMot = unMotclusters[ucmask], nMotclusters[ucmask], NseqMot[ucmask]
    
    if args.minimum_size is not None:
        outname += 'ms'+str(args.minimum_size)
        ucmask = NseqMot >= args.minimum_size
        if np.sum(ucmask) == 0:
            print(f'Minimum size of {args.minimum_size} too high, 0 clusters left')
            sys.exit()
        unMotclusters, nMotclusters, NseqMot = unMotclusters[ucmask], nMotclusters[ucmask], NseqMot[ucmask]
        fullmask = np.isin(motif_clusters, unMotclusters)
        motif_clusters, motif_effects, seq_names, conditions, motif_names = motif_clusters[fullmask], motif_effects[fullmask], seq_names[fullmask], conditions[fullmask], motif_names[fullmask]
        unSeqs = np.unique(seq_names)
        
    
    
    
    effectmatrix = np.zeros((len(unMotclusters), len(unSeqs)))
    for u, um in enumerate(unMotclusters):
        efclust = motif_clusters == um
        efseqs = np.unique(seq_names[efclust])
        for ie, es in enumerate(efseqs):
            e = list(unSeqs).index(es)
            effcond = np.zeros(len(unCond))
            seqmask = seq_names == es
            seqclustmask = seqmask * efclust
            effects = motif_effects[seqclustmask]
            if args.motif_statistic == 'max':
                effectmatrix[u,e] = np.amax(effects)
            else:
                cond = conditions[seqclustmask]
                # sort by size
                sorteff = np.argsort(-np.abs(effects))
                effects, cond = effects[sorteff], cond[sorteff]
                # condind contains the larger entry for a condition
                cond, condind = np.unique(cond, return_index = True)
                effcond[np.isin(unCond, cond)] = effects[condind]
                if args.motif_statistic == 'mean':
                    effectmatrix[u,e] = np.mean(effcond)
                elif args.motif_statistic == 'maxdiff':
                    effectmatrix[u,e] = np.amax(effcond)-np.amin(effcond)
                elif args.motif_statistic == 'sum':
                    effectmatrix[u,e] = np.sum(effcond)
                elif args.motif_statistic == 'presence':
                    effectmatrix[u,e] = np.sign(effcond[np.argmax(np.abs(effcond))])
                elif args.motif_statistic == 'count':
                    effectmatrix[u,e] = len(effcond)
                
    
    
    # Summarize to sequence clusters
    if args.sequence_clusters is not None:
        seq_clusters = np.genfromtxt(args.sequence_clusters, dtype = str)
        common_seqs = np.isin(seq_clusters[:,0], unSeqs)
        seq_clusters = seq_clusters[common_seqs]
        unseqclusters = np.unique(seq_clusters[:,1])
        seqclust_effectmatrix = np.zeros((len(unMotclusters), len(unseqclusters)))
        for u, usc in enumerate(unseqclusters):
            seqmask = np.isin(unSeqs, seq_clusters[seq_clusters[:,1]==usc,0])
            seqclust_effectmatrix[:, u] = np.mean(effectmatrix[:, seqmask], axis = 1)
        effectmatrix = seqclust_effectmatrix
        unSeqs = unseqclusters 
        
            
    # Save output with given outfmt
    print(outname + args.outfmt)
    if args.outfmt == '.npz':
        np.savez_compressed(outname + '.npz', values = effectmatrix, names = unMotclusters.astype(str), columns = unSeqs.astype(str))
    else:
        np.savetxt(outname + '.txt', np.append(unMotclusters.reshape(-1,1).astype(str), np.around(effectmatrix,3), axis =1), header = ' '.join(unSeqs), fmt = '%s')
        
        
        
