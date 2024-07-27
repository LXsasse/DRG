'''
Uses cluster assignment and kernel effect matrix to subcluster initial clusters
based on correlation between kernel matrices
'''

import numpy as np
import sys, os
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

from drg_tools.stats_functions import correlation_to_pvalue, pvalue_to_correlation
from drg_tools.io_utils import read_matrix_file

import argparse



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('clusterfile', type = str, help = 'File that contains the names and assigned clusters')
    parser.add_argument('importance_matrices', type = str, help = 'With comma connected files that contain effect matrices')
    #importance_matrices = sys.argv[2] # with comma connected files
    # output corrected clusterfile, and summarized importance matrix
    parser.add_argument("--mincorr", required = False, type=float, default=0.7, help='The correlation distance cutoff, i.e 1-correlation cutoff, default is 0.7')
    parser.add_argument('--outname', required = False, type = str, default = None, help='Define name of output file')
    
    args = parser.parse_args()
    
    if args.outname is None:
        args.outname = os.path.splitext(args.clusterfile)[0]
        
    pvalue = False

    clusters = np.genfromtxt(args.clusterfile, dtype = str)
    cnames, cfiles = np.array([cl.rsplit('_',1)[0] for cl in clusters[:,0]]), np.array([cl.rsplit('_',1)[-1] for cl in clusters[:,0]])

    if ',' in args.importance_matrices:
        importance_matrices = args.importance_matrices.split(',')
    else:
        importance_matrices = [args.importance_matrices]

    # make sure that order in clusterfile is the same as the matrix files. 
    imp_matrix, imp_matnames, imp_matstats, imp_matclusters = [], [], [], []
    for c, cf in enumerate(np.unique(cfiles)):
        print(cf, importance_matrices[c])
        names, header, matrix = read_matrix_file(importance_matrices[c], data_start_column = 2)
        
        if np.array_equal(cnames[cfiles == cf], names):
            imp_matrix.append(matrix)
            imp_matnames.append(cnames[cfiles == cf])
            imp_matstats.append(cfiles[cfiles == cf])
            imp_matclusters.append(clusters[cfiles == cf,1].astype(int))
        else:
            print('names in cluster file not the same as in matrix file')
            sys.exit()
        if c == 0:
            checkheader = header
        elif not np.array_equal(header, checkheader):
            print('Cell types different in matrices, check selection', header, checkheader)
            sys.exit()

    imp_matrix = np.concatenate(imp_matrix, axis = 0)
    imp_matnames = np.concatenate(imp_matnames, axis = 0)
    imp_matstats = np.concatenate(imp_matstats, axis = 0)
    imp_matclusters = np.concatenate(imp_matclusters, axis = 0)

    if pvalue:
        args.mincorr = 1.-pvalue_to_correlation(args.mincorr, np.shape(imp_matrix)[-1])
    
    print(np.shape(imp_matrix))
    print('Correlation cutoff is', args.mincorr)

    unclusters = np.unique(imp_matclusters)
    print('Initial clusters', len(unclusters), 'in', len(imp_matclusters))

    newclusters = -np.ones(len(imp_matclusters), dtype = int)
    newmatrix = []
    clu = 0
    for u, unc in enumerate(unclusters):
        mask = np.where(imp_matclusters == unc)[0]
        if len(mask) > 1:
            
            dist = cdist(imp_matrix[mask], imp_matrix[mask], 'correlation')
            clustering = AgglomerativeClustering(n_clusters = None, metric = 'precomputed', linkage = 'complete', distance_threshold = args.mincorr).fit(dist)
            newclust = clustering.labels_
            unnew = np.unique(newclust)
            
            for n, nc in enumerate(unnew):
                newclusters[mask[newclust == nc]] = clu
                newmatrix.append(np.mean(imp_matrix[mask[newclust == nc]], axis = 0))
                clu += 1
        else:
            newclusters[mask[0]] = clu
            newmatrix.append(imp_matrix[mask[0]])
            clu += 1
    
    print('Final clusternumber', clu, np.shape(newmatrix))
    newmatrix = np.array(newmatrix)
    uniquenewclust, Nuniquenew = np.unique(newclusters, return_counts = True)
    print('Final clusters', len(uniquenewclust), 'in', len(imp_matclusters))
    
    assert clu == len(uniquenewclust)
    
    print(np.shape(newmatrix))

    impmatrixtype = importance_matrices[0].rsplit('_',1)[-1]

    np.savetxt(args.outname+'_cl'+os.path.splitext(impmatrixtype)[0]+'.txt', np.array([clusters[:,0], newclusters.astype(int)]).T, fmt = '%s')

    uclust, nclust = np.unique(newclusters, return_counts = True)
    nhist, yhist = np.unique(nclust, return_counts = True)
    for n, nh in enumerate(nhist):
        print(nh, yhist[n])

    np.savetxt(args.outname+'_clm'+os.path.splitext(impmatrixtype)[0]+'.dat', np.concatenate([uniquenewclust.astype(int).astype(str).reshape(-1,1), Nuniquenew.astype(int).astype(str).reshape(-1,1), newmatrix],axis = 1), fmt = '%s', header = 'Kernelcluster N_kernels '+' '.join(header))

    print(args.outname+'_clm'+os.path.splitext(impmatrixtype)[0]+'.dat')
    print(args.outname+'_cl'+os.path.splitext(impmatrixtype)[0]+'.txt')


















