import numpy as np
import sys, os
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from translate_pearson_to_pvalue import correlation_to_pvalue, pvalue_to_correlation

def read_matrix(fi):
    obj = open(fi, 'r').readlines()
    header = None
    matrix, names = [], []
    for l, line in enumerate(obj):
        if l == 0 and line[0] == '#':
            header = np.array(line.strip('#').strip().split()[2:])
        else:
            line = line.strip('#').strip().split()
            names.append(line[0])
            matrix.append(line[2:])
    return np.array(names), np.array(matrix, dtype = float), header


if __name__ == '__main__':

    clusterfile = sys.argv[1]
    importance_matrices = sys.argv[2] # with comma connected files
    # output corrected clusterfile, and summarized importance matrix
    mincorr = 0.7
    pvalue = False

    clusters = np.genfromtxt(clusterfile, dtype = str)
    cnames, cfiles = np.array([cl.rsplit('_',1)[0] for cl in clusters[:,0]]), np.array([cl.rsplit('_',1)[-1] for cl in clusters[:,0]])

    if ',' in importance_matrices:
        importance_matrices = importance_matrices.split(',')
    else:
        importance_matrices = [importance_matrices]

    # make sure that order in clusterfile is the same as the matrix files. 
    imp_matrix, imp_matnames, imp_matstats, imp_matclusters = [], [], [], []
    for c, cf in enumerate(np.unique(cfiles)):
        print(cf, importance_matrices[c])
        names, matrix, header = read_matrix(importance_matrices[c])
        
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
            print('Cell types different in matrices, check selection')
            sys.exit()

    imp_matrix = np.concatenate(imp_matrix, axis = 0)
    imp_matnames = np.concatenate(imp_matnames, axis = 0)
    imp_matstats = np.concatenate(imp_matstats, axis = 0)
    imp_matclusters = np.concatenate(imp_matclusters, axis = 0)

    if pvalue:
        mincorr = 1.-pvalue_to_correlation(mincorr, np.shape(imp_matrix)[-1])
        
    print('Correlation cutoff is', mincorr)

    unclusters = np.unique(imp_matclusters)
    print('Initial clusters', len(unclusters), 'in', len(imp_matclusters))

    newclusters = -np.ones(len(imp_matclusters), dtype = int)
    newmatrix = []
    clu = 0
    for u, unc in enumerate(unclusters):
        mask = np.where(imp_matclusters == unc)[0]
        if len(mask) > 0:
            #print(len(mask))
            dist = cdist(imp_matrix[mask], imp_matrix[mask], 'correlation')
            clustering = AgglomerativeClustering(n_clusters = None, affinity = 'precomputed', linkage = 'complete', distance_threshold = mincorr).fit(dist)
            newclust = clustering.labels_
            unnew = np.unique(newclust)
            #if len(unnew > 1):
                #print('>', len(unnew))
            for n, nc in enumerate(unnew):
                newclusters[mask[newclust == nc]] = clu
                newmatrix.append(np.mean(imp_matrix[mask[newclust == nc]], axis = 0))
                clu += 1
        else:
            newclusters[mask] == clu
            newmatrix.append(imp_matrix[mask[0]])
            clu += 1

    newmatrix = np.array(newmatrix)
    uniquenewclust, Nuniquenew = np.unique(newclusters, return_counts = True)
    print('Final clusters', len(uniquenewclust), 'in', len(imp_matclusters))

    print(np.shape(newmatrix))

impmatrixtype = importance_matrices[0].rsplit('_',1)[-1]

np.savetxt(os.path.splitext(clusterfile)[0]+'_cl'+os.path.splitext(impmatrixtype)[0]+'.txt', np.array([clusters[:,0], newclusters.astype(int)]).T, fmt = '%s')

np.savetxt(os.path.splitext(clusterfile)[0]+'_clm'+os.path.splitext(impmatrixtype)[0]+'.dat', np.concatenate([uniquenewclust.astype(int).astype(str).reshape(-1,1), Nuniquenew.astype(int).astype(str).reshape(-1,1), newmatrix],axis = 1), fmt = '%s', header = 'Kernelcluster N_kernels '+' '.join(header))

print(os.path.splitext(clusterfile)[0]+'_clm'+os.path.splitext(impmatrixtype)[0]+'.dat')
print(os.path.splitext(clusterfile)[0]+'_cl'+os.path.splitext(impmatrixtype)[0]+'.txt')


















