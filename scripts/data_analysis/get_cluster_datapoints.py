import numpy as np
import sys, os
from data_processing import check 
from scipy.linalg import svd
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from functools import reduce

from drg_tools.io_utils import readalign_matrix_files
from drg_tools.data_processing import reduce_dim, getcentroids, determine_cluster





if __name__ == '__main__': 
    
    data = sys.argv[1]
    distance = sys.argv[2]
    clustering = sys.argv[3]
    clustpar = sys.argv[4]
    
    if '--outname' in sys.argv:
        outname = sys.argv[sys.argv.index('--outname')+1]
    else:
        outname = os.path.splitext(data)[0] 
        
    outname += '_cluster'+distance
    
    xnames, ynames, X = readalign_matrix_files(data)
    
    if '--reduce_dim' in sys.argv:
        red = sys.argv[sys.argv.index('--reduce_dim')+1]
        var = check(sys.argv[sys.argv.index('--reduce_dim')+2])
        X = reduce_dim(X, red, var)
        outname += 'reddim'+str(red) + str(var)
        print(np.shape(X))
    
    #ca = AgglomerativeClustering(n_clusters=None, affinity='correlation', linkage='average', distance_threshold = float(clustpar) )
    #ca.fit(X)
    #labels = cs.labels_
    #sys.exit()
    
    outname += '_'+ clustering + clustpar.strip('.')
    cparms = {}
    if '--clusterparams' in sys.argv:
        clusterparams = sys.argv[sys.argv.index('--clusterparams')+1]
        if "+" in clusterparams:
            clusterparams = clusterparams.split('+')
        else:
            clusterparams = [clusterparams]
        for c, clp in enumerate(clusterparams):
            clp = clp.split('=')
            cparms[clp[0]] = check(clp[1])
        
    
    maxsize = 10000
    if '--maxsize' in sys.argv:
        maxsize = int(sys.argv[sys.argv.index('--maxsize')+1])
    
    clusters = determine_cluster(X, distance, clustering, float(clustpar), cparms, maxsize = maxsize)
    
    np.savetxt(outname+'.txt', np.array([xnames, clusters]).T, fmt = '%s')
    print(outname+'.txt')
    
    
    
    
    
    
    
    
    
    
    
    
