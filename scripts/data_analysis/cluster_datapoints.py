import numpy as np
import sys, os
from data_processing import check 
from scipy.linalg import svd
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from functools import reduce



def read(outputfile, delimiter = None):
    if os.path.isfile(outputfile):
        if os.path.splitext(outputfile)[1] == '.npz':
            Yin = np.load(outputfile, allow_pickle = True)
            if 'counts' in Yin.files:
                Y = Yin['counts']
            elif 'values' in Yin.files:
                Y = Yin['values']
            outputnames = Yin['names'] # Y should of shape (nexamples, nclasses, l_seq/n_resolution)
        else:
            Yin = np.genfromtxt(outputfile, dtype = str, delimiter = delimiter)
            Y, outputnames = Yin[:, 1:].astype(float), Yin[:,0]
    elif ',' in outputfile:
        Y, outputnames = [], []
        for putfile in outputfile.split(','):
            if os.path.splitext(putfile)[1] == '.npz':
                Yin = np.load(putfile, allow_pickle = True)
                onames = Yin['names']
                sort = np.argsort(onames)
                if 'counts' in Yin.files:
                    yname = 'counts'
                elif 'values' in Yin.files:
                    yname = 'values'
                Y.append(Yin[yname][sort])
                outputnames.append(onames[sort])
            else:
                Yin = np.genfromtxt(putfile, dtype = str, delimiter = delimiter)
                onames = Yin[:,0]
                sort = np.argsort(onames)
                Y.append(Yin[:, 1:].astype(float)[sort]) 
                outputnames.append(onames[sort])
            
        comnames = reduce(np.intersect1d, outputnames)
        for i, yi in enumerate(Y):
            Y[i] = yi[np.isin(outputnames[i], comnames)]
        outputnames = comnames
        Y = np.concatenate(Y, axis = 1)
    print(len(outputnames), np.shape(Y))
    return outputnames, Y

def reduce_dim(X, red, var):
    Xmean = np.mean(X, axis = 0)
    X = X - Xmean
    if 'standard' in red:
        X /= np.std(X, axis = 0)
    U, S, Vh = svd(X, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=True, lapack_driver='gesdd')
    print(np.shape(U))
    # check if singular values are in correct order
    if not np.array_equal(np.argsort(-S), np.arange(len(S))):
        print('Singular values not sorted')
    if var > 1:
        U, S, Vh = U[:,:var], S[:var], Vh[:var]
    else: 
        Sv = S**2
        SV = np.cumsum(Sv)/np.sum(Sv)
        var = np.where(SV < var)[0][-1] + 1
        print('Selected', var, 'singular values')
        U, S, Vh = U[:,:var], S[:var], Vh[:var]
        print(SV[:var])
    if 'normed' not in red:
        U = U * S
    return U

def getcentroids(labels, distmat):
    clust = np.unique(labels)
    print(len(clust), 'centroids in', len(labels))
    centroids = []
    maxdist = []
    for c, cl in enumerate(clust):
        mask = np.where(labels == cl)[0]
        mdist = distmat[mask][:,mask]
        argmin = np.argmin(np.sum(mdist, axis = 1))
        cent = mask[argmin]
        centroids.append(cent)
        maxdist.append(np.amax(mdist[argmin]))
    return np.array(centroids), np.array(maxdist)
    

def determine_cluster(X, distance, clustering, clustpar, cparms, maxsize = 10000):
    if type(clustpar) == int:
        n_clusters = clustpar
        distance_threshold = None
    else:
        n_clusters = None
        distance_threshold = clustpar
    
    Xold = None
    if len(X) > maxsize:
        mask = np.zeros(len(X)) # select random set of maxsize data points for clustering
        mask[np.random.permutation(len(X))[:maxsize]] = 1
        mask = mask == 1
        Xold = np.copy(X) # copy original X for assigning rest of data points later
        X= X[mask] # compute distance matrix only for maxsize data points
    distmat = cdist(X, X, distance)
    
    if clustering == 'agglomerative' or clustering == 'Agglomerative' or clustering == 'AgglomerativeClustering':
        ca = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', distance_threshold = distance_threshold, **cparms)
    ca.fit(distmat)
    labels = ca.labels_
    
    # possible changes:
        # add option to compare to the mean of clusters instead of centroid
        # add option to compare data points to other clusters if they were set to -1 because of cluster specific threshold.
    if Xold is not None:
        centroids, distcentroid = getcentroids(labels, distmat) # determine the centroids of all clusters to assign leftover data points by measuring the distance to them
        adjust = np.median(distcentroid[distcentroid != 0])# adjust centroid dists for single clusters to the median of distances of other clusters
        distcentroid[distcentroid == 0] = adjust # distcentroid contains the max distance of the centroid to other data points in the same cluster
        ndist = cdist(Xold, X[centroids], distance)
        argmins = np.argmin(ndist, axis = 1) # determine which centroid is closest for all data points
        amins = np.amin(ndist, axis = 1)
        nlabels = -np.ones(len(ndist), dtype = int) # new labels for all data points, intialized as -1
        nlabels = labels[centroids][argmins] # quickly assign all data points the closest cluster
        for l, lab in enumerate(np.unique(labels)): # set cluster labels back to -1 if distance to centroid does not fulfill distance threshold
            mask = np.where(nlabels == lab)[0]
            nlabels[mask[amins[mask] > distcentroid[l]]] = -1
        #nlabels[amins > distance_threshold] = -1 # alternatively one could simply check if it is closer than the original cut-off to the centroid. That would make it less likely that all other data points are within that distance
        print(int(np.sum(nlabels == -1)), 'not assigned out of', len(nlabels))
        labels = nlabels
    return labels




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
    
    xnames, X = read(data)
    
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
    
    
    
    
    
    
    
    
    
    
    
    
