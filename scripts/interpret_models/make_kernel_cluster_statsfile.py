import numpy as np
import sys, os


clusterfile = np.genfromtxt(sys.argv[1], dtype = str)
outname = os.path.splitext(sys.argv[1])[0]
cnames, clusters = clusterfile[:,0], clusterfile[:,1].astype(int)
model = np.array([cn.split('_')[-1] for cn in cnames])

uclust = np.unique(clusters)
nmodel, ndup = [], []
for u, uc in enumerate(uclust):
    mask = clusters == uc
    mods, nmods = np.unique(model[mask], return_counts = True)
    nmodel.append(len(mods))
    ndup.append(np.sum(nmods>1))
    
np.savetxt(outname+'_reprod.txt', np.array([uclust,nmodel]).T, fmt = '%s')
np.savetxt(outname+'_duplic.txt', np.array([uclust,ndup]).T, fmt = '%s')



