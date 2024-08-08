import numpy as np
import sys, os

from drg_tools.io_utils import readtomtom, isint, readalign_matrix_files, sortafter


if __name__ == '__main__':

    
    clusterfile = np.genfromtxt(sys.argv[1], dtype = str)
    
    tomtom = sys.argv[1] # output tsv from tomtom
    tnames, target, pvals, qvals = readtomtom(tomtom)
    
    if '--outname' in sys.argv:
        outname = sys.argv[sys.argv.index('--outname')+1]
    else:
        outname = os.path.splitext(sys.argv[1])[0]
    
    cnames, clusters = clusterfile[:,0], clusterfile[:,1].astype(int)
    model = np.array([cn.split('_')[-1] for cn in cnames])

    uclust = np.unique(clusters)
    nmodel, ndup, members = [], [], []
    bestmatch, bestq, best_p = [], [], []
    matches, qs, ps = [] ,[], []
    for u, uc in enumerate(uclust):
        mask = clusters == uc
        members.append(','.join(cnames[mask]))
        mods, nmods = np.unique(model[mask], return_counts = True)
        nmodel.append(len(mods))
        ndup.append(np.sum(nmods>1))
        if str(uc) in tnames:
            tmask = np.where(tnames == str(uc))[0]
            matches.append(','.join(target[tmask]))
            qs.append(','.join(qvals[tmask].astype(str)))
            ps.append(','.join(pvals[tmask].astype(str)))
            bestmatch.append(target[tmask[0]])
            bestq.append(qvals[tmask[0]])
            bestp.append(pvals[tmask[0]])
    
    nmodel = np.array(nmodel)
    ndup = np.array(ndup)
    members = np.array(members)
    matches, qs, ps, bestmatch, bestq, bestp = np.array(matches), np.array(qs), np.array(ps), np.array(bestmatch), np.array(bestq), np.array(bestp)
    
    header = 'Cluster_ID\treproducibility\tMembers\tN_members\tBestMatch\tBest_q\tBest_p\tOtherMatches\tOther_q\tOther_p'
    
    datamatrix = np.array([uclust, nmodel, ndup, members, ndup, bestmatch, bestq, bestp, matches, qs, ps]).T
    
    if '--add_data' in sys.argv:
        adddata = sys.argv[sys.argv.index('--add_data')+1]
        rownames, columnnames, data = readalign_matrix_files(adddata, concatenate_axis = None)
        if '--add_dtypes' in sys.argv:
            dtypes = sys.argv[sys.argv.index('--add_dtype')+1]
            if ',' in dtypes:
                dtypes = dtypes.split(',')
                columnnames = np.array([cn+'.'+dtypes[c] for c, cno in enumerate(columnnames) for cn in cno])
            else:
                columnnames = np.array([cn+'.'+dtypes for c, cn in enumerate(columnnames)])
        if isinstance(data,list) :
            data = np.concatenate(data, axis = 1)
            if isinstance(columnnames,list):
                columnnames = np.concatenate(columnnames)
            
        
        
        if not np.array_equal(rownames, uclust.astype(str)):
            sort = sortafter(rownames, uclust.astype(str))
            data = data[sort]
        
        datamatrix = np.append(datamatrix, data, axis = 1)
        header = header +'\t'+'\t'.join()
            
            
        
        
        
