import numpy as np
import sys, os
from scipy.stats import pearsonr 
from statsmodels.stats.multitest import multipletests
from sklearn.cluster import AgglomerativeClustering
import time
from cluster_pwms import compare_ppms, read_pwm, read_meme, pfm2iupac, write_pwm, combine_pwms
# Can be used in combination with cluster_pwms.py: First cluster a sample of sequences and then compare rest to clusters and assign to cluster based on best match

def write_tomtom(outname, datanames, pwmnames, passed, pvals, qvals, correlation, ofs, revcomp_matrix):
    obj = open(outname+'.tomtom.tsv', 'w')
    obj.write('Query_ID\tTarget_ID\tOptimal_offset\tp-value\tCorrelation\tq-value\tOrientation\n')
    for i,j in zip(passed[0], passed[1]):
        obj.write(pwmnames[i]+'\t'+datanames[j]+'\t'+str(ofs[i,j])+'\t'+str(pvals[i,j])+'\t'+str(correlation[i,j])+'\t'+str(qvals[i,j])+'\t'+str(revcomp_matrix[i,j])+'\n')
    

def readin(pwmfile):
    infmt= os.path.splitext(pwmfile)[1]    
    if infmt == '.meme':
        pwm_set,pwmnames = read_meme(pwmfile)
    elif infmt == '.npz':
        pf = np.load(pwmfile, allow_pickle = True)
        pwm_set,pwmnames = pf['pwms'] , pf['pwmnames']
    else:
        pwm_set,pwmnames = read_pwm(pwmfile, nameline = nameline)
    return pwm_set, pwmnames
        
if __name__ == '__main__':
    # Read pwms that need to be assigned
    pwmfile = sys.argv[1]
    outname = os.path.splitext(pwmfile)[0]
    pwm_set, pwmnames = readin(pwmfile)
    
    # Read assigned pwms from data base or clustered
    databasefile = sys.argv[2]
    outname += '_in_'+os.path.splitext(os.path.split(databasefile)[1])[0]
    data_set, datanames = readin(databasefile)

    cutoff = float(sys.argv[3])
    outname += 'lt'+str(cutoff)
    # choose if q-value or p-value should be used for cutoff
    corrected = True
    if '--uncorrected' in sys.argv:
        outname += '-pv'
        corrected = False
    
    # choose if best assignment only or all assignments should be kept.
    best = True
    if '--select_all' in sys.argv:
        best = False

    min_sim = 4
    if '--min_overlap' in sys.argv:
        min_sim = int(sys.argv[sys.argv.index('--min_overlap')+1])
        outname += 'ms'+str(min_sim)
    
    infocont=False
    if '--infocont' in sys.argv:
        infocont = True
        outname += 'ic'
    
    revcom_array = np.zeros(len(pwmnames), dtype = int)
    if '--reverse_complement' in sys.argv:
        revcomfile = sys.argv[sys.argv.index('--reverse_complement')+1]
        if os.path.isfile(revcomfile):
            revcom_array = np.genfromtxt(sys.argv[sys.argv.index('--reverse_complement')+1], dtype = str)
            if np.array_equal(revcom_array[:,0], pwmnames):
                revcom_array = revcom_array[:,1].astype(int)
            else:
                revcom_array = np.array([revcom_array[list(revcom_array[:,0]).index(n),1] for n in pwmnames], dtype = int)
        elif revcomfile == 'all':
            revcom_array = np.ones(len(pwmnames), dtype = int)
            
    if '--njobs' in sys.argv:
        from joblib import Parallel, delayed

        njobs = int(sys.argv[sys.argv.index('--njobs')+1])
        spaces = np.append(np.arange(0,len(pwm_set), len(data_set), dtype = int), [len(pwm_set)])
        print(spaces)
        
        results = Parallel(n_jobs=njobs)(delayed(compare_ppms)(pwm_set[spaces[i]:spaces[i+1]], data_set, find_bestmatch = True, one_half = False, fill_logp_self = 1000, min_sim = min_sim, infocont = infocont, reverse_complement = revcom_array[spaces[i]:spaces[i+1]], ctrl = i) for i in range(0, len(spaces)-1))
        
        correlation, logs, ofs, revcomp_matrix = np.ones((len(pwm_set), len(data_set)), dtype = np.float32)*2, np.zeros((len(pwm_set), len(data_set)), dtype = np.float32), np.zeros((len(pwm_set), len(data_set)), dtype = np.int16), np.zeros((len(pwm_set), len(data_set)), np.int8)
                                                                                                                                                                                    
        for res in results:
            i = res[-1]
            
            correlation[spaces[i]:spaces[i+1]] = res[0]
            logs[spaces[i]:spaces[i+1]] = res[1]
            revcomp_matrix[spaces[i]:spaces[i+1]] = res[3]
            ofs[spaces[i]:spaces[i+1]] = res[2]
    else:
            correlation, logs, ofs, revcomp_matrix, best_match, _ctrl = compare_ppms(pwm_set, data_set, find_bestmatch = True, one_half = False, fill_logp_self = 1000, min_sim = min_sim, infocont = infocont, reverse_complement = revcom_array)
    
    pvals = 10**(-logs)
    stats = np.copy(pvals)
    qvals = np.ones_like(pvals)
    for p, pv in enumerate(pvals):
        rej_, qvals[p], a_, b_ = multipletests(pv, alpha=cutoff, method='fdr_bh')
    if corrected:
        stats = np.copy(qvals)
    
    #print(pvals, np.shape(pvals))
    #print(np.amax(correlation), np.amin(correlation), np.shape(correlation))
    # generate tomtom type output
    if not best:
        passed = np.where(stats <= cutoff)
        print(len(passed[0]), '/', np.prod(np.shape(stats)))
        write_tomtom(outname, datanames, pwmnames, passed, pvals, qvals, 1.-correlation, ofs, revcomp_matrix)
    else: # generate clusterfile
        clusters = np.argmin(stats, axis =1)
        clusterstats = np.amin(stats, axis = 1)
        clustermask = clusterstats>cutoff
        clusters[clustermask] = -1
        np.savetxt(outname + '.txt', np.array([pwmnames,np.array(datanames)[clusters]]).T, fmt = '%s')
        print(outname + '.txt')
        
        print(len(pwm_set), 'form', len(np.unique(clusters)), 'clusters')
        print(np.sum(clusters == -1), 'could not be assigned to pre-existing data set')
        
        pwmnames, pwm_set, clusters, logs, ofs, revcomp_matrix = np.array(pwmnames)[clustermask], np.array(pwm_set, dtype = object)[clustermask], clusters[clustermask], logs[clustermask], ofs[clustermask], revcomp_matrix[clustermask]
        clusterpwms = combine_pwms(pwm_set, clusters, logs, ofs, revcomp_matrix)
    
        if '--clusternames' in sys.argv:
            clusternames = ['Cluster_'+str(i) for i in np.unique(clusters)]
        else:
            clusternames = [';'.join(np.array(pwmnames)[clusters == i]) for i in np.unique(clusters)]
     
        write_pwm(outname +'pfms.txt', clusterpwms, clusternames)

        iupac = pfm2iupac(clusterpwms)
        for c, clname in enumerate(clusternames):
            print(clname, iupac[c], np.sum(clusters == c))
    

