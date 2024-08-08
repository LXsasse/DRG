# cluster_pwms.py
import numpy as np
import sys, os
from scipy.stats import pearsonr 
from sklearn.cluster import AgglomerativeClustering

from drg_tools.motif_analysis import reverse, align_compute_similarity_motifs
from drg_tools.io_utils import readin_motif_files, write_pwm, write_meme_file
from drg_tools.motif_analysis import pfm2iupac, combine_pwms


if __name__ == '__main__':
    pwmfile = sys.argv[1]
    outname = os.path.splitext(pwmfile)[0]
    infmt= os.path.splitext(pwmfile)[1]
    
    if '--outname' in sys.argv:
        outname = sys.argv[sys.argv.index('--outname')+1]
    
    if '--loadstats' in sys.argv:
        pf = np.load(pwmfile, allow_pickle = True)
        pwmnames =pf['pwmnames']
        correlation = pf['correlation']
        logs = pf['logpvalues']
        ofs = pf['offsets']
        pwm_set = pf['pwms']
        revcomp_matrix = pf['revcomp_matrix']
        outname = outname.split('_stats')[0]
       
    else:
        nameline = 'TF Name'
        if '--nameline' in sys.argv:
            nameline = sys.argv[sys.argv.index('--nameline')+1]
        
        pwm_set, pwmnames, nts = readin_motif_files(pwmfile)
        
        if '--randomset' in sys.argv:
            np.random.seed(1)
            rs = int(sys.argv[sys.argv.index('--randomset')+1])
            outname += 'rs'+str(rs)
            rs = np.random.permutation(len(pwm_set))[:rs]
            pwm_set, pwmnames = pwm_set[rs], pwmnames[rs]
        
        print(np.shape(pwm_set), len(pwmnames))
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
                
        njobs = 1
        if '--njobs' in sys.argv:
            njobs = int(sys.argv[sys.argv.index('--njobs')+1])
        
        correlation, logs, ofs, revcomp_matrix= align_compute_similarity_motifs(pwm_set, pwm_set, fill_logp_self = 1000, min_sim = min_sim, infocont = infocont, reverse_complement = revcom_array, njobs = njobs)
        
        if '--save_stats' in sys.argv:
            np.savez_compressed(outname+'_stats.npz', pwmnames = pwmnames, correlation = correlation, logpvalues = logs, offsets = ofs, pwms = pwm_set, revcomp_matrix = revcomp_matrix)
            
    if '--outname' in sys.argv and '--loadstats' in sys.argv:
        outname = sys.argv[sys.argv.index('--outname')+1]
        
    linkage = sys.argv[2]
    if os.path.isfile(linkage):
        outname = os.path.splitext(linkage)[0]
        if '--outname' in sys.argv:
            outname = sys.argv[sys.argv.index('--outname')+1]
        clusters = np.genfromtxt(linkage, dtype = str)
        if not np.array_equal(pwmnames, clusters[:,0]):
            sort = []
            for p, pwn in enumerate(pwmnames):
                sort.append(list(clusters[:,0]).index(pwn))
            clusters = clusters[sort]
        clusters = clusters[:,1].astype(int)
        
    else:
        distance_threshold = float(sys.argv[3])
        outname += '_cld' + linkage
        
        n_clusters = None
        if '--nclusters' in sys.argv:
            n_clusters = int(distance_threshold)
            distance_threshold = None
            outname += 'N'
        
        outname +=str(distance_threshold)
        
        if '--clusteronlogp' in sys.argv:
            outname += 'pv'
            #logs = 10**-logs
            clustering = AgglomerativeClustering(n_clusters = n_clusters, metric = 'precomputed', linkage = linkage, distance_threshold = distance_threshold).fit(10**-logs)
        else:
            clustering = AgglomerativeClustering(n_clusters = n_clusters, metric = 'precomputed', linkage = linkage, distance_threshold = distance_threshold).fit(correlation)
        
        clusters = clustering.labels_
        np.savetxt(outname + '.txt', np.array([pwmnames,clusters]).T, fmt = '%s')
        print(outname + '.txt')
        
    print(len(pwm_set), 'form', len(np.unique(clusters)), 'clusters')
    
    clusterpwms = combine_pwms(np.array(pwm_set, dtype = object), clusters, logs, ofs, revcomp_matrix)
    #clusterpwms_sing = combine_pwms_single(pwm_set, clusters, logs, ofs)
        
    if '--clusternames' in sys.argv:
        clusternames = [str(i) for i in np.unique(clusters)]
    else:
        clusternames = [';'.join(np.array(pwmnames)[clusters == i]) for i in np.unique(clusters)]
     
    write_pwm(outname +'pfms.txt', clusterpwms, clusternames)
    
   
    if '--print_clusters' in sys.argv:
        iupac = pfm2iupac(clusterpwms)
        for c, clname in enumerate(clusternames):
            print(clname, iupac[c])
    else:
        uclust, nclust = np.unique(clusters, return_counts = True)
        nhist, yhist = np.unique(nclust, return_counts = True)
        for n, nh in enumerate(nhist):
            print(nh, yhist[n])


    
    
    
    
    
    
    
    
    
