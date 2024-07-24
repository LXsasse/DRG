# cluster_pwms.py
import numpy as np
import sys, os
from scipy.stats import pearsonr 
from sklearn.cluster import AgglomerativeClustering

from drg_tools.motif_analysis import reverse, compare_ppms
from drg_tools.io_utils import read_pwm, read_meme, write_pwm, write_meme
from drg_tools.motif_analysis import pfm2iupac, combine_pwms
from drg_tools.plotlib import plot_pwm




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
        #print(outname)
        #print(correlation[0])
        #print(logs[0])
        #sys.exit()
    else:
        nameline = 'TF Name'
        if '--nameline' in sys.argv:
            nameline = sys.argv[sys.argv.index('--nameline')+1]
        
        if infmt == '.meme':
            pwm_set,pwmnames = read_meme(pwmfile)
        elif infmt == '.npz':
            pf = np.load(pwmfile, allow_pickle = True)
            pwm_set,pwmnames = pf['pwms'] , pf['pwmnames']
        else:
            pwm_set,pwmnames = read_pwm(pwmfile, nameline = nameline)
        
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
                
        
        if '--njobs' in sys.argv:
            from joblib import Parallel, delayed

            njobs = int(sys.argv[sys.argv.index('--njobs')+1])
            spaces = np.linspace(0,len(pwm_set), njobs + 1, dtype = int)
            print(spaces)
            results = Parallel(n_jobs=njobs)(delayed(compare_ppms)(pwm_set[spaces[i]:spaces[i+1]], pwm_set[spaces[j]:spaces[j+1]], find_bestmatch = True, one_half = i == j, fill_logp_self = 1000, min_sim = min_sim, infocont = infocont, reverse_complement = [revcom_array[spaces[i]:spaces[i+1]], revcom_array[spaces[j]:spaces[j+1]]], ctrl = (i,j)) for i in range(0, njobs) for j in range(i,njobs))
            
            correlation, logs, ofs, revcomp_matrix = np.ones((len(pwm_set), len(pwm_set)), dtype = np.float32)*2, np.zeros((len(pwm_set), len(pwm_set)), dtype = np.float32), np.zeros((len(pwm_set), len(pwm_set)), dtype = np.int16), np.zeros((len(pwm_set), len(pwm_set)), np.int8)
                                                                                                                                                                                        
            for res in results:
                idx = res[-1]
                
                correlation[spaces[idx[0]]:spaces[idx[0]+1], spaces[idx[1]]:spaces[idx[1]+1]] = res[0]
                logs[spaces[idx[0]]:spaces[idx[0]+1], spaces[idx[1]]:spaces[idx[1]+1]] = res[1]
                revcomp_matrix[spaces[idx[0]]:spaces[idx[0]+1], spaces[idx[1]]:spaces[idx[1]+1]] = res[3]
                
                if idx[0] == idx[1]:
                    ofs[spaces[idx[0]]:spaces[idx[0]+1], spaces[idx[1]]:spaces[idx[1]+1]] = res[2]
                else:
                    correlation[spaces[idx[1]]:spaces[idx[1]+1], spaces[idx[0]]:spaces[idx[0]+1]] = res[0].T
                    logs[spaces[idx[1]]:spaces[idx[1]+1], spaces[idx[0]]:spaces[idx[0]+1]] = res[1].T
                    ofs[spaces[idx[0]]:spaces[idx[0]+1], spaces[idx[1]]:spaces[idx[1]+1]] = res[2][0]
                    ofs[spaces[idx[1]]:spaces[idx[1]+1], spaces[idx[0]]:spaces[idx[0]+1]] = res[2][1].T
                    revcomp_matrix[spaces[idx[1]]:spaces[idx[1]+1], spaces[idx[0]]:spaces[idx[0]+1]] = res[3].T
                
        else:
            correlation, logs, ofs, revcomp_matrix, best, _ctrl = compare_ppms(pwm_set, pwm_set, find_bestmatch = True, one_half = True, fill_logp_self = 1000, min_sim = min_sim, infocont = infocont, reverse_complement = revcom_array)
        
        if '--save_stats' in sys.argv:
            np.savez_compressed(outname+'_stats.npz', pwmnames = pwmnames, correlation = correlation, logpvalues = logs, offsets = ofs, pwms = pwm_set, revcomp_matrix = revcomp_matrix)
            
    if '--outname' in sys.argv and '--loadstats' in sys.argv:
        outname = sys.argv[sys.argv.index('--outname')+1]
        
    linkage = sys.argv[2]
    if os.path.isfile(linkage):
        outname = os.path.splitext(linkage)[0]
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
        clusternames = ['Cluster_'+str(i) for i in np.unique(clusters)]
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



    #for c, clname in enumerate(clusternames):
        #print(clname, iupac[c])
        #mask = np.where(clusters == c)[0]
        #print(revcomp_matrix[mask][:,mask])
        #fig = plot_pwm(clusterpwms[c], log = True, axes = True)
        #fig2 = plot_pwm(clusterpwms_sing[c], log = True, axes = True)
        #for m in mask:
            #nf = plot_pwm(pwm_set[m], log = True, axes = True)
        #plt.show()
    
    
    
    
    
    
    
    
    
