# cluster_pwms.py
import numpy as np
import sys, os
from scipy.stats import pearsonr 
from sklearn.cluster import AgglomerativeClustering
import time


def reverse(ppm):
    rppm = np.copy(ppm)
    rppm = rppm[::-1][:,::-1]
    return rppm

def compare_ppms(ppms, ppms_ref, find_bestmatch = True, fill_logp_self = 0, one_half = True, min_sim = 5, padding = 0.25, infocont = False, bk_freq = 0.25, non_zero_elements = False, reverse_complement = None, ctrl = None, verbose = False):
    # reverse_complement array determines if one should also compare the reverse complement of the pwms to all other pwms
    if reverse_complement is None:
        reverse_complement = np.array([np.zeros(len(ppms), dtype = int), np.zeros(len(ppms_ref), dtype = int)])
    elif len(reverse_complement) != 2:
        reverse_complement = np.array([reverse_complement, np.ones(len(ppms_ref), dtype = int)])
    
    if ctrl is not None:
        print(ctrl)
    # measure lengths of motifs
    motif_len, motif_len_ref = [np.shape(ppm)[0] for ppm in ppms], [np.shape(ppm)[0] for ppm in ppms_ref]
    # alignment offsets that will be saved
    offsets = np.zeros((len(ppms), len(ppms_ref)), dtype = np.int8)
    if not one_half and ctrl is not None:
        offdiagoffsets = np.zeros((len(ppms), len(ppms_ref)), dtype = np.int8)
    # log_pvalues of correlation that will be saved
    log_pvalues = np.zeros((len(ppms), len(ppms_ref)), dtype = np.float32)
    # correlation matrix itself
    correlation = np.zeros((len(ppms), len(ppms_ref)), dtype = np.float32)*2
    # reverse complement matrix
    revcomp_matrix = np.zeros((len(ppms), len(ppms_ref)), dtype = np.int8)
    
    # whether to measure correlation of frequencies (pfms) of information content (pwms)
    if infocont:
        padding = max(0,np.log2(padding/bk_freq)) # padding needs to be adjusted if information content is chosen
    t0 = time.time()
    for p, ppm0 in enumerate(ppms):
        # just to check the speed of the algorithm
        if p> 0 and p% 25 == 0 and verbose:
            print(p, round(time.time() - t0,3))
            t0 = time.time()
        
        if one_half: # if one_half then ppms and ppms_ref have to be identical 
            p_start = p+1
        else:
            p_start = 0
        # normalize pfm to pwm
        if infocont:
            ppm0 = np.log2((ppm0+1e-8)/bk_freq)
            ppm0[ppm0<0] = 0
        
        for q in range(p_start, len(ppms_ref)):
            qpm = ppms_ref[q]
            if infocont:
                qpm = np.log2((qpm+1e-8)/bk_freq)
                qpm[qpm<0] = 0
            pvals = []
            pearcors = []
            offs = []
            rvcmp = []
            
            for rc in range(reverse_complement[0][p] * reverse_complement[1][q] +1):
                if rc == 1:
                    ppm = reverse(ppm0)
                else:
                    ppm = np.copy(ppm0)
                
                for i in range(min(0,-motif_len[p]+min_sim), motif_len_ref[q]-min(motif_len[p],min_sim) + 1):
                    if padding is not None:
                        refppm, testppm = np.ones((motif_len_ref[q]-min(motif_len[p],min_sim) + motif_len[p]-min(0,-motif_len[p]+min_sim),4))*padding, np.ones((motif_len_ref[q]-min(motif_len[p],min_sim) + motif_len[p]-min(0,-motif_len[p]+min_sim),4))*padding
                        
                        refppm[-min(0,-motif_len[p]+min_sim):motif_len_ref[q]-min(0,-motif_len[p]+min_sim)] = qpm
                        
                        testppm[i-min(0,-motif_len[p]+min_sim):i-min(0,-motif_len[p]+min_sim)+motif_len[p]] = ppm
            
                        mask = np.sum(testppm+refppm != 2*padding ,axis = 1) > 0
                        refppm, testppm = refppm[mask], testppm[mask]
                    else:
                        refppm = qpm[max(i,0):min(motif_len_ref[q],motif_len[p]+i)]
                        testppm = ppm[max(0,-i): min(motif_len[p],motif_len_ref[q]-i)]
                    
                    if non_zero_elements:
                        nonzmask = (testppm != 0) | (refppm != 0)
                        peacor, pval = pearsonr(testppm[nonzmask].flatten(), refppm[nonzmask].flatten())
                    else:
                        peacor, pval = pearsonr(testppm.flatten(), refppm.flatten())

                    #vectors.append([testppm.flatten(), refppm.flatten()])
                    #shorts.append(pfm2iupac([testppm, refppm], bk_freq = 0.28))
                    pvals.append(np.sign(peacor)*-np.log10(pval))
                    pearcors.append(peacor)
                    offs.append(i)
                    rvcmp.append(rc)
            #if p == 0 and q == 2:
                #sys.exit()
            maxp = np.argmax(pvals)
            log_pvalues[p,q] = pvals[maxp]
            offsets[p,q] = offs[maxp]
            correlation[p,q] = 1.-pearcors[maxp]
            revcomp_matrix[p,q] = rvcmp[maxp]

            if one_half:
                log_pvalues[q,p] = pvals[maxp]
                if rvcmp[maxp]:
                    offsets[q,p] = motif_len[p] - motif_len_ref[q] +offs[maxp] #### Need to adjust when reverse complement
                else:
                    offsets[q,p] = -offs[maxp]
                correlation[q,p] = 1.-pearcors[maxp]
                revcomp_matrix[q,p] = rvcmp[maxp]
            elif not one_half and ctrl is not None:
                if rvcmp[maxp]:
                    offdiagoffsets[p,q] = motif_len[p] - motif_len_ref[q] +offs[maxp] #### Need to adjust when reverse complement
                else:
                    offdiagoffsets[p,q] =-offs[maxp]
                    
            #if pearcors[maxp] > 0.8:
                #print(shorts[maxp])
                #plt.scatter(vectors[maxp][0], vectors[maxp][1])
                #plt.show()
                #plt.close()
    log_pvalues[np.isinf(log_pvalues)] = fill_logp_self
    #log_pvalues = log_pvalues * np.sign(correlation)
    if one_half:
        np.fill_diagonal(log_pvalues, fill_logp_self)
    
    
    if find_bestmatch or len(ppms)!=len(ppms_ref):
        # independent best match, one ppm from ppm_ref can be assigned to several ppm
        bestmatch = np.argmax(log_pvalues, axis = 1)
    else:
        # dependent best match. ppms in ppms_ref can only be assiged to one other.
        bestmatch = -np.ones(len(ppms), dtype = int)
        n_refs = np.arange(len(ppms_ref), dtype = int)
        n_ppms = np.arange(len(ppms_ref), dtype = int)
        asar = np.copy(log_pvalues)
        while True:
            maxr = (int(np.argmax(asar)/len(asar)), np.argmax(asar)%len(asar))
            bestmatch[n_ppms[maxr[0]]] = n_refs[maxr[1]]
            asar = np.delete(asar, maxr[0], axis = 0)
            asar = np.delete(asar, maxr[1], axis = 1)
            n_refs = np.delete(n_refs, maxr[1])
            n_ppms = np.delete(n_ppms, maxr[0])
            if len(n_refs) == 0:
                break
    if ctrl is not None and not one_half:
        offsets = [offsets, offdiagoffsets]
    return correlation, log_pvalues, offsets, revcomp_matrix, bestmatch, ctrl

# check if string can be integer or float
def numbertype(inbool):
    try:
        int(inbool)
    except:
        pass
    else:
        return int(inbool)
    try:
        float(inbool)
    except:
        pass
    else:
        return float(inbool)
    return inbool


# Read text files with PWMs
def read_pwm(pwmlist, nameline = 'Motif'):
    names = []
    pwms = []
    pwm = []
    obj = open(pwmlist, 'r').readlines()
    for l, line in enumerate(obj):
        line = line.strip().split('\t')
        if ((len(line) == 0) or (line[0] == '')) and len(pwm) > 0:
            pwm = np.array(pwm, dtype = float)
            pwms.append(np.array(pwm))
            pwm = []
            names.append(name)
        elif len(line) > 0:
            if line[0] == nameline:
                name = line[1]
                pwm = []
            elif line[0] == 'Pos':
                nts = line[1:]
            elif isinstance(numbertype(line[0]), int):
                pwm.append(line[1:])
    return pwms, names

def read_meme(pwmlist, nameline = 'MOTIF'):
    names = []
    pwms = []
    pwm = []
    obj = open(pwmlist, 'r').readlines()
    for l, line in enumerate(obj):
        line = line.strip().split()
        if ((len(line) == 0) or (line[0] == '')) and len(pwm) > 0:
            pwm = np.array(pwm, dtype = float)
            pwms.append(np.array(pwm))
            pwm = []
            names.append(name)
        elif len(line) > 0:
            if line[0] == nameline:
                name = line[1]
                pwm = []
            elif line[0] == 'ALPHABET=':
                nts = list(line[1])
            elif isinstance(numbertype(line[0]), float):
                pwm.append(line)
    if len(pwm) > 0:
        pwm = np.array(pwm, dtype = float)
        pwms.append(np.array(pwm))
        names.append(name)
    return pwms, names


def pfm2iupac(pwms, bk_freq = None):
    hash = {'A':16, 'C':8, 'G':4, 'T':2}
    dictionary = {'A':16, 'C':8, 'G':4, 'T':2, 'R':20, 'Y':10, 'S':12, 'W':18, 'K':6, 'M':24, 'B':14, 'D':22, 'H':26, 'V':28, 'N':0}
    res = dict((v,k) for k,v in dictionary.items())
    n_nts = len(pwms[0][0])
    if bk_freq is None:
        bk_freq = (1./float(n_nts))*np.ones(n_nts)
    else:
        bk_freq = bk_freq*np.ones(n_nts)
    motifs = []
    for pwm in pwms:
        m = ''
        for p in pwm:
            score = 0
            for i in range(len(p)):
                if p[i] > bk_freq[i]:
                    score += list(hash.values())[i]
            m += res[score]
        motifs.append(m)
    return motifs
          
  
def write_pwm(file_path, pwms, names):
    obj = open(file_path, 'w')
    for n, name in enumerate(names):
        obj.write('Motif\t'+name+'\n'+'Pos\tA\tC\tG\tT\n')
        for l, line in enumerate(pwms[n]):
            obj.write(str(l+1)+'\t'+'\t'.join(np.around(line,3).astype(str))+'\n')
        obj.write('\n')
    

def combine_pwms_single(pwms, clusters, similarity, offsets, maxnorm = True, remove_low = 0.5, method = 'sum'):
    lenpwms = np.array([len(pwm) for pwm in pwms])
    unclusters = np.unique(clusters)
    comb_pwms = []
    
    for u in unclusters:
        mask = np.where(clusters == u)[0]
        if len(mask) > 1:
            #print(pwmotif[mask])
            simcluster = np.argmax(np.sum(similarity[mask][:,mask], axis = 1))
            offsetcluster = offsets[mask][:,mask]
            off = offsetcluster.T[simcluster]
            #print(simcluster, off)
            clusterlen = lenpwms[mask]
            #print(similarity[mask][:, mask])
            #print(correlation[mask][:,mask])
            seed = np.zeros((np.amax(off+clusterlen)-np.amin(off),len(pwms[0][0])))
            seedcount = np.zeros(len(seed))
            #seed2 = np.copy(seed)
            seed[-np.amin(off):lenpwms[mask[simcluster]]-np.amin(off)] = pwms[mask[simcluster]]
            seedcount[-np.amin(off):lenpwms[mask[simcluster]]-np.amin(off)] += 1
            #seed1 = np.copy(seed)
            #print(pfm2iupac([seed], bk_freq = 0.28)[0])
            for m, ma in enumerate(mask):
                #print(m,ma)
                if m != simcluster:
                    seed[-np.amin(off)+off[m]:lenpwms[ma]+off[m]-np.amin(off)] += pwms[ma]
                    seedcount[-np.amin(off)+off[m]:lenpwms[ma]+off[m]-np.amin(off)] += 1
                    #check = np.copy(seed2)
                    #check[-np.amin(off)+off[m]:lenpwms[ma]+off[m]-np.amin(off)] = pwms[ma]
                    #print(pfm2iupac([check], bk_freq = 0.28)[0], pearsonr(check.flatten(), seed1.flatten()))
            if method == 'mean':
                seed = seed/seedcount[:,None]
        else:
            seed = pwms[mask[0]]

        if maxnorm:
            seed = seed/np.amax(seedcount)
        else:
            seed = seed/np.sum(seed,axis = 1)[:, None]
        if remove_low > 0:
            edges = np.where(np.sum(seed, axis = 1)>remove_low)[0]
            seed = seed[edges[0]:edges[-1]+1]
        comb_pwms.append(seed)
    return comb_pwms





import matplotlib as mpl
import logomaker
import pandas as pd
import matplotlib.pyplot as plt
def plot_pwm(pwm, log = False, axes = False):
        fig = plt.figure(figsize = (2.5,1), dpi = 300)
        ax = fig.add_subplot(111)
        lim = [min(0, -np.ceil(np.amax(-np.sum(np.ma.masked_array(pwm, pwm >0),axis = 1)))), np.ceil(np.amax(np.sum(np.ma.masked_array(pwm, pwm <0),axis = 1)))]
        if log:
            pwm = np.log2((pwm+1e-16)/0.25)
            pwm[pwm<0] = 0
            lim = [0,2]
        logomaker.Logo(pd.DataFrame(pwm, columns = list('ACGT')), ax = ax, color_scheme = 'classic')
        ax.set_ylim(lim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if not axes:
            ax.spines['left'].set_visible(False)
            ax.tick_params(labelleft = False, left = False, labelbottom = False, bottom = False)
        ax.set_yticks(lim)
        return fig

def signbool(i):
    return int(i*2)-1

def combine_pwms(pwms, clusters, similarity, offsets, orientation, maxnorm = True, remove_low = 0.45, method = 'sum', minlen = 4):
    lenpwms = np.array([len(pwm) for pwm in pwms])
    unclusters = np.unique(clusters)
    unclusters = unclusters[unclusters>=0]
    comb_pwms = []
    for u in unclusters:
        mask = np.where(clusters == u)[0]
        if len(mask) > 1:
            #print(mask)
            sim = similarity[mask][:,mask]
            offsetcluster = offsets[mask][:,mask]
            orient = orientation[mask][:,mask]
            pwmscluster = pwms[mask]
            clusterlen = lenpwms[mask]
            
            sort = np.argsort(-np.sum(sim, axis = 1))
            sim = sim[sort][:, sort]
            offsetcluster = offsetcluster[sort][:, sort]
            orient = orient[sort][:, sort]
            pwmscluster = pwmscluster[sort]
            clusterlen = clusterlen[sort]
            
            # could use entire matrices to align pwm pairs to seed, but is hard with two orients and two offsets and three lengths of motifs
            
            centerpiece = pwmscluster[0]
            
            centeroffsets = offsetcluster.T[0] # offset to the pwm that is most similar to all others
            centerorient = orient.T[0]
            
            lenmat = len(centerpiece)+max(0,np.amax(centeroffsets+clusterlen-len(centerpiece))) - min(0,np.amin(centeroffsets))
            
            #print(lenmat, len(centerpiece), max(0,np.amax(centeroffsets+clusterlen-len(centerpiece))))
            seed = np.zeros((lenmat,len(pwms[0][0])))
            seedcount = np.zeros(len(seed))
            
            center = - min(0,np.amin(centeroffsets))
            #print(center)
            centerrev = center + len(centerpiece)
            
            for o in range(len(pwmscluster)):
                pwm0 = pwmscluster[o]
                if centerorient[o] == 1:
                    pwm0 = reverse(pwm0)
                
                #combpwm = np.zeros(np.shape(seed))
                #p1 = np.copy(combpwm)
                #p1[center:center + len(centerpiece)] = centerpiece
                #combpwm[center + centeroffsets[o]: center +centeroffsets[o] + len(pwm0)] = pwm0
                #figa = plot_pwm(p1, axes = True)
                #figc = plot_pwm(combpwm, axes = True)
                #plt.show()
                
                seed[center + centeroffsets[o]: center +centeroffsets[o] + len(pwm0)] += pwm0.astype(float)
                seedcount[center + centeroffsets[o]: center +centeroffsets[o] + len(pwm0)] += 1
                
                
                
            if method == 'mean':
                seed = seed/seedcount[:,None]
        else:
            seed = pwms[mask[0]]

        if maxnorm:
            seed = seed/np.amax(seedcount)
        else:
            seed = seed/np.sum(np.absolute(seed),axis = 1)[:, None]
        if remove_low > 0:
            edges = np.where(np.sum(np.absolute(seed), axis = 1)>remove_low)[0]
            if len(edges) > 0:
                if edges[-1]-edges[0]>= minlen:
                    seed = seed[edges[0]:edges[-1]+1]
                else:
                    seed = seed
            else:
                seed = seed
                
        comb_pwms.append(seed)
    return comb_pwms





if __name__ == '__main__':
    pwmfile = sys.argv[1]
    outname = os.path.splitext(pwmfile)[0]
    infmt= os.path.splitext(pwmfile)[1]
    
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
        outname += '_clustered' + linkage
        
        n_clusters = None
        if '--nclusters' in sys.argv:
            n_clusters = int(distance_threshold)
            distance_threshold = None
            outname += 'N'
        
        outname +=str(distance_threshold)
        
        if '--clusteronlogp' in sys.argv:
            outname += 'pvalclust'
            #logs = 10**-logs
            clustering = AgglomerativeClustering(n_clusters = n_clusters, affinity = 'precomputed', linkage = linkage, distance_threshold = distance_threshold).fit(10**-logs)
        else:
            clustering = AgglomerativeClustering(n_clusters = n_clusters, affinity = 'precomputed', linkage = linkage, distance_threshold = distance_threshold).fit(correlation)
        
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
    
    
    
    
    
    
    
    
    
