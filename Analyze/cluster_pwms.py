# cluster_pwms.py
import numpy as np
import sys, os
from scipy.stats import pearsonr 
from sklearn.cluster import AgglomerativeClustering

def compare_ppms(ppms, ppms_ref, find_bestmatch = True, fill_logp_self = 0, one_half = True, min_sim = 5, padding = 0.25, infocont = False, bk_freq = 0.25, non_zero_elements = False):
    # measure lengths of motifs
    motif_len, motif_len_ref = [np.shape(ppm)[0] for ppm in ppms], [np.shape(ppm)[0] for ppm in ppms_ref]
    # alignment offsets that will be saved
    offsets = np.zeros((len(ppms), len(ppms_ref)), dtype = int)
    # log_pvalues of correlation that will be saved
    log_pvalues = np.zeros((len(ppms), len(ppms_ref)), dtype = float)
    # correlation matrix itself
    correlation = np.zeros((len(ppms), len(ppms_ref)), dtype = float)
    # whether to measure correlation of frequencies (pfms) of information content (pwms)
    if infocont:
        padding = max(0,np.log2(padding/bk_freq)) # padding needs to be adjusted if information content is chosen
    for p, ppm in enumerate(ppms):
        # just to check the speed of the algorithm
        if p% 25 == 0:
            print(p)
        
        if one_half: # if one_half then ppms and ppms_ref have to be identical 
            p_start = p+1
        else:
            p_start = 0
        # normalize pfm to pwm
        if infocont:
            ppm = np.log2(ppm/bk_freq)
            ppm[ppm<0] = 0
            
        for q in range(p_start, len(ppms_ref)):
            qpm = ppms_ref[q]
            if infocont:
                qpm = np.log2(qpm/bk_freq)
                qpm[qpm<0] = 0
            pvals = []
            pearcors = []
            offs = []
            
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
                pvals.append(-np.sign(peacor)*np.log10(pval))
                pearcors.append(peacor)
                offs.append(i)
            maxp = np.argmax(pvals)
            log_pvalues[p,q] = pvals[maxp]
            offsets[p,q] = offs[maxp]
            correlation[p,q] = 1.-pearcors[maxp]
            if one_half:
                log_pvalues[q,p] = pvals[maxp]
                offsets[q,p] = -offs[maxp]
                correlation[q,p] = 1.-pearcors[maxp]
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
        bestmatch = np.argmax(log_pvalues, axis = 1)
    else:
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
    return correlation, log_pvalues, offsets, bestmatch

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
    

def combine_pwms(pwms, clusters, similarity, offsets, maxnorm = True, remove_low = 0.5, method = 'sum'):
    lenpwms = np.array([len(pwm) for pwm in pwms])
    unclusters = np.unique(clusters)
    comb_pwms = []
    
    #pwmotif = np.array(pfm2iupac(pwms, bk_freq = 0.28))
    
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
            seed = seed/np.amax(np.sum(seed,axis = 1))
        else:
            seed = seed/np.sum(seed,axis = 1)[:, None]
        if remove_low > 0:
            edges = np.where(np.sum(seed, axis = 1)>remove_low)[0]
            seed = seed[edges[0]:edges[-1]+1]
        comb_pwms.append(seed)
    return comb_pwms



if __name__ == '__main__':
    pwmfile = sys.argv[1]
    outname = os.path.splitext(pwmfile)[0]
    nameline = 'TF Name'
    if '--nameline' in sys.argv:
        nameline = sys.argv[sys.argv.index('--nameline')+1]
    pwm_set,pwmnames = read_pwm(pwmfile, nameline = nameline)
    
    print(np.shape(pwm_set[0]))
    
    correlation, logs, ofs, best = compare_ppms(pwm_set, pwm_set, find_bestmatch = True, fill_logp_self = 1000)
    
    linkage = sys.argv[2]
    distance_threshold = float(sys.argv[3])
    
    outname += '_clustered' + linkage+str(distance_threshold)
    
    clustering = AgglomerativeClustering(n_clusters = None, affinity = 'precomputed', linkage = linkage, distance_threshold = distance_threshold).fit(correlation)
    
    clusters = clustering.labels_
    print(len(pwm_set), 'form', len(np.unique(clusters)), 'clusters')
        
    clusterpwms = combine_pwms(pwm_set, clusters, logs, ofs)
    clusternames = [';'.join(np.array(pwmnames)[clusters == i]) for i in np.unique(clusters)]
    iupac = pfm2iupac(clusterpwms)
    for c, clname in enumerate(clusternames):
        print(clname, iupac[c])
    
    write_pwm(outname +'.txt', clusterpwms, clusternames)
    
    
    
    
    
    
    
    
