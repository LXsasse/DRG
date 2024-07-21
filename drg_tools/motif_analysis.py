# motif_analysis.py
'''
Contains functions to extract, compare, modify and summarize sets of 2D arrays (motifs)
'''

import numpy as np
import sys, os
from scipy.stats import pearsonr 
import time



def reverse(ppm):
    '''
    Generates the reverse complement of a pwm
    '''
    rppm = np.copy(ppm)
    rppm = rppm[::-1][:,::-1]
    return rppm

def compare_ppms(ppms, ppms_ref, find_bestmatch = True, fill_logp_self = 0, one_half = True, min_sim = 5, padding = 0.25, infocont = False, bk_freq = 0.25, non_zero_elements = False, reverse_complement = None, ctrl = None, verbose = False):
    '''
    Aligns PWMs and returns a correlation and p-value matrix for downstream analysis
    '''
    
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
            seedcount = np.ones(len(seed))

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



def find_motifs(a, cut, mg, msig):
    ''' 
    Extract motifs from sequence attributions 
    '''
    
    aloc = np.absolute(a)>cut
    sign = np.sign(a) # get sign of effects
    motiflocs = []

    gap = mg +1 # gapsize count
    msi = 1 # sign of motif
    potloc = [] # potential location of motif
    i = 0 
    while i < len(a):
        if aloc[i]: # if location significant
            if len(potloc) == 0: # if potloc just started
                msi = np.copy(sign[i]) # determine which sign the entire motif should have
            
            if sign[i] == msi: # check if base has same sign as rest of motif
                potloc.append(i)
                gap = 0
            elif msi *np.mean(a[max(0,i-mg): min(len(a),i+mg+1)]) < cut: # if the average with gapsize around the location is smaller than the cut then count as gap
                gap += 1
                if gap > mg: # check that gap is still smaller than maximum gap size
                    if len(potloc) >= msig: # if this brought the gapsize over the max gapsize but the motif is long enough, then add to motifs
                        motiflocs.append(potloc)
                    if len(potloc) > 0: # restart where the gap started so that motifs with different direction following directly on other motifs can be counted
                        i -= gap
                    gap = mg + 1
                    potloc = []
        elif msi *np.mean(a[max(0,i-mg): min(len(a),i+mg+1)]) < cut:
            gap +=1
            if gap > mg:
                if len(potloc) >= msig:
                    motiflocs.append(potloc)
                    #print(a[potloc], a[potloc[0]:potloc[-1]])
                if len(potloc) > 0:
                    i -= gap
                gap = mg + 1
                potloc = []
        i += 1
    if len(potloc) >= msig:
        motiflocs.append(potloc)
    return motiflocs




    
    
    
    
    
    
