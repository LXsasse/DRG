# motif_analysis.py
'''
Contains functions to extract, compare, modify and summarize sets of 2D arrays (motifs)
'''

import numpy as np
import sys, os
from scipy.stats import pearsonr 
import time
from joblib import Parallel, delayed


def reverse(ppm):
    '''
    Generates the reverse complement of a pwm
    '''
    rppm = np.copy(ppm)
    rppm = rppm[::-1][:,::-1]
    return rppm

def determine_best_unique_matches(similarity):
    '''
    dependent best match. ppms in ppms_ref can only be assiged to one other.
    Parameters
    ----------
    similarity : similarity matrix
        i.e. the best matches are high values
    Returns
    -------
    bestmatch : np.ndarray
        indices of entris in axis 1 for entries in axis 0
    '''
    bestmatch = -np.ones(np.shape(similarity)[0], dtype = int)
    n_refs = np.arange(np.shape(similarity)[1], dtype = int)
    n_ppms = np.arange(np.shape(similarity)[0], dtype = int)
    asar = np.copy(similarity)
    while True:
        maxr = (int(np.argmax(asar)/np.shape(asar)[0]), np.argmax(asar)%np.shape(asar)[1])
        bestmatch[n_ppms[maxr[0]]] = n_refs[maxr[1]]
        asar = np.delete(asar, maxr[0], axis = 0)
        asar = np.delete(asar, maxr[1], axis = 1)
        n_refs = np.delete(n_refs, maxr[1])
        n_ppms = np.delete(n_ppms, maxr[0])
        if len(n_refs) == 0 or len(n_ppms) == 0:
            break
    return bestmatch

def align_compute_similarity_motifs(ppms, ppms_ref, fill_logp_self = 1000, min_sim = 5, padding = 0.25,
                                     infocont = False, bk_freq = 0.25, 
                                     non_zero_elements = False, reverse_complement
                                     = False, njobs = 1, verbose = False):
    '''
    Wrapper function for _align_compute_similarity_motifs that uses joblib to 
    parallize the computation of the similarity matrix
    
    Parameters
    ----------
    
    ppms : list or np.ndarray
        of motifs of shape (length, channels)
    ppms_ref : list or np.ndarray
        motif to compare ppms to
    fill_logp_self: 
        if ppms and ppms_ref are the same, diagonal elements will not be computed and just filled with 
        this value
    min_sim : int
        minimum bases that have to overlap in a comparison
    padding : float
        padding value, for nucleotides 0.25, representing uniform prob. that
        base is present
    non_zero_elements: boolean
        if True, only compare overlapping elements between pwms, or in other words
        mask everyting that is zero one of the motifs
    reverse_complement: boolean
        or two arrays defining if an individual pwm in
        one of the sets should be compared with its reverse complement
    njobs: int
        Number of processors to perform computation
    
    Returns
    -------
    correlation : 
        matrix with correlation distance between ppms and ppms_ref
    log_pvalues : 
        matrix with -log10 p values for the correlation matrix
    offsets : 
        best offset between motif in ppms and motif in ppms_ref to align motifs
    revcomp_matrix: 
        binary matrix determines if pwms was best aligned when in reverse complement
    '''
    
    # reverse_complement array determines if one should also compare the reverse complement of the pwms to all other pwms
    if isinstance(reverse_complement, bool):
        if reverse_complement == False:
            reverse_complement = np.array([np.zeros(len(ppms), dtype = int), 
                                       np.zeros(len(ppms_ref), dtype = int)])
        elif reverse_complement == True:
            reverse_complement = np.array([np.ones(len(ppms), dtype = int), 
                                       np.ones(len(ppms_ref), dtype = int)])
    elif len(reverse_complement) != 2:
        reverse_complement = np.array([reverse_complement, 
                                       np.ones(len(ppms_ref), dtype = int)])
    
    # check if pwm set is the same or different
    l_in, l_ref = len(ppms), len(ppms_ref)
    is_the_same = False
    if l_in == l_ref:
        all_the_same = []
        for p in range(len(ppms)):
            all_the_same.append(np.array_equal(ppms[p], ppms_ref[p]))
        is_the_same = np.array(all_the_same).all()
    
    one_half = is_the_same
    
    if njobs == 1:
        correlation, log_pvalues, offsets, revcomp_matrix, _ctrl = _align_compute_similarity_motifs(ppms, ppms_ref, one_half = one_half,                                          fill_logp_self = 1000, min_sim = min_sim, infocont = infocont, reverse_complement=reverse_complement, verbose = verbose)
    else:
        correlation = 2*np.ones((l_in, l_ref), dtype = np.float32)
        log_pvalues = np.zeros((l_in, l_ref), dtype = np.float32)
        offsets = 100*np.ones((l_in, l_ref), dtype = np.int16)
        revcomp_matrix = -np.ones((l_in, l_ref), dtype = np.int8)
        
        spacesi = np.linspace(0,np.shape(ppms)[0], njobs + 1, dtype = int)
        spacesj = np.linspace(0,np.shape(ppms)[0], njobs*int(one_half)+1,
                              dtype = int)
    
        if verbose:
            print('Computation split into', spaces)
        if one_half:
            results = Parallel(n_jobs=njobs)(delayed(_align_compute_similarity_motifs)
                                             (ppps[spacesi[i]:spacesi[i+1]], 
                                              ppms_ref[spacesj[j]:spacesj[j+1]], 
                                              one_half = (i == j) & one_half, 
                                              fill_logp_self = fill_logp_self, 
                                              min_sim = min_sim, 
                                              infocont = infocont, 
                                              reverse_complement = [reverse_complement[0][spacesi[i]:spacesi[i+1]], reverse_complement[1][spacesj[j]:spacesj[j+1]]],
                                              ctrl = (i,j)) 
                                             for i in range(0, njobs) 
                                             for j in range(i*int(one_half), njobs*int(one_half)+1)
                                             )
                                                                                                       
            for res in results:
                idx = res[-1]
                
                correlation[spacesi[idx[0]]:spacesi[idx[0]+1], spacesj[idx[1]]:spacesj[idx[1]+1]] = res[0]
                log_pvalues[spacesi[idx[0]]:spacesi[idx[0]+1], spacesj[idx[1]]:spacesj[idx[1]+1]] = res[1]
                revcomp_matrix[spacesi[idx[0]]:spacesi[idx[0]+1], spacesj[idx[1]]:spacesj[idx[1]+1]] = res[3]
                
                if idx[0] == idx[1]:
                    offsets[spacesi[idx[0]]:spacesi[idx[0]+1], spacesj[idx[1]]:spacesj[idx[1]+1]] = res[2]
                else:
                    correlation[spacesj[idx[1]]:spacesj[idx[1]+1], spacesi[idx[0]]:spacesi[idx[0]+1]] = res[0].T
                    log_pvalues[spacesj[idx[1]]:spacesj[idx[1]+1], spacesi[idx[0]]:spacesi[idx[0]+1]] = res[1].T
                    revcomp_matrix[spaces[idx[1]]:spacesj[idx[1]+1], spacesi[idx[0]]:spacesi[idx[0]+1]] = res[3].T
                    offsets[spacesi[idx[0]]:spacesi[idx[0]+1], spacesj[idx[1]]:spacesj[idx[1]+1]] = res[2][0]
                    offset[spacesj[idx[1]]:spacesj[idx[1]+1], spacesi[idx[0]]:spacesi[idx[0]+1]] = res[2][1].T
                    
    return correlation, log_pvalues, offsets, revcomp_matrix

# Previously compare_ppms
def _align_compute_similarity_motifs(ppms, ppms_ref, one_half = False, 
                                     fill_logp_self = 1000, min_sim = 5, padding = 0.25,
                                     infocont = False, bk_freq = 0.25, 
                                     non_zero_elements = False, reverse_complement
                                     = None, ctrl = None, verbose = False):
    '''
    Aligns PWMs and returns a correlation and p-value matrix for downstream analysis
    
    Parameters
    ----------
    
    ppms : list or np.ndarray
        of motifs of shape (length, channels)
    ppms_ref : list or np.ndarray
        motif to compare ppms to
    one_half: boolean
        if ppms and ppms_ref are equal, we only need to compute one half of the pairs
    fill_logp_self: 
        if one_half, diagonal elements will not be computed and just filled with 
        this value
    min_sim : int
        minimum bases that have to overlap in a comparison
    padding : float
        padding value, for nucleotides 0.25, representing uniform prob. that
        base is present
    non_zero_elements: boolean
        if True, only compare overlapping elements between pwms, or in other words
        mask everyting that is zero one of the motifs
    reverse_complement: boolean
        or two arrays defining if an individual pwm in
        one of the sets should be compared with its reverse complement
    ctrl: tuple
        if not None, return this tuple for joblib
    
    '''
    
    # reverse_complement array determines if one should also compare the reverse complement of the pwms to all other pwms
    if isinstance(reverse_complement, bool):
        if reverse_complement == False:
            reverse_complement = np.array([np.zeros(len(ppms), dtype = int), 
                                       np.zeros(len(ppms_ref), dtype = int)])
        elif reverse_complement == True:
            reverse_complement = np.array([np.ones(len(ppms), dtype = int), 
                                       np.ones(len(ppms_ref), dtype = int)])
    elif len(reverse_complement) != 2:
        reverse_complement = np.array([reverse_complement, 
                                       np.ones(len(ppms_ref), dtype = int)])
    
    if ctrl is not None and verbose:
        print('Computing', ctrl, 'part of matrix')
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

                    pvals.append(np.sign(peacor)*-np.log10(pval))
                    pearcors.append(peacor)
                    offs.append(i)
                    rvcmp.append(rc)
            
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
                    
    log_pvalues[np.isinf(log_pvalues)] = fill_logp_self
    
    if one_half:
        np.fill_diagonal(log_pvalues, fill_logp_self)
    
    if ctrl is not None and not one_half:
        offsets = [offsets, offdiagoffsets]
        
    return correlation, log_pvalues, offsets, revcomp_matrix, ctrl



def pfm2iupac(pwms, bk_freq = None):
    '''
    Translates position frequency matrix to iupac annotation
    Parameters
    ----------
    pwms: 
        list of pfms
    bk_freq:
        cut_off when to consider a chance for base to appear position
    '''
    hash = {'A':16, 'C':8, 'G':4, 'T':2}
    dictionary = {'A':16, 'C':8, 'G':4, 'T':2, 'R':20, 'Y':10, 'S':12, 'W':18, 'K':6, 'M':24, 'B':14, 'D':22, 'H':26, 'V':28, 'N':0}
    
    res = dict((v,k) for k,v in dictionary.items()) # reverse of dictionary
    n_nts = len(pwms[0][0])
    
    if bk_freq is None:
        bk_freq = (1./float(n_nts))*np.ones(n_nts)
    else:

        bk_freq = bk_freq*np.ones(n_nts)
    motifs = []
    for pwm in pwms:
        m = ''
        for p in pwm:
            score = 0 # score to look up ipac in 'res'
            for i in range(len(p)):
                if p[i] > bk_freq[i]: # only consider base p[i] if over bk_freq
                    score += list(hash.values())[i]
            m += res[score]
        motifs.append(m)
    return motifs
          




def combine_pwms(pwms, clusters, similarity, offsets, orientation, maxnorm = True,
                 remove_low = 0.45, method = 'mean', minlen = 4):
    
    '''
    Combine set of pwms based on their cluster assignments
    Need also precomputed offsets and orientation to each other (reverse of forward)
    '''
    lenpwms = np.array([len(pwm) for pwm in pwms])
    unclusters = np.unique(clusters)
    unclusters = unclusters[unclusters>=0]
    comb_pwms = []
    for u in unclusters:
        mask = np.where(clusters == u)[0]
        if len(mask) > 1:
 
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
            
            centerpiece = pwmscluster[0]
            
            centeroffsets = offsetcluster.T[0] # offset to the pwm that is most similar to all others
            centerorient = orient.T[0]
            
            lenmat = len(centerpiece)+max(0,np.amax(centeroffsets+clusterlen-len(centerpiece))) - min(0,np.amin(centeroffsets))
            
            seed = np.zeros((lenmat,len(pwms[0][0])))
            seedcount = np.zeros(len(seed))
            
            center = - min(0,np.amin(centeroffsets))
            
            centerrev = center + len(centerpiece)
            
            for o in range(len(pwmscluster)):
                pwm0 = pwmscluster[o]
                if centerorient[o] == 1:
                    pwm0 = reverse(pwm0)
                
                seed[center + centeroffsets[o]: center +centeroffsets[o] + len(pwm0)] += pwm0.astype(float)
                seedcount[center + centeroffsets[o]: center +centeroffsets[o] + len(pwm0)] += 1
                
            if method == 'mean':
                seed = seed/seedcount[:,None]
            elif method == 'sum':
                seed = seed/np.amax(seedcount)
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
                    seed[:] = 0.
            else:
                seed[:] = 0.
                
        comb_pwms.append(seed)
    return comb_pwms



def find_motifs(ref_attribution, cut, max_gap = 1, min_motif_size = 4, smooth_gap = True):
    ''' 
    Determine locations of motifs in attribution profiles at reference
    Parameters
    ----------
    ref_attribution : 
        array of shape = (length,) with the attributions at the reference
    cut : 
        cutoff for calling a base to be significant, and in a motif
    max_gap: 
        maximal allowed gap size
    min_motif_size: 
        minimal size of motifs to be added to motif set
    
    Returns
    -------
        list of lists that contain every position which is considered to be part of a motif. 
        
    '''
    # if smooth_gap:
    # Define more relaxed rule on what to call a gap based on the average
    # around the base within max_gap that is not significant
    def gap_call(i, msi, ref_attribution, cut, max_gap = 0):
        return msi *np.mean(ref_attribution[max(0,i-max_gap): min(len(ref_attribution),i+max_gap+1)]) < cut
    
    aloc = np.absolute(ref_attribution)>cut
    sign = np.sign(ref_attribution) # get sign of effects
    lra = len(ref_attribution)
    motiflocs = []

    gap = max_gap +1 # gapsize count
    msi = 1 # sign of motif
    potloc = [] # potential location of motif
    i = 0 
    while i < lra:
        if aloc[i]: # if location significant
            if len(potloc) == 0: # if potloc just started
                msi = np.copy(sign[i]) # determine which sign the entire motif should have
            
            if sign[i] == msi: # check if base has same sign as rest of motif
                potloc.append(i)
                gap = 0
            
            elif gap_call(i, msi, ref_attribution, cut, max_gap = max_gap * int(smooth_gap)): 
                # if the average with gapsize around the location is smaller than the cut, 
                # only then count as gap, otherwise just count as nothing
                gap += 1
                
                if gap > max_gap: # check that gap is still smaller than maximum gap size
                    # if this brought the gapsize over the max gapsize but the
                    # motif is long enough, then add to motifs
                    if len(potloc) >= min_motif_size: 
                        motiflocs.append(potloc)
                    # restart where the gap started so that motifs with 
                    # different direction following directly on other motifs can be counted
                    if len(potloc) > 0: 
                        i -= gap
                    # reset gap and poloc
                    gap = max_gap + 1
                    potloc = []
        
        elif gap_call(i, msi, ref_attribution, cut, max_gap = max_gap * int(smooth_gap)):
            gap +=1
            if gap > max_gap:
                if len(potloc) >= min_motif_size:
                    motiflocs.append(potloc)
                    
                if len(potloc) > 0:
                    i -= gap
                gap = max_gap + 1
                potloc = []
        i += 1
    
    if len(potloc) >= min_motif_size:
        motiflocs.append(potloc)
    return motiflocs




    
    
    
    
    
    
