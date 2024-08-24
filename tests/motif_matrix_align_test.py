# test_kernel_correlationmat.py
from drg_tools import io_utils as io
from drg_tools import motif_analysis as ma
from drg_tools import stats_functions as stf
import time
import numpy as np
import sys, os
from drg_tools.plotlib import plot_pwms
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr
import torch


def checknegsym(matrix):
    diff = matrix + matrix.T
    if np.sum(diff) != 0:
        mask = np.where(diff != 0)
        print(mask)
    

if __name__ == "__main__":
    
    #pwms, pwmnames, nts = io.readin_motif_files('../data/KERfromCTCFaH3K27acaH3K36me3aH3K4me3aH33aH3K27me3aH3K4me1aATAConseq2krcomp_mhall_Cormsek512l19TfEXPGELUmax10rcTvlCota_tc2dNoned1s1r1l7ma5nfc3s1024cbnoTfdo0.1tr1e-05SGD0.9bs64-F_seqset20.0KrcEXPmax0.5_ksiccomplete0.01pv_clkerneffctpfms.txt')
    np.random.seed(1)
    fpwm = np.array([[1,0,0,0],
                    [0,0,0,1],
                    [0,0,0,1],
                    [1,0,0,0],
                    [1,0,0,0],
                    [0,0,0,1],
                    [0,0,0,1],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,1,0,0],
                    [0,1,0,0],
                    [1,0,0,0]])
    
    pwms = [np.random.random(size=(np.random.randint(len(fpwm)-4,len(fpwm)+4), 4))*0.25 for i in range(20)]
    for p, pwm in enumerate(pwms):
        lpwm = len(pwm)
        offset = np.random.randint(-3,lpwm-(len(fpwm)-4))
        pwms[p][max(offset,0): min(lpwm, offset+len(fpwm))] += fpwm[-min(0,offset):-min(0,offset)+min(lpwm, offset+len(fpwm))-max(offset,0)]
        if np.random.randint(2) == 1:
            pwms[p] = pwms[p][::-1][:,::-1]
        pwms[p]/=np.sum(pwms[p], axis = 1)[:,None]
    pwms = [fpwm] + pwms
    
    plen = np.array([len(p) for p in pwms])
    print(plen)
    
    
    ti = time.time()
    correlation_torch_exact, offsets_torch_exact, revcomp_matrix_torch_exact = ma.torch_compute_similarity_motifs(pwms, pwms, fill_logp_self = 64, min_sim = 5, padding = 0.25, infocont = False, bk_freq = 0.25, reverse_complement = True, metric = 'correlation_pvalue', verbose = False, device = 'cpu', exact = True, return_alignment = True)
    to = time.time()
    print('Torch module {:.2f}s'.format(to-ti))
    
    ti = time.time()
    correlation_torch, offsets_torch, revcomp_matrix_torch = ma.torch_compute_similarity_motifs(pwms, pwms, fill_logp_self = 64, min_sim = 5, padding = 0.25, infocont = False, bk_freq = 0.25, reverse_complement = True, verbose = False, device = 'cpu', metric = 'correlation_pvalue', exact = False, return_alignment = True)
    to = time.time()
    
    print('Torch module approximated {:.2f}s'.format(to-ti))
    
    tick = time.time()
    correlation, log_pvalues, offsets, revcomp_matrix = ma.align_compute_similarity_motifs(pwms, pwms, fill_logp_self = 64, min_sim = 5, padding = 0.25, infocont = False, bk_freq = 0.25, non_zero_elements = False, reverse_complement= True, njobs = 1, verbose = False)
    tock = time.time()
    
    print('Original module {:.2f}s'.format(tock-tick))
    
    
    
    # visual check
    if False: 
        co = 9
        figoriginal = plot_pwms(pwms, offsets=offsets[:,co], revcomp_matrix = revcomp_matrix[:,co], showaxes = True, log = True)
        figtorch = plot_pwms(pwms, offsets=offsets_torch[:,co], revcomp_matrix = revcomp_matrix_torch[:,co], showaxes = True, log = True)
        
        figtorchexact = plot_pwms(pwms, offsets=offsets_torch_exact[:,co], revcomp_matrix = revcomp_matrix_torch_exact[:,co], showaxes = True, log = True)
        
        plt.show()

    '''
    print('mean correlation torch', np.mean(stf.correlation(correlation, correlation_torch, axis = 1, distance = False)))
    print('mean correlation torch_exact', np.mean(stf.correlation(correlation, correlation_torch_exact, axis = 1, distance = False)))
    '''
    print('mean correlation logp torch', np.mean(stf.correlation(log_pvalues, -np.log10(correlation_torch), axis = 1, distance = False)))
    print('mean correlation logp torch_exact', np.mean(stf.correlation(log_pvalues, -np.log10(correlation_torch_exact), axis = 1, distance = False)))
    
    print(np.mean(log_pvalues +np.log10(correlation_torch)))
    print(np.mean(log_pvalues +np.log10(correlation_torch_exact)))
    
    dcorrelation = correlation - correlation_torch
    doffsets = offsets - offsets_torch
    drevcomp_matrix = revcomp_matrix - revcomp_matrix_torch
    
    print(drevcomp_matrix)
    print(doffsets)
    where = np.where(dcorrelation<-0.1)
    print(where)
    '''
    for w in zip(where[0], where[1]):
        print(w, revcomp_matrix[w], offsets[w])
        print(correlation[w], correlation_torch[w])
    '''
    print(np.sum(np.absolute(doffsets))/np.prod(np.shape(doffsets)))
    print(np.sum(np.absolute(drevcomp_matrix)))
    

    dcorrelation = correlation - correlation_torch_exact
    doffsets = offsets - offsets_torch_exact
    drevcomp_matrix = revcomp_matrix - revcomp_matrix_torch_exact
    
    print(drevcomp_matrix)
    print(doffsets)
    where = np.where(np.absolute(dcorrelation)>0.02)
    print(where)
    for w in zip(where[0], where[1]):
        print(w, revcomp_matrix[w], offsets[w])
        print(correlation[w], correlation_torch_exact[w])
        plen, qlen  = len(pwms[w[0]]), len(pwms[w[1]])
        print(plen, qlen)
        clen = max(qlen, plen + offsets[w])-min(0,offsets[w])
        newpwm = np.ones((2, clen ,4))*0.25
        if revcomp_matrix[w] ==1:
            fpwm = ma.reverse(pwms[w[0]])
        else:
            fpwm = pwms[w[0]]
        newpwm[1, max(0,offsets[w]):max(0,offsets[w])+plen] = fpwm
        newpwm[0,-min(0,offsets[w]):-min(0,offsets[w])+qlen] = pwms[w[1]]
        figtorchexact = plot_pwms(list(newpwm), offsets=np.zeros(2,dtype = int), revcomp_matrix =np.zeros(2,dtype =int), showaxes = True, log = True)
        
        print(pearsonr(newpwm[0].flatten(), newpwm[1].flatten()))
        print(np.mean(newpwm,axis = (1,2)), np.std(newpwm,axis = (1,2)))
        newpwm -= np.mean(newpwm,axis = (1,2))[:,None,None]
        newpwm /= np.std(newpwm,axis = (1,2))[:,None,None]
        newpwm = np.transpose(newpwm, axes=(0,2,1))
        npwm0, npwm1 = torch.tensor(newpwm[[0]]), torch.tensor(newpwm[[1]])
        print(npwm0, npwm0.shape)
        print(ma.padded_weight_conv1d(npwm0, npwm1, clen))
       
        npwm0, npwm1 = torch.tensor(np.copy(pwms[w[1]])).to(torch.float32).transpose(-1,-2).unsqueeze(0), torch.tensor(np.copy(fpwm).T).to(torch.float32).unsqueeze(0)
        print(npwm0.dtype)
        print(ma.padded_weight_conv1d(npwm0, npwm1, 5, centered = True, standard = True, verbose=True))
        plt.show()
    
    print(np.sum(np.absolute(doffsets))/np.prod(np.shape(doffsets)))
    print(np.sum(np.absolute(drevcomp_matrix)))
        
    
    
    
    
