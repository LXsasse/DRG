# test_kernel_correlationmat.py
from drg_tools import io_utils as io
from drg_tools import motif_analysis as ma
from drg_tools import stats_functions as stf
import time
import numpy as np
import sys, os
from drg_tools.plotlib import plot_pwms
import matplotlib.pyplot as plt 



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
    correlation_torch_exact, log_pvalues_torch_exact, offsets_torch_exact, revcomp_matrix_torch_exact = ma.torch_compute_similarity_motifs(pwms, pwms, fill_logp_self = 64, min_sim = 5, padding = 0.25, infocont = True, bk_freq = 0.25, reverse_complement = True, verbose = False, device = 'cpu', exact = True)
    to = time.time()
    print('Torch module {:.2f}s'.format(to-ti))
    
    ti = time.time()
    correlation_torch, log_pvalues_torch, offsets_torch, revcomp_matrix_torch = ma.torch_compute_similarity_motifs(pwms, pwms, fill_logp_self = 64, min_sim = 5, padding = 0.25, infocont = True, bk_freq = 0.25, reverse_complement = True, verbose = False, device = 'cpu', exact = False)
    to = time.time()
    
    print('Torch module approximated {:.2f}s'.format(to-ti))
    
    tick = time.time()
    correlation, log_pvalues, offsets, revcomp_matrix = ma.align_compute_similarity_motifs(pwms, pwms, fill_logp_self = 64, min_sim = 5, padding = 0.25, infocont = True, bk_freq = 0.25, non_zero_elements = False, reverse_complement= True, njobs = 1, verbose = False)
    tock = time.time()
    
    print('Original module {:.2f}s'.format(tock-tick))
    
    # visual check
    if False: 
        figoriginal = plot_pwms(pwms, offsets=offsets[:,0], revcomp_matrix = revcomp_matrix[:,0], showaxes = True, log = True)
        figtorch = plot_pwms(pwms, offsets=offsets_torch[:,0], revcomp_matrix = revcomp_matrix_torch[:,0], showaxes = True, log = True)
        figtorchexact = plot_pwms(pwms, offsets=offsets_torch_exact[:,0], revcomp_matrix = revcomp_matrix_torch_exact[:,0], showaxes = True, log = True)
        plt.show()
    
    
    dcorrelation = correlation - correlation_torch
    dlog_pvalues = log_pvalues - log_pvalues_torch
    doffsets = offsets - offsets_torch
    drevcomp_matrix = revcomp_matrix - revcomp_matrix_torch
    
    print('mean correlation torch', np.mean(stf.correlation(correlation, correlation_torch, axis = 1, distance = False)))
    print('mean correlation torch_exact', np.mean(stf.correlation(correlation, correlation_torch_exact, axis = 1, distance = False)))
    
    print('mean correlation logp torch', np.mean(stf.correlation(log_pvalues, log_pvalues_torch, axis = 1, distance = False)))
    print('mean correlation logp torch_exact', np.mean(stf.correlation(log_pvalues, log_pvalues_torch_exact, axis = 1, distance = False)))
    
    
    print(drevcomp_matrix)
    print(doffsets)
    print(np.where(np.absolute(dcorrelation)>0.2))
    print(np.where(np.absolute(dlog_pvalues/log_pvalues)>0.1))
    print(np.sum(np.absolute(doffsets))/np.prod(np.shape(doffsets)))
    print(np.sum(np.absolute(drevcomp_matrix)))
    

    dcorrelation = correlation - correlation_torch_exact
    dlog_pvalues = log_pvalues - log_pvalues_torch_exact
    doffsets = offsets - offsets_torch_exact
    drevcomp_matrix = revcomp_matrix - revcomp_matrix_torch_exact
    
    print(drevcomp_matrix)
    print(doffsets)
    print(np.where(np.absolute(dcorrelation)>0.2))
    print(np.where(np.absolute(dlog_pvalues/log_pvalues)>0.5))
    print(np.sum(np.absolute(doffsets))/np.prod(np.shape(doffsets)))
    print(np.sum(np.absolute(drevcomp_matrix)))
        
    
    
    
    
