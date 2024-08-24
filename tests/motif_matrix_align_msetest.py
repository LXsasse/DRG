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
from scipy.spatial import distance as scd
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
    correlation, offsets, revcomp_matrix = ma.torch_compute_similarity_motifs(pwms, pwms, fill_logp_self = 64, min_sim = 5, padding = 0.25, infocont = False, bk_freq = 0.25, reverse_complement = True, metric = 'mse', verbose = False, device = 'cpu', exact = True, return_alignment = True)
    to = time.time()
    print('Torch module {:.2f}s'.format(to-ti))
    
    where = np.triu_indices(len(plen),1)
    for w in zip(where[0], where[1]):
        print(w, revcomp_matrix[w], offsets[w])
        print(correlation[w])
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
        #print(scd.correlation(newpwm[0].flatten(), newpwm[1].flatten()))
        #print(scd.cosine(newpwm[0].flatten(), newpwm[1].flatten()))
        print(np.mean((newpwm[0].flatten()- newpwm[1].flatten())**2))
        print(np.mean(newpwm[0].flatten()*newpwm[0].flatten() + newpwm[1].flatten()*newpwm[1].flatten()-2*newpwm[0].flatten()* newpwm[1].flatten()))
        print(np.sum(newpwm[0].flatten()*newpwm[0].flatten() + newpwm[1].flatten()*newpwm[1].flatten()), np.mean(newpwm[0].flatten()* newpwm[1].flatten()))
        '''
        newpwm /= np.sqrt(np.mean(newpwm**2, axis = (1,2)))[:,None,None]
        newpwm = np.transpose(newpwm, axes=(0,2,1))
        npwm0, npwm1 = torch.tensor(newpwm[[0]]), torch.tensor(newpwm[[1]])
        print(npwm0, npwm0.shape)
        print(ma.padded_weight_conv1d(npwm0, npwm1, clen))
        
        npwm0, npwm1 = torch.tensor(np.copy(pwms[w[1]])).to(torch.float32).transpose(-1,-2).unsqueeze(0), torch.tensor(np.copy(fpwm).T).to(torch.float32).unsqueeze(0)

        print(ma.padded_weight_conv1d(npwm0, npwm1, 5, standard = True, verbose=False))
        '''
        plt.show()
    
    print(np.sum(np.absolute(doffsets))/np.prod(np.shape(doffsets)))
    print(np.sum(np.absolute(drevcomp_matrix)))
        
    
    
    
    
