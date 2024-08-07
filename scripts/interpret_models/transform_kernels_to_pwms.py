'''
Read kernel parameters from meme file and use activation functions and random
seqlets to create motifs from most activated seqlets
'''


import numpy as np
import sys, os
import torch 
from drg_tools.interpret_cnn import kernel_to_ppm
from drg_tools.modules import func_dict_single as func_dict

import time

from drg_tools.io_utilis import read_motifs, readinfasta, write_meme_file
from drg_tools.io_utils import readin_sequence_return_onehot as readinseqs
from drg_tools.sequence_utils import split_seqs, generate_random_onehot
from drg_tools.interpret_cnn import kernels_to_pwms_from_seqlets as kernels_to_pwms


if __name__ == '__main__':

    np.random.seed(1)
    pth=sys.argv[1]

    weights, motifnames, biases = read_motifs(pth, dtype = 'meme')
    weights = np.transpose(weights, axes = (0,2,1))
    
    if biases is None:
        biases = np.zeros(len(weights))
    
    outname = os.path.splitext(pth)[0].replace('_kernelweights','')

    l_kernels = np.shape(weights)[-1]
    
    if '--sequences' in sys.argv:
        # one-hot encoded or fasta file
        outname += '_seqset'
        seqfile = sys.argv[sys.argv.index('--sequences')+1]
        seqfullseqs = readinseqs(seqfile)
        if '--nrandom' in sys.argv:
            nseq = int(sys.argv[sys.argv.index('--nrandom')+1])
            outname += str('{:.1f}'.format(nseq/1000)+'K')
            seqfullseqs = seqfullseqs[np.random.permutation(len(seqfullseqs))[:nseq]]
            
        seq_seq = split_seqs(seqfullseqs, l_kernels)
        print(np.shape(seq_seq))
        if '--reverse_complement' in sys.argv:
            seq_seq = np.append(seq_seq, seq_seq[:,::-1][:,:,::-1], axis = 0)
            outname += 'rc'
        print(np.shape(seq_seq))
        if '--unique' in sys.argv:
            seq_seq = np.unique(seq_seq, axis = 0)
            outname += 'unq'
            print('Unique seqlets', np.shape(seq_seq))
        seq_seq = np.transpose(seq_seq, axes = (0,2,1))
        
    
    else:
        nseq = 200000
        if '--nrandom' in sys.argv:
            nseq = int(sys.argv[sys.argv.index('--nrandom')+1])
            outname += str('{:.1f}'.format(nseq/1000000)+'M')
        seq_seq = generate_random_onehot(l_kernels, nseq) # generate 200000 random sequences
        print(np.shape(seq_seq))
    
    if '--transform' in sys.argv:
        ppms = kernel_to_ppm(ppms[:,:,:], kernel_bias =biases)
        write_meme_file(ppms, motifnames, 'ACGT', outname+'_kernel_ppms'+'.meme')    
    else:
        a_func = func_dict['Id0']
        if '--activated' in sys.argv:
            actfunc = sys.argv[sys.argv.index('--activated')+1]
            outname += actfunc
            a_func = func_dict[actfunc]
        
        zscore = True
        maxact = 1.64
        if '--cutoff' in sys.argv:
            maxact = float(sys.argv[sys.argv.index('--cutoff')+1])
            outname += 'cut'+str(maxact)
        if '--maxactivation' in sys.argv:
            maxact=float(sys.argv[sys.argv.index('--maxactivation')+1])
            zscore = False
            outname += 'max'+str(maxact)
        
        device = 'cpu'
        if '--device' in sys.argv:
            device = sys.argv[sys.argv.index('--device')+1]

        print(outname)
        bsize = None
        if '--batchsize' in sys.argv:
            bsize = int(sys.argv[sys.argv.index('--batchsize')+1])
        
        pwms = kernels_to_pwms(weights, seq_seq, maxact, biases = biases, activation_func = a_func, zscore = zscore, device = device, batchsize = bsize)

        
        write_meme_file(np.around(pwms,3), motifnames, 'ACGT', outname+'_kernel_pwms.meme')

