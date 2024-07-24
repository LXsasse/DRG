
'''
extract kernel motifs from pth filter
and save as different forms
'''

import numpy as np
import sys, os
import torch 

from drg_tools.io_utils import write_meme_file, pfm2iupac, numbertype
from drg_tools.interpret_cnn import kernel_to_ppm, pwms_from_seqs

if __name__ == '__main__':
        
    pth=sys.argv[1]

    ppmparal = False
    addppmname = ''
    if len(sys.argv) > 2:
        stppm = numbertype(sys.argv[2])
        Nppm = numbertype(sys.argv[3])
        if isinstance(stppm, int) and isinstance(Nppm, int):
            ppmparal = True
            addppmname = str(stppm)+'-'+str(Nppm + stppm)


    outname = os.path.splitext(pth)[0].replace('_parameter','')

    state_dict = torch.load(pth, map_location = 'cpu')
    has_bias = False
    for d in state_dict:
        if 'convolutions.conv1d.weight' in d:
            l_kernels = state_dict[d].size(dim=-1)
        if 'convolutions.conv1d.bias' in d:
            has_bias = True

    ppms = []
    weights = []
    biases = []
    motifnames = []

    i =0
    for namep in state_dict:
        if namep.split('.')[-2] == 'conv1d' and namep.split('.')[-3] == 'convolutions' and namep.rsplit('.')[-1] == 'weight':
            print(i, namep, state_dict[namep].size())
            # collect the first layer convolution kernels
            kernelweight = state_dict[namep].detach().cpu().numpy()
            if ppmparal: 
                kernelweight = kernelweight[stppm : stppm+Nppm]
            else:
                stppm, Nppm = 0, len(kernelweight)
            print(np.shape(kernelweight))
            ppms.append(kernelweight)
            weights.append(kernelweight)
            # collect the biases if bias is not None
            if has_bias:
                bias = state_dict[namep.replace('weight', 'bias')].detach().cpu().numpy()[stppm : Nppm + stppm]
            else:
                bias = np.zeros(len(ppms[-1]))
            biases.append(bias)
            # Generate names for all kernels
            motifnames.append(np.array(['filter'+str(j+stppm)+'_'+namep.split('.')[1] for j in range(len(ppms[-1]))]))
            i += 1
                    
    motifnames = np.concatenate(motifnames)
    ppms = np.concatenate(ppms, axis = 0)
    weights = np.concatenate(weights, axis = 0)
    biases = np.concatenate(biases, axis = 0)
    # create ppms directly from kernel matrix
    ppms = np.around(kernel_to_ppm(ppms[:,:,:], kernel_bias =biases),3)
    weights = np.around(np.concatenate(weights, axis = 0),6)

    write_meme_file(ppms, motifnames, 'ACGT', outname+'_kernel_ppms'+addppmname+'.meme')
    write_meme_file(pwms, motifnames, 'ACGT', outname+'_kernelweights'+addppmname+'.meme', biases = biases)
