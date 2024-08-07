
'''
extract kernel motifs from pth filter
and save as different forms
'''

import numpy as np
import sys, os
import torch 

from drg_tools.io_utils import write_meme_file, pfm2iupac, numbertype
from drg_tools.interpret_cnn import kernel_to_ppm, pwms_from_seqs, extract_kernelweights_from_state_dict

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
    
    weights, biases, motifnames = extract_kernelweights_from_state_dict(state_dict, kernel_layer_name = 'convolutions.conv1d', full_name = False)
    # create ppms directly from kernel matrix
    ppms = np.around(kernel_to_ppm(weights[:,:,:], kernel_bias =biases),3)
    weights = np.around(np.concatenate(weights, axis = 0),6)

    write_meme_file(ppms, motifnames, 'ACGT', outname+'_kernel_ppms'+addppmname+'.meme')
    write_meme_file(pwms, motifnames, 'ACGT', outname+'_kernelweights'+addppmname+'.meme', biases = biases)
