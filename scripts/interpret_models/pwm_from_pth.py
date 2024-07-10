import numpy as np
import sys, os
import torch 
from interpret_cnn import write_meme_file, pfm2iupac, kernel_to_ppm, compute_importance, pwms_from_seqs, genseq
from data_processing import numbertype

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
        
print(l_kernels)
seq_seq = genseq(l_kernels, 200000) # generate 200000 random sequences

ppms = []
pwms = []
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
        # compute motif means from the activation of kernels with the random sequences.
        seqactivations = np.sum(ppms[-1][:,None]*seq_seq[None,...],axis = (2,3))
        # generate motifs from aligned sequences with activation over 0.9 of the maximum activation
        pwms.append(pwms_from_seqs(seq_seq, seqactivations, 2.326))

        i += 1
                
motifnames = np.concatenate(motifnames)
ppms = np.concatenate(ppms, axis = 0)
weights = np.concatenate(weights, axis = 0)
biases = np.concatenate(biases, axis = 0)
# create ppms directly from kernel matrix
ppms = np.around(kernel_to_ppm(ppms[:,:,:], kernel_bias =biases),3)


# generate pwms from most activated sequences
pwms = np.around(np.concatenate(pwms, axis = 0),3)
weights = np.around(np.concatenate(weights, axis = 0),6)

write_meme_file(ppms, motifnames, 'ACGT', outname+'_kernel_ppms'+addppmname+'.meme')
write_meme_file(pwms, motifnames, 'ACGT', outname+'_kernel_pwms'+addppmname+'.meme')
write_meme_file(pwms, motifnames, 'ACGT', outname+'_kernelweights'+addppmname+'.meme', biases = biases)
