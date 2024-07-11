import numpy as np
import sys, os
import torch 
from interpret_cnn import write_meme_file, pfm2iupac, kernel_to_ppm, pwms_from_seqs, genseq
from modules import func_dict
from data_processing import numbertype
# Read text files with PWMs
def read_motifs(pwmlist, nameline = 'Motif', delimiter = '\t', alphabet_line = 'Pos', dtype = 'txt'):
    names = []
    pwms = []
    pwm = []
    other = []
    obj = open(pwmlist, 'r').readlines()
    if dtype == 'meme':
        nameline = "MOTIF"
        delimiter = None
        alphabet_line = 'ALPHABET='
    for l, line in enumerate(obj):
        line = line.strip().split(delimiter)
        if ((len(line) == 0) or (line[0] == '')) and len(pwm) > 0:
            pwm = np.array(pwm, dtype = float)
            pwms.append(np.array(pwm))
            pwm = []
            names.append(name)
        elif len(line) > 0:
            if line[0] == nameline:
                name = line[1]
                pwm = []
            elif line[0][:len(alphabet_line)] == alphabet_line:
                nts = list(line[1:])
            elif dtype == 'txt':
                if isinstance(numbertype(line[0]), int):
                    pwm.append(line[1:])
            elif dtype == 'meme':
                if isinstance(numbertype(line[0]), float):
                    pwm.append(line)
                if 'bias=' in line:
                    other.append(float(line[line.index('bias=')+1]))
    
    if len(pwm) > 0:
        pwms.append(np.array(pwm))
        names.append(name)
    if len(other) == 0:
        other = None
    else:
        other = np.array(other)
    pwms, names = np.array(pwms, dtype = float), np.array(names)
    return pwms, names, other



if __name__ == '__main__':


    pth=sys.argv[1]

    weights, motifnames, biases = read_motifs(pth, dtype = 'meme')
    weights = np.transpose(weights, axes = (0,2,1))
    
    outname = os.path.splitext(pth)[0].replace('_kernelweights','')

    l_kernels = np.shape(weights)[-1]
    seq_seq = genseq(l_kernels, 200000) # generate 200000 random sequences
    
    if '--transform' in sys.argv:
        ppms = kernel_to_ppm(ppms[:,:,:], kernel_bias =biases)
        write_meme_file(ppms, motifnames, 'ACGT', outname+'_kernel_ppms'+'.meme')    
    else:
        seqactivations = np.sum(weights[:,None]*seq_seq[None,...],axis = (2,3))
        if biases is not None:
            seqactivations += biases[:,None]
        
        if '--activated' in sys.argv:
            actfunc = sys.argv[sys.argv.index('--activated')+1]
            outname += actfunc
            a_func = func_dict[actfunc]
            seqactivations = a_func(torch.Tensor(seqactivations)).cpu().detach().numpy()
        
        zscore = True
        maxact = 2.326
        if '--maxactivation' in sys.argv:
            maxact=float(sys.argv[sys.argv.index('--maxactivation')+1])
            zscore = False
            outname += 'max'+str(maxact)
            
        pwms = pwms_from_seqs(seq_seq, seqactivations, maxact, z_score = zscore)
        write_meme_file(np.around(pwms,3), motifnames, 'ACGT', outname+'_kernel_pwms.meme')

