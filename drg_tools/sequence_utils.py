# sequence_utils.py
'''
Contains functions to manipulate sequences or their onehot encodings
Most functions work with numpy arrays

'''
import numpy as np


def avgpool(x,window):
    lx = np.shape(x)[-1]
    if lx%window!=0:
        xtend = [int(np.floor((lx%window)/2)), int(np.ceil((lx%window)/2))]
        x = np.pad(x, pad_width = [[0,0],[0,0],xtend])
    lx = np.shape(x)[-1]
    xavg = np.zeros(list(np.shape(x)[:-1])+[int(lx/window)])
    for i in range(int(lx/window)):
        xavg[..., i] = np.mean(x[..., i*window:(i+1)*window], axis = -1)
    return xavg
    

def realign(X):
    '''
    Shifts a onehot encoded sequence with zeros on one end,
    f.e. transcripts of different length, to right side of 
    the array
    '''
    end_of_seq = np.sum(X,axis = (1,2)).astype(int)
    Xmirror = []
    lx = np.shape(X)[-2]
    for s, es in enumerate(end_of_seq):
        xmir = np.zeros(np.shape(X[s]), dtype = np.int8)
        xmir[lx-es:] = X[s,:es]
        Xmirror.append(xmir)
    return np.append(X, np.array(Xmirror), axis = -2)

def generate_random_onehot(lseq, nseq):
    seqs = np.zeros((nseq,4,lseq))
    pos = np.random.randint(0,4,lseq*nseq)
    pos0 = (np.arange(lseq*nseq,dtype=int)/lseq).astype(int)
    pos1 = np.arange(lseq*nseq,dtype=int)%lseq
    seqs[pos0,pos,pos1] = 1
    return seqs

def seqlen(arrayofseqs):
    '''
    Returns array with length of strings in arrayofseqs
    '''
    return np.array([len(seq) for seq in arrayofseqs])

def check_addonehot(onehotregion, shapeohvec1, selen):
    # check conditions if one-hot encoded regions can be added
    addonehot = False
    if onehotregion is not None:
        if np.shape(onehotregion)[1] == shapeohvec1 :
            addonehot = True
        else:
            print("number of genetic regions do not match sequences")
            sys.exit()
    return addonehot

# generates one-hot encoding by comparing arrays
def quick_onehot(sequences, nucs = 'ACGT', wildcard = None, onehotregion = None, align = 'left'):
    selen = seqlen(sequences)
    nucs = np.array(list(nucs))
    if align == 'bidirectional':
        mlenseqs = 2*np.amax(selen) + 20
    else:
        mlenseqs = np.amax(selen) 
    ohvec = np.zeros((len(sequences),mlenseqs, len(nucs)), dtype = np.int8)
    
    if wildcard is not None:
        nucs.append(wildcard)
    nucs = np.array(nucs)
    # check conditions if one-hot encoded regions can be added
    addonehot = check_addonehot(onehotregion, mlenseqs, selen)

    if align == 'left' or align == 'bidirectional':
        for s, sequence in enumerate(sequences):
            ohvec[s][:len(sequence)] = np.array(list(sequence))[:, None] == nucs
    if align == 'right' or align == 'bidirectional':
        for s, sequence in enumerate(sequences):
            ohvec[s][-len(sequence):] = np.array(list(sequence))[:, None] == nucs
    ohvec = ohvec.astype(np.int8)
    
    if addonehot:
        ohvec = np.append(ohvec, onehotregion, axis = -1)
    return ohvec, nucs


def reverse_complement(X):
    X = np.append(X,X[:,::-1], axis = 1)
    return X

# given a pwm, generate a kmer sequence for largest frequencies along the length of the pwm   
def kmer_from_pwm(pwm, nts = 'ACGT', axis = None):
    nts = np.array(list('ACGT'))
    if axis is None:
        axis = np.where(np.array(np.shape(pwm)) == len(nts))[0][-1]
    if axis == 0:
        pwm = pwm[:,np.sum(np.absolute(pwm).T,axis = 1)>0]
    else:
        pwm = pwm[np.sum(np.absolute(pwm),axis = 1)>0]
    kmer = ''.join(nts[np.argmax(pwm, axis = axis)])
    return kmer

# given a kmer sequence, generate a one-hot encoded pwm and normalize to 1.
def pwm_from_kmer(kmer, nts = 'ACGT', l_pwm = None):
    pwm = np.array(list(kmer))[None, :] == np.array(list(nts))[:,None]
    if l_pwm is None:
        pwm = pwm.astype(float)/np.sum(pwm)
    else:
        npwm = np.zeros((np.shape(pwm)[0], l_pwm))
        npwm[:, int((l_pwm-np.shape(pwm)[1])/2):int((l_pwm-np.shape(pwm)[1])/2)+np.shape(pwm)[1]] = pwm.astype(float)/np.sum(pwm)
        pwm = npwm
    return pwm

# Counts k-mers of lenght l_kmer in onehot encoded sequence
# K-mers that are looked for can be given as allkmers
def kmer_count(onehot, l_kmer, allkmers = None):
    kmerlist = []
    for s, seq in enumerate(onehot):
        seq = seq.T
        seq = kmer_from_pwm(seq)
        seqlist = []
        for p in range(len(seq)-l_kmer+1):
            seqlist.append(seq[p:p+l_kmer])
        kmerlist.append(seqlist)
    if allkmers is None:
        allkmers = np.unique(np.concatenate(kmerlist))
    kmernumbers = np.zeros((len(onehot), len(allkmers)))
    for s, slist in enumerate(kmerlist):
        slist, n = np.unique(slist, return_counts = True)
        kmernumbers[s,np.isin(allkmers, slist)] = n[np.isin(slist, allkmers)]
    return kmernumbers, allkmers



