from torch.utils.data import TensorDataset, DataLoader, Dataset
import os, sys
import numpy as np
import torch
import torch.nn as nn
from modules import cosine_loss, cosine_both, correlation_loss, correlation_both



def load_parameters(model, PATH, translate_dict = None, allow_reduction = False, exclude = [], include = False):
    if isinstance(PATH, str):
        state_dict = torch.load(PATH, map_location = 'cpu')
    else:
        state_dict = PATH
    cstate_dict = model.state_dict()
    for n, name0 in enumerate(cstate_dict):
        name = name0
        #print(name)
        if translate_dict is not None:
            if name in translate_dict:
                name = translate_dict[name]
        #print(name, state_dict.keys())
        if name in state_dict and ((name0 in exclude) == include):
            print("Loaded", name0 ,'with', name)
            ntens = None
            if cstate_dict[name0].size() == state_dict[name].size():
                cstate_dict[name0] = state_dict[name]
            elif allow_reduction:
                if (cstate_dict[name0].size(dim = 0) > state_dict[name].size(dim = 0)) and ((cstate_dict[name0].size(dim = -1) == state_dict[name].size(dim = -1)) or ((len(cstate_dict[name0].size()) ==1) and (len(state_dict[name].size())==1))):
                    ntens = cstate_dict[name0]
                    ntens[:state_dict[name].size(dim = 0)] = state_dict[name]
                elif cstate_dict[name0].size(dim = 0) <= state_dict[name].size(dim = 0) and ((cstate_dict[name0].size(dim = -1) == state_dict[name].size(dim = -1)) or ((len(cstate_dict[name0].size()) ==1) and (len(state_dict[name].size())==1))):
                    ntens = state_dict[name][:cstate_dict[name0].size(dim = 0)]
                cstate_dict[name0] = ntens
            
    model.load_state_dict(cstate_dict)
    

# For custom data sets that that also return the indices for the batch
class MyDataset(Dataset):
    def __init__(self, data, targets, axis = 0):
        self.data = data
        self.targets = targets
        self.axis = axis
        
    def __getitem__(self, index):
        y = self.targets[index]
        if self.axis == 0:
            x = self.data[index]
        elif self.axis == 1:
            x = [dx[index] for dx in self.data]
        return x, y, index
        
    def __len__(self):
        return len(self.targets)
        
# Either use cpu or use gpu with largest free memory
def get_device():
    tot = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Total').readlines()
    used = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used').readlines()
    tot = np.array([int(x.split()[2]) for x in tot])
    used = np.array([int(x.split()[2]) for x in used])
    memory_available = tot - used
    if len(memory_available) > 0:
        bcuda = np.argmax(memory_available)
    else:
        bcuda = 0
    device = torch.device("cuda:"+str(bcuda) if torch.cuda.is_available() else "cpu")
    return device



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

# Kernel initialization with statistical motif enrichment or Lasso regression
def kernel_hotstart(n_kernels, l_kernels, X, Y, kmervalues = 'Lasso', XYval = None, verbose = True, alpha = 1.):
    # minimal occurance of motif in sequence
    n_minimal = 200.
    # maximal length of initiated patterns is capped to 7, l_kernels or statistical number of different motifs
    maxlen = min(7,min(int(np.log(np.shape(X)[0]*np.shape(X)[-1]/n_minimal)/np.log(4))+1, l_kernels))
    print('Hotstart with', maxlen, 'mers')
    # Generate k-merlist from onehot encoding, and count occurrance, return occurance matrix
    kmernumbers, allkmers = kmer_count(X, maxlen)
    
    if XYval is not None:
        Xval, Yval = XYval[0], XYval[1]
        kmernumbersval, allkmers = kmer_count(Xval, maxlen, allkmers = allkmers)
    
    zscores = np.zeros((len(allkmers), np.shape(Y)[-1]))
    if kmervalues == 'Lasso':
        kmernumbers = (kmernumbers-np.mean(kmernumbers,axis = 0))/np.std(kmernumbers, axis = 0)
        # Start regularization for Lasso
        while True:
            print('Hotstart with Lasso', alpha, "Classes left", len(np.where(np.sum(zscores, axis = 0) == 0)[0]))
            model = Lasso(fit_intercept = True, alpha = alpha, warm_start = False, max_iter = 250)
            model.fit(kmernumbers, Y)
            modpred = model.predict(kmernumbers)
            if len(np.shape(modpred)) == 1:
                modpred = modpred.reshape(-1,1)
            print('Predicted correlation', ' '.join(np.around(np.diagonal(cdist(modpred.T, Y.T, 'correlation')),3).astype(str)))
            coef_ = model.coef_
            if len(np.shape(coef_)) ==1:
                coef_ = coef_.reshape(1,-1)
            ndetected = []
            for yi, ycoef in enumerate(np.absolute(coef_)):
                ndetected.append(int(np.sum(ycoef > 0)))
                if np.sum(ycoef != 0) > n_kernels and np.sum(zscores[:,yi]) == 0:
                    zscores[:, yi] = ycoef*alpha
            print('Ndtected', ' '.join(np.array(ndetected).astype(str)))
            if (np.sum(zscores, axis = 0) == 0).any():
                alpha = alpha/2.
            else:
                break
      
    zscores = np.amax(zscores,axis = 1)
    chosenk = np.argsort(-zscores)[:n_kernels]
    select_kmers = allkmers[chosenk]
    if verbose:
        print('Initialized kernels', select_kmers)
        theomodel = LinearRegression(fit_intercept = True).fit(kmernumbers[:, chosenk], Y)
        theopred = theomodel.predict(kmernumbers[:, chosenk])
        if len(np.shape(theopred)) == 1:
            theopred = theopred.reshape(-1,1)
        print('Final predicted correlation with chosenk', np.diagonal(cdist(theopred.T, Y.T, 'correlation')))
        
        if XYval is not None:
            theopred = theomodel.predict(kmernumbersval[:, chosenk])
            if len(np.shape(theopred)) == 1:
                theopred = theopred.reshape(-1,1)
            print('Final validation correlation with chosenk', np.diagonal(cdist(theopred.T, Yval.T, 'correlation')))
    # Return pwms for top k-mers
    hotpwms = np.array([pwm_from_kmer(km, l_pwm = l_kernels) for km in select_kmers])
    return hotpwms

        

    
