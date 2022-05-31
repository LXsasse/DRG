import numpy as np
import sys
import torch
import torch.nn as nn



# Generates a list of pwms with targetlen from a pwm with different or same length
# for example, creates two pwms of length 7 from pwm of length 8
def pwmset(pwm, targetlen):
    if np.shape(pwm)[-1] == targetlen:
        return pwm[None, :]
    elif np.shape(pwm)[-1] > targetlen:
        npwms = []
        for l in range(np.shape(pwm)[-1]-targetlen+1):
            npwm = np.zeros((np.shape(pwm)[0], targetlen))
            npwm[:] = pwm[:, l:l+targetlen]
            npwms.append(npwm)
        return np.array(npwms)
    elif np.shape(pwm)[-1] < targetlen:
        npwms = []
        for l in range(targetlen - np.shape(pwm)[-1]+1):
            npwm = np.zeros((np.shape(pwm)[0], targetlen))
            npwm[:,l:l+np.shape(pwm)[-1]] = pwm[:]
            npwms.append(npwm)
        return np.array(npwms)
        
# Scans onehot encoded numpy array for pwms
def pwm_scan(sequences, pwms, targetlen = None, activation = 'max', motif_cutoff = None, set_to = 0., verbose = False):
    # if pwms are longer than the targeted kernel size then its unclear if we should the convolution with the right side of the pwm or the left side. Each would create  a different positional pattern. Therefore we take the mean over both options all options of pwms with the target length.
    # If Pwms are smaller than the target len we use all the options of padded pwms to create scanning pattern
    if targetlen is None:
        targetlen = np.amax([len(pqm.T) for pqm in pwms])
    outscan = np.zeros((np.shape(sequences)[0], len(pwms), np.shape(sequences)[-1]-targetlen+1))
    if verbose:
        print('Scanning', len(sequences), 'sequences with', len(pwms), 'PWMs')
    for p, pwm in enumerate(pwms):
        if p%10 == 0 and verbose:
            print(p)
        setps = pwmset(pwm, targetlen = targetlen)
        for l in range(np.shape(sequences)[-1]-targetlen+1):
            setscans = np.sum(sequences[:,None,:,l:l+targetlen] * setps[None, :, :, :], axis = (-1, -2))
            # max activation across all shorter subpwms
            if activation == 'max':
                outscan[:,p, l] = np.amax(setscans, axis = -1)
            elif activation == 'mean':
                outscan[:,p, l] = np.mean(setscans, axis = -1)
    # pwms also assign values to partial fits of the sequence, to remove these partial fits one can 
    if motif_cutoff is not None:
        outscan[outscan < motif_cutoff] = set_to
    return outscan
        

# The prediction after training are performed on the cpu
def batched_predict(model, X, pwm_out = None, mask = None, device = 'cpu', batchsize = None):
    model.eval()
    if device is None:
        device = model.device
    if device != model.device:
        model.to(device)
    # Use no_grad to avoid computation of gradient and graph
    with torch.no_grad():
        X = torch.Tensor(X)
        if batchsize is not None:
            predout = []
            for i in range(0, int(X.size(dim=0)/batchsize)+int(X.size(dim=0)%batchsize != 0)):
                if pwm_out is not None:
                    pwm_outin = pwm_out[i*batchsize:(i+1)*batchsize]
                    pwm_outin = pwm_outin.to(device)
                else:
                    pwm_outin = None
                xin = X[i*batchsize:(i+1)*batchsize]
                xin = xin.to(device)
                predout.append(model.forward(xin, xadd = pwm_outin, mask = mask).detach().cpu().numpy())
            predout = np.concatenate(predout, axis = 0)
        else:
            X = X.to(device)
            predout = model.forward(X, xadd = pwm_out, mask = mask)
            predout.detach().cpu().numpy()
    return predout


