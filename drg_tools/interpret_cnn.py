# interpret_cnn.py

'''
functions to interpret the kernels of models
and to create sequence attributions from the model

'''

import numpy as np
import sys, os
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as F
import time 

from .modules import func_dict, SoftmaxNorm
from .sequence_utils import generate_random_onehot
from .motif_analysis import pfm2iupac
from .stats_functions import correlation, mse

def kernel_to_ppm(kernels, kernel_bias = None, bk_freq = None):
    n_kernel, n_input, l_kernel = np.shape(kernels)
    if kernel_bias is not None:
        kernels += kernel_bias[:,None,None]/(n_input*l_kernel)
    # strechting factor to increase impact of large parameter values
    # alternatively use: kernels *= l_kernel
    kernels /= np.std(kernels, axis = (-1,-2))[...,None, None]
    if bk_freq is None:
        bk_freq = np.ones(n_input)*np.log2(1./float(n_input))
    elif isinstance(bk_freq, float) or isinstance(bk_freq, int):
        bk_freq = np.ones(n_input)*np.log2(1./float(bk_freq))
    kernels += bk_freq[None,:,None]
    ppms = 2.**kernels
    ppms = ppms/np.amax(np.sum(ppms, axis = 1),axis = -1)[:,None, None] #[:,None, :]
    return ppms



def pwms_from_seqs(ohseqs, activations, cut, z_score = True):
    # scale kernel activations between 0 and 1
    minact = np.amin(activations, axis = 1)
    activations = activations - minact[:,None]
    maxact = np.amax(activations, axis = 1)
    activations = activations/maxact[:,None]
    
    if z_score:
        # instead of rescaling the activations, rescale the cut parameter
        cut = (np.mean(activations, axis = 1)+cut*np.std(activations,axis =1))[:, None]
    seqs = np.where(activations >= cut)
    pwms = []
    for a, act in enumerate(activations):
        mask = seqs[1][seqs[0]==a]
        chseq = act[mask][:,None,None]*ohseqs[mask]
        pwms.append(np.sum(chseq, axis = 0)/np.sum(chseq, axis = (0,1))[None, :])
        
        # check distribution
        #bins = np.linspace(0,1,21)
        #plt.hist(act, bins = bins)
        #plt.hist(act[seqs[1][seqs[0]==a]],bins = bins, color='orange',zorder = 1)
        #plt.show()
    return np.array(pwms)
 


# IDEA: Use this to propagate importance through network and determine individual nodes and then interpret them.
# perform zscore test for each parameter as in linear regression
def parameter_importance(ypred, y, coef_, inputs):
    invcovardiag = np.diagonal(np.pinv(np.dot(inputs.T, inputs),rcond = 1e-8, hermetian = True))
    loss = np.mean((ypred-y)**2, axis = 0)
    var_b = loss[None,:]*(invcovardiag)[:,None]
    sd_b = np.sqrt(var_b)
    z_scores = coef_ * sd_b
    z_scores = np.nan_to_num(z_scores)
    return z_scores




def correct_by_mean(grad):
    return grad - np.mean(grad, axis = -2)[...,None,:]



def ism(x, model, tracks, correct_from_neutral = True):
    baselines = model.predict(x)
    if isinstance(baselines, list):
        baselines = np.concatenate(baselines, axis = 1)
    baselines = baselines[..., tracks]
    if isinstance(x,list):
        x = [torch.Tensor(xi) for xi in x]
    else:
        x = torch.Tensor(x)
    if isinstance(x, list):
        Nin = len(x)
        ismout = []
        for n in range(Nin):
            xi = x[n]
            size = list(np.shape(xi))
            size += [len(tracks)]
            ismout0 = np.zeros(size)
            for i, si in enumerate(xi):
                isnot = torch.where(si == 0)
                # Needs cloning otherwise no changes can be made to si (don't know why)
                xalt = torch.clone(si.expand([len(isnot[0])] + list(si.size())))
                for j in range(len(isnot[0])):
                    xalt[j,:,isnot[1][j]] = 0
                    xalt[j,isnot[0][j],isnot[1][j]] = 1
                xin = []
                for k in range(Nin):
                    if k != n: 
                        xin.append(torch.clone(x[k][i].expand([len(isnot[0])] + list(si.size()))))
                    else:
                        xin.append(xalt)
                altpred = model.predict(xin)
                if isinstance(altpred,list):
                    altpred = np.concatenate(altpred,axis =1)
                altpred = altpred[...,tracks]
                for j in range(len(isnot[0])):
                    ismout0[i, isnot[0][j], isnot[1][j]] = altpred[j] - baselines[i]
            ismout.append(np.swapaxes(np.swapaxes(ismout0, 1, -1), -1,-2))
            if correct_from_neutral:
                ismout[-1] = correct_by_mean(ismout[-1])
    else:
        Nin = 1
        size = list(np.shape(x))
        size += [len(tracks)]
        ismout = np.zeros(size)
        start = time.time()
        for i, si in enumerate(x):
            #beginning = time.time()
            isnot = torch.where(si == 0)
            # Needs cloning otherwise no changes can be made to si (don't know why)
            xalt = torch.clone(si.expand([len(isnot[0])] + list(si.size())))
            for j in range(len(isnot[0])):
                xalt[j,:,isnot[1][j]] = 0
                xalt[j,isnot[0][j],isnot[1][j]] = 1
            #before = time.time()
            #print('making seqs', before - beginning)
            altpred = model.predict(xalt)
            #after = time.time()
            #print('forwards', after - before, xalt.size(), model.batchsize)
            if isinstance(altpred,list):
                altpred = np.concatenate(altpred,axis =1)
            altpred = altpred[...,tracks]
            for j in range(len(isnot[0])):
                ismout[i, isnot[0][j], isnot[1][j]] = altpred[j] - baselines[i]
            #afterall = time.time()
            #print('sorting', after-afterall)
        end = time.time()
        print('ISM time for', x.size(dim = 0), len(tracks),  end-start)
        ismout = np.swapaxes(np.swapaxes(ismout, 1, -1), -1,-2)
        if correct_from_neutral:
            ismout = correct_by_mean(ismout)
    return ismout
                                                     

def takegrad(x, model, tracks, ensemble = 1, correct_from_neutral = True, top=None):
    grad = []
    if isinstance(x, list):
        x = [torch.Tensor(xi) for xi in x]
    else:
        x = torch.Tensor(x)
    if ensemble > 1:
        model.train()
        
    if isinstance(x,list):
        Nin = len(x)
        grad = [[] for n in range(Nin)]
        for i in range(x[0].size(dim = 0)):
            xi = []
            for n in range(Nin):
                xij = torch.clone(x[n][[i]])
                xij.requires_grad = True
                xi.append(xij)
            gra = [[[] for e in range(ensemble)] for n in range(Nin)]
            for e in range(ensemble):
                pred = model.predict(xi, enable_grad = True)
                if isinstance(pred, list):
                    pred = torch.cat(pred, axis = 1)
                for t, tr in enumerate(tracks):
                    pred[0, tr].backward(retain_graph = True)
                    for n, xij in enumerate(xi):
                        gr = xij.grad.clone().cpu().numpy()
                        gra[n][e].append(gr)
                        xij.grad.zero_()
            
            for n in range(Nin):
                gra[n] = np.mean(gra[n],axis = 0)
                gra[n] = np.concatenate(gra[n],axis = 0)
                if correct_from_neutral:
                    gra[n] = correct_by_mean(gra[n])
                if top is not None:
                    ngrashape = list(np.shape(gra[n]))
                    ngrashape[-2] += 1
                    ngrashape[-1] = top
                    ngra = np.zeros(ngrashape)
                    for t in range(np.shape(gra[n])[0]):
                        # change this to largest at reference
                        #print(n, t, np.amax(np.absolute(gra[n][t]), axis = -1))
                        lcn = np.argsort(-np.amax(np.absolute(gra[n][t]*x[n][i].numpy()),axis = -2))
                        #print(gra[n][t][:,lcn[:10]])
                        lcn = np.sort(lcn[:top])
                        #print(len(lcn))
                        ngra[t] = np.append(gra[n][t][...,lcn], lcn[None, :], axis = -2)
                        #print(n, t, np.amax(np.absolute(ngra[t]), axis = -1))
                    gra[n] = ngra
                
                #for t in range(np.shape(gra[n])[0]):
                    #print(n, t, np.amax(np.absolute(gra[n][t]), axis = -1))
                    
                grad[n].append(gra[n])
                
                    
        for n in range(Nin):
            grad[n] = np.array(grad[n])
            
            
    else:
        start = time.time()
        for i in range(x.size(dim = 0)):
            #checkcorr = []
            xi = torch.clone(x[[i]])
            xi.requires_grad = True
            gra = [[] for e in range(ensemble)]
            for e in range(ensemble):
                pred = model.predict(xi, enable_grad = True)
                if isinstance(pred, list):
                    pred = torch.cat(pred, axis = 1)
                for t, tr in enumerate(tracks):
                    #print(i,tr,round(float(pred[0,tr]),2))
                    pred[0, tr].backward(retain_graph = True)
                    gr = xi.grad.clone().cpu().numpy()
                    gra[e].append(gr)
                    xi.grad.zero_()
                    #print(np.amax(correct_by_mean(gra[-1])))
                    #checkcorr.append([float(pred[0,tr]), np.amax(correct_by_mean(gra[-1]))])
                #print(i,'corr', pearsonr(np.array(checkcorr)[:,0],np.array(checkcorr)[:,1])) 
            gra = np.mean(gra, axis = 0)
            grad.append(np.concatenate(gra,axis = 0))
        grad = np.array(grad)
        end = time.time()
        if correct_from_neutral:
            grad = correct_by_mean(grad)
        #for i in range(x.size(dim = 0)):
            #for t, tr in enumerate(tracks):
                #print(i, tr, np.amax(grad[i,t]))
        print('TISM time for', x.size(dim = 0), len(tracks),  end-start)
    return grad




def correct_deeplift(dlvec, freq):
    '''
    Compute hypothetical attributions from multipliers to a baseline frequency
    '''
    if len(np.shape(freq)) < len(np.shape(dlvec)):
        freq = np.ones_like(dlvec) * freq[:,None] # for each base, we have to "take step" from the frequency in baseline to 1.
    # This whole procedure is similar to ISM, where we remove the reference base and set it to zero, so take a step of -1, and then go a step of one in the direction of the alternative base. DeepLift returns the regression coefficients, which need to be scaled to the steps that are taken to arrive at the base we're looking at.
    negmul = np.sum(dlvec * freq, axis = -2)
    dlvec = dlvec - negmul[...,None,:]
    return dlvec



class cnn_multi_deeplift_wrapper(torch.nn.Module):
    '''
    Wrapper for cnn_multi to use tangermeme's deepshap on one of the inputs
    
    Parameters
    ----------
    model : pytorch.nn.module
        The cnn_multi object
    N : int 
        Number of sequences that the model takes as input
    n : int 
        The index of the sequence that will be investigated with the wrapper
        
    '''
    def __init__(self, model, N = 2, n = 0):
        super(Wrapper, self).__init__()
        self.model = model
        self.N = N
        self.n = n
    def forward(self, X, arg):
        '''
        X : torch.Tensor 
            is the sequence of interest
        arg : list of torch.Tensors 
            contains other sequences in order as given originally 
        '''
        x = []
        for i in range(N):
            t = 0
            if i == n:
                x.append(X)
                t = 1
            else:
                x.append(args[i-t])
        return self.model(x)



# TODO: Use the cnn_multi_deeplift_wrapper to get attributions for multi sequence models

def deeplift(x, model, tracks, baseline = None, batchsize = None, effective_attributions = True, raw_outputs = True, hypothetical = False):
    from tangermeme.deep_lift_shap import deep_lift_shap, _nonlinear, _maxpool
    from tangermeme.ersatz import dinucleotide_shuffle
    
    x = torch.Tensor(x)
    model.eval()
    
    # batch size will be taken from model
    if batchsize is None:
        batchsize = model.batchsize
    
    n_shuffles = None
    if baseline is None:
        basefreq = 1./x.size(-2)
        baseline = torch.ones_like(x)*basefreq
    
    elif isinstance(baseline, int):
        n_shuffles = baseline
        baseline = dinucleotide_shuffle
        raw_outputs = False
        hypothetical = True
        
    if baseline is not None:
        if isinstance(baseline, np.ndarray):
            baseline = torch.Tensor(baseline)
        if isinstance(baseline, torch.Tensor):
            if baseline.size() != x.size():
                # always need to expand baseline to number of data points to 
                baseline = baseline.unsqueeze(0).expand((x.size(0),) + tuple(baseline.size()))
                # if only frequencies were given, need to expand it along the length of the sequences
                if baseline.size() != x.size():
                    baseline = baseline.unsqueeze(-1).expand(tuple(baseline.size())+(x.size(-1),))
            baseline = baseline.unsqueeze(1)
    
    # when shift_sequence in my model is not None, inputs will be padded
    if model.shift_sequence is not None:
        if n_shuffles is not None:
            x = torch.nn.functional.pad(x, (model.shift_sequence, model.shift_sequence), mode = 'circular')
        else:
            x = torch.nn.functional.pad(x, (model.shift_sequence, model.shift_sequence), mode = 'constant', value = 0.25)
        
            if isinstance(baseline, torch.Tensor):
                baseline = torch.nn.functional.pad(baseline, (model.shift_sequence, model.shift_sequence), mode = 'constant', value = 0.25)

    #predx = model.forward(x.to(model.device)).cpu()
    #predbase = model.forward(baseline.squeeze(1).to(model.device)).cpu()
    
    
    grad = []
    for t, tr in enumerate(tracks):
        # if you want to represent certain custom functions with replacement functions use:
        #additional_nonlinear_ops = {Padded_AvgPool1d: _weightedpool}
        gr = deep_lift_shap(model, x, target = tr, references = baseline, n_shuffles = n_shuffles, device=model.device, raw_outputs = raw_outputs, hypothetical = hypothetical, additional_nonlinear_ops = {SoftmaxNorm: _nonlinear}).cpu().detach().numpy()
        if raw_outputs:
            gr = np.mean(gr, axis = 1)
        grad.append(gr)
        #deltas = (predx[:,[tr]] - predbase[:,[tr]]) - torch.sum(gr * (x.unsqueeze(1)-baseline), dim = (-1,-2))
        #print(tr, deltas)
    
    grad = np.array(grad)
    # return array if shape = (Ndatapoints, Ntracks, Nbase, Lseq)
    grad = np.transpose(grad, axes = (1,0,2,3))
    
    # return array if shape = (Ndatapoints, Ntracks, Nbase, Lseq)
    if model.shift_sequence is not None:
        grad = grad[..., model.shift_sequence: np.shape(grad)[-1]-model.shift_sequence]
        if isinstance(baseline, torch.Tensor):
            baseline = baseline[..., model.shift_sequence: np.shape(baseline)[-1]-model.shift_sequence]
    # correct the deeplift output
    if not hypothetical and effective_attributions:
        grad = correct_deeplift(grad, baseline.detach().numpy())
    #attmax = np.amax(grad, axis = (-1,-2))
    #for a, am in enumerate(attmax):
        #print(pearsonr(am, x_pred[a]))
    print(np.shape(grad))
    return grad  


# Does not work well for all models: Needs atleast GELU manually added to nonlinar functions. 
# But could be used for other methods.
def captum_sequence_attributions(x, model, tracks, deepshap = False, basefreq = None, batchsize = None, effective_attributions = True):
    from captum.attr import DeepLift, DeepLiftShap
    # TODO use for other methods
    #GradientShap,
    #DeepLift,
    #DeepLiftShap,
    #IntegratedGradients
    
    if deepshap:
        dl = DeepLiftShap(model.eval(), multiply_by_inputs=False)
    else:
        dl = DeepLift(model.eval(), multiply_by_inputs=False, eps=1e-6)
    
    x = torch.Tensor(x)
    x_pred = model.predict(x)[:,tracks]
    #print(np.shape(x_pred))

    # batch size will be taken from model
    if batchsize is None:
        batchsize = model.batchsize
    
    fnum = x.size(-2)
    if basefreq is None:
        basefreq = 1./x.size(-2)
    if isinstance(basefreq, float):
        basefreq = torch.ones(fnum) * basefreq
    
    # basefreq needs to be an array of length fnum by now, can also be provided as an array
    baseline = torch.ones_like(x)*basefreq[:,None]
    # Version were each sequence is tested against a random set of permutations is not implemented yet and requires way more resources
    baseline_pred = model.predict(baseline)[:,tracks]
    #print(np.shape(baseline_pred))
    # when shift_sequence in my model is not None, inputs will be padded
    if model.shift_sequence is not None:
        x = torch.nn.functional.pad(x, (model.shift_sequence, model.shift_sequence), mode = 'constant', value = 0.25)
        baseline = torch.nn.functional.pad(baseline, (model.shift_sequence, model.shift_sequence), mode = 'constant', value = 0.25)
        
    grad = []
    deltas = []
    # perform deeplift independently for each output track
    
    for t, tr in enumerate(tracks):
        gr = []
        delt = []
        # use batchsize to avoid memory issues
        for b in range(0,x.size(dim = 0), batchsize):
            xb = x[b:b+batchsize].to(model.device)
            bb = baseline[b:b+batchsize].to(model.device)
            attributions, delta = dl.attribute(xb, bb, target=int(tr), return_convergence_delta=True)
            delt.append(delta.cpu().detach().numpy())
            gr.append(attributions.cpu().detach().numpy())
        grad.append(np.concatenate(gr, axis = 0))
        delt = np.concatenate(delt)
    # return array if shape = (Ndatapoints, Ntracks, Nbase, Lseq)
    grad = np.transpose(np.array(grad), axes = (1,0,2,3))
    if model.shift_sequence is not None:
        grad = grad[..., model.shift_sequence: np.shape(grad)[-1]-model.shift_sequence]
    # correct the deeplift output
    if effective_attributions:
        grad = correct_deeplift(grad, basefreq.detach().numpy())
    #attmax = np.amax(grad, axis = (-1,-2))
    #for a, am in enumerate(attmax):
        #print(pearsonr(am, x_pred[a]))
    print(np.shape(grad))
    return grad            


def kernel_assessment(model, X, Y, testclasses = None, onlyppms = True, genewise = False, stppm= None, Nppm = None, respwm = 200000, activate_kernel = False, kactivation_cut = 1.64, kactivation_selection = True): 
        ppms = [] # position frequency matrices from kernel weights, with biases included 
        pwms = [] # position frequency matrices from alignment of highest scoring sequences, zscore > 2.326
        weights = []
        
        biases = [] # biases of kernels
        motifmeans = [] # mean activation of kernel
        motifnames = [] # generate list of names
        seqactivations = [] # activations of random sequences in seq_seq for pwm creation from alignment
        
        importance = [] # change in correlation across data points for each experiment shape = (kernelxexperiments)
        mseimportance = [] # change in mse across data points for each experiment shape = (kernelxexperiments)
        effect = [] # weighted change in prediction, prediction is weighted by mse of each gene, if mse is small then no not much of an effect, shape = (kernelxexperiments)
        abseffect = [] # absoltue different of predictions shape = (kernelxexperiments)
        genetraincorr = []
        # generate genewise kernel impact matrix for all test classes
        geneimportance = None
        if genewise:
            geneimportance = [[] for t in range(len(np.unique(testclasses)))]
        
        # use random sequences to compute mean activations AND pwms from sequence alignments
        seq_seq = generate_random_onehot(model.l_kernels, respwm) # generate 200000 random sequences with length l_kernels
        i =0
        islist = isinstance(X, list) # check if several or only one input
        # compute unmasked performances
        if not onlyppms:
            # predict original training predictions
            Ypredtrain = model.predict(X)
            # compute the correlation between data points for each output track
            traincorr = correlation(Ypredtrain,Y, axis = 0)
            trainmse = mse(Ypredtrain, Y, axis = 0)
            # compute correlation for each training gene
            for t, testclass in enumerate(np.unique(testclasses)):
                consider = np.where(testclasses == testclass)[0]
                if len(consider) > 1:
                    genetraincorr.append(correlation(Ypredtrain[:,consider],Y[:,consider], axis = 1))
        # Find the kernels of all networks and iterate over them
        for namep, parms in model.named_parameters():
            if (namep.split('.')[0] == 'cnns' and namep.split('.')[2] == 'convolutions' and namep.rsplit('.')[-1] == 'weight') or (namep.split('.')[0] == 'convolutions' and namep.rsplit('.')[-1] == 'weight'):
                print(i, namep, parms.size())
                nameadd = ''
                if namep.split('.')[0] == 'cnns':
                    nameadd = '_'+namep.split('.')[1]
                    
                # collect the first layer convolution kernels
                kernelweight = parms.detach().cpu().numpy()
                if stppm is not None: 
                    kernelweight = kernelweight[stppm : stppm+Nppm]
                else:
                    stppm, Nppm = 0, len(kernelweight)
                # collect kernels to transform weights directly to ppms.
                weights.append(kernelweight)
                ppms.append(kernelweight)
                # collect the biases if bias is not None
                if model.kernel_bias:
                    bias = model.state_dict()[namep.replace('weight', 'bias')].detach().cpu().numpy()[stppm : Nppm + stppm]
                else:
                    bias = np.zeros(len(ppms[-1]))
                biases.append(bias)
                # Generate names for all kernels
                motifnames.append(np.array(['filter'+str(j+stppm)+nameadd for j in range(len(ppms[-1]))]))
                # compute motif means from the activation of kernels with the random sequences.
                seqactivations = np.sum(ppms[-1][:,None]*seq_seq[None,...],axis = (2,3))
                if not onlyppms:
                    motifmeans.append(np.mean(seqactivations + bias[:,None] , axis = 1))
                if activate_kernel:
                    # use activation function of model to adjust sequence activations
                    kernel_activation = func_dict[model.kernel_function]
                    print(model.kernel_function)
                    seqactivations = kernel_activation(torch.Tensor(seqactivations)).numpy()
                # generate motifs from aligned sequences with activation over 0.9 of the maximum activation
                pwms.append(pwms_from_seqs(seq_seq, seqactivations, kactivation_cut, z_score = kactivation_selection))
                # take the mean of these activations from all 100000 sequences as a mean actiavation value for each filter.
                if not onlyppms:
                    motifmeans.append(np.mean(seqactivations + bias[:,None] , axis = 1))
                    t0 = time.time()
                    for m in range(len(ppms[-1])):
                        # make predictions from models with meaned activations from kernels
                        if m%10==0:
                            print(m+stppm, '/' , stppm+Nppm, round(time.time()-t0,2))
                            t0 = time.time()
                        if islist:
                            mask = [i,m+stppm]
                        else:
                            mask = m+stppm
                        Ymask = model.predict(X, mask = mask, mask_value = motifmeans[-1][m])
                        # importance as difference in correlation between full model to train data and masked model to train data
                        importance.append(np.around(correlation(Ymask,Y, axis = 0) - traincorr,3))
                        mseimportance.append(np.around(mse(Ymask,Y, axis = 0) - trainmse,6))
                        # compute the importance of the kernel for every gene
                        if genewise:
                            for t, testclass in enumerate(np.unique(testclasses)):
                                consider = np.where(testclasses == testclass)[0]
                                if len(consider) > 1:
                                    geneimportance[t].append(np.around(correlation(Ymask[:,consider],Y[:,consider], axis = 1) - genetraincorr[t],3))
                        # effect for a track shows the direction that the kernel causes on average over all genes
                        # it is weighted by how much it changes the mse of each gene
                        effect.append(np.around(np.sum((Ypredtrain-Ymask)**3/np.sum((Ypredtrain-Ymask)**2,axis = 1)[:,None],axis = 0),6))
                        abseffect.append(np.around(np.sum(Ypredtrain-Ymask,axis = 0),6))
                i += 1
        
        motifnames = np.concatenate(motifnames)
        weights = np.concatenate(weights, axis = 0)
        ppms = np.concatenate(ppms, axis = 0)
        biases = np.concatenate(biases, axis = 0)
        # create ppms directly from kernel matrix
        ppms = np.around(kernel_to_ppm(ppms[:,:,:], kernel_bias =biases),3)
        weights = np.around(weights, 6)
        # generate pwms from most activated sequences
        pwms = np.around(np.concatenate(pwms, axis = 0),3)
        min_nonrand_freq = 0.3
        iupacmotifs = pfm2iupac(pwms, bk_freq = min_nonrand_freq)
    
        return ppms, pwms, weights, biases, iupacmotifs, motifnames, importance, mseimportance, effect, abseffect, geneimportance, genetraincorr


def indiv_network_contribution(model, X, Y, testclasses, outname, names):
    # pick a random set of input sequences to compute mean output of individual networks
    randomseqs = np.random.permutation(len(X[0]))[:7000]
    # predict original predictions to compare losses to 
    Ypredtrain = model.predict(X)
    Ymask = []
    netbsize = 10 # batchsize for computing the mean
    for i in range(len(X)):
        # get mean representations for individual networks
        mean_rep = []
        for r in range(0,len(randomseqs), netbsize):
            rand = randomseqs[r:r+netbsize]
            mrep = model.forward([torch.Tensor(x[rand]).to(model.device) for x in X], location = '0', cnn = i)
            if isinstance(mrep,list):
                print(mrep[0].size(), len(mrep))
                print('--test_individual_network : Does not work with direct, difference or any analytic connection')
                sys.exit()
            mrep = mrep.detach().cpu().numpy()
            mean_rep.append(mrep)
        mean_rep = np.concatenate(mean_rep, axis = 0)
        #print(mean_rep[0], mean_rep[1], mean_rep[2])
        #print(np.std(mean_rep,axis = 0))
        mean_rep = np.mean(mean_rep,axis = 0)
        # predict values with masked networks
        Ymask.append(model.predict(X, mask = i, mask_value = torch.Tensor(mean_rep).to(model.device)))
        #print(i, mean_rep)
    
    for tclass in np.unique(testclasses):
        # go through all the different classes, for example cell types
        consider = np.where(testclasses == tclass)[0]
        genecont = []
        header= ['MSEtoReal', 'CorrtoReal']
        # compute the performance for this class for the full predictions
        trainmse = mse(Ypredtrain[:,consider],Y[:,consider] ,axis =1)
        traincorr = correlation(Ypredtrain[:,consider],Y[:,consider] ,axis =1)
        #print(tclass)
        #print(traincorr)
        for i in range(len(X)):
            header.append('CNN'+str(i)+'DeltaMSE')
            header.append('CNN'+str(i)+'DeltaCorr')
            #header.append('CNN'+str(i)+'DeltaAvgYpred')
            # compare the performance of the masked predictions with the full predicdtions
            MSEdif = mse(Ymask[i][:,consider],Y[:,consider],axis = 1) - trainmse
            genecont.append(MSEdif)
            Corrdif = correlation(Ymask[i][:,consider],Y[:,consider],axis = 1) - traincorr
            genecont.append(Corrdif)
            #DYdiff = np.mean(Ymask[i][:,consider] - Ypredtrain[:,consider], axis - 1)
            
        np.savetxt(outname+'_netatt'+tclass+'.dat', np.concatenate([[names],np.around([trainmse,traincorr],4), np.around(np.array(genecont),6)],axis = 0).T, header = 'Gene '+' '.join(np.array(header)), fmt = '%s')




