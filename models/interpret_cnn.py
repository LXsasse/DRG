from functions import dist_measures, correlation, mse
import numpy as np
import sys, os
from scipy.spatial.distance import cdist
import torch
import time 
# Need per-sequence method that accounts dependencies of between positions
    # Maybe first scan for one mutation, then sample from ones to create twos, then sample from twos to create threes and so on. 
    # Scrambler does that basically --> can be used

# Need method that summarizes the importance scores along all sequences
# Basically cluster/align sequences based on their importance score profiles
    # Extract single important motifs from sequences, cluster them.
    # Measure impact of motifs by mutating all single and double bases in it
    # Then take motifs that occur in combinations and determine expected impact from multiplication
    # Then scramble two bases, one from each motif and determine combined impact of motifs
    # Compare combined impact to single impact for each base pair to determine motif interaction
    # Them move the motifs around and measure the distance importance.

#Integrated gradients faster than in silico mutuations
# Install also integrated hessians and derive motif interactions from this
#Could be used before ISM to only do ISM on sequences with motifs?



def compute_importance(model, in_test, out_test, activation_measure = 'euclidean', direction = True, pwm_in = None, normalize = True):
    n_kernels = model.num_kernels
    complete_predict = model.predict(in_test, pwm_out = pwm_in)
    #activation_measures: euclidean, correlation
    ## replace cdist with funciton that does not compute the entire matrix
    #full_predict = np.diagonal(cdist(full_predict.T, out_test.T, activation_measure))
    full_predict = dist_measures(complete_predict.T, out_test.T, activation_measure, axis = 1)
    importance = []
    impacts = []
    for n in range(n_kernels):
        mnpredict = model.predict(in_test, mask = n, pwm_out = pwm_in)
        reduce_predict = np.diagonal(cdist(mnpredict.T, out_test.T, activation_measure))
        reduce_predict = dist_measures(mnpredict.T, out_test.T, activation_measure, axis = 1)
        importance.append(reduce_predict - full_predict)
        impact = mnpredict-complete_predict
        impacts.append(np.sum(impact**3, axis = 0)/np.sum(impact**2, axis = 0))
    if pwm_in is not None:
        for n in range(n_kernels, n_kernels + np.shape(pwm_in)[-2]):
            mnpredict = model.predict(in_test, mask = n, pwm_out = pwm_in)
            reduce_predict = dist_measures(mnpredict.T, out_test.T, activation_measure, axis = 1)
            importance.append(reduce_predict - full_predict)
            impact = mnpredict-complete_predict
            impacts.append(np.sum(impact**3, axis = 0)/np.sum(impact**2, axis = 0))
    importance = np.array(importance)
    impacts = np.array(impacts)
    if normalize:
        importance = np.around(importance/np.amax(importance),4)
    return importance, impacts

def kernel_to_ppm(kernels, kernel_bias = None, bk_freq = None):
    n_kernel, n_input, l_kernel = np.shape(kernels)
    if kernel_bias is not None:
        kernels += kernel_bias[:,None,None]/(n_input*l_kernel)
    #kernels *= l_kernel
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
        #bins = np.linspace(0,1,21)
        #plt.hist(act, bins = bins)
        #plt.hist(act[seqs[1][seqs[0]==a]],bins = bins, color='orange',zorder = 1)
        #plt.show()
    return np.array(pwms)
 
def genseq(lseq, nseq):
    seqs = np.zeros((nseq,4,lseq))
    pos = np.random.randint(0,4,lseq*nseq)
    pos0 = (np.arange(lseq*nseq,dtype=int)/lseq).astype(int)
    pos1 = np.arange(lseq*nseq,dtype=int)%lseq
    seqs[pos0,pos,pos1] = 1
    return seqs

# Could use this to propagate importance through network and determine individual nodes and then interpret them.
# perform zscore test for each parameter as in linear regression
def parameter_importance(ypred, y, coef_, inputs):
    invcovardiag = np.diagonal(np.pinv(np.dot(inputs.T, inputs),rcond = 1e-8, hermetian = True))
    loss = np.mean((ypred-y)**2, axis = 0)
    var_b = loss[None,:]*(invcovardiag)[:,None]
    sd_b = np.sqrt(var_b)
    z_scores = coef_ * sd_b
    z_scores = np.nan_to_num(z_scores)
    return z_scores


def pfm2iupac(pwms, bk_freq = None):
    hash = {'A':16, 'C':8, 'G':4, 'T':2}
    dictionary = {'A':16, 'C':8, 'G':4, 'T':2, 'R':20, 'Y':10, 'S':12, 'W':18, 'K':6, 'M':24, 'B':14, 'D':22, 'H':26, 'V':28, 'N':0}
    res = dict((v,k) for k,v in dictionary.items())
    n_nts = len(pwms[0])
    if bk_freq is None:
        bk_freq = (1./float(n_nts))*np.ones(n_nts)
    else:
        bk_freq = bk_freq*np.ones(n_nts)
    motifs = []
    for pwm in pwms:
        m = ''
        for p in pwm.T:
            score = 0
            for i in range(len(p)):
                if p[i] > bk_freq[i]:
                    score += list(hash.values())[i]
            m += res[score]
        motifs.append(m)
    return np.array(motifs)


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
'''
# slower because grad is computed for all inputs
def takegrad(x, model, tracks):
    grad = []
    x = torch.Tensor(x)
    x.requires_grad = True
    pred = model.predict(x, enable_grad=True)
    for i in range(x.size(dim = 0)):
        gra = []
        for t, tr in enumerate(tracks):
            pred[i, tr].backward(retain_graph = True)
            gr = x.grad.clone().cpu().numpy()[i]
            gra.append(gr)
            x.grad.zero_()
        grad.append(np.concatenate(gra,axis = 0))
    grad = np.array(grad)
    return grad
'''                                                        

def takegrad(x, model, tracks, correct_from_neutral = True, top=None):
    grad = []
    if isinstance(x, list):
        x = [torch.Tensor(xi) for xi in x]
    else:
        x = torch.Tensor(x)
    if isinstance(x,list):
        Nin = len(x)
        grad = [[] for n in range(Nin)]
        for i in range(x[0].size(dim = 0)):
            xi = []
            for n in range(Nin):
                xij = torch.clone(x[n][[i]])
                xij.requires_grad = True
                xi.append(xij)
            gra = [[] for n in range(Nin)]
            pred = model.predict(xi, enable_grad = True)
            if isinstance(pred, list):
                pred = torch.cat(pred, axis = 1)
            for t, tr in enumerate(tracks):
                pred[0, tr].backward(retain_graph = True)
                for n, xij in enumerate(xi):
                    gr = xij.grad.clone().cpu().numpy()
                    gra[n].append(gr)
                xij.grad.zero_()
            for n in range(Nin):
                gra[n] = np.concatenate(gra[n],axis = 0)
                if correct_from_neutral:
                    gra[n] = correct_by_mean(gra[n])
                if top is not None:
                    ngrashape = list(np.shape(gra[n]))
                    ngrashape[-2] += 1
                    ngrashape[-1] = top
                    ngra = np.zeros(ngrashape)
                    for t in range(np.shape(gra[n])[0]):
                        lcn = np.argsort(-np.amax(np.absolute(gra[n][t]), axis = -2), axis = -1)
                        lcn = np.sort(lcn[:top])
                        ngra[t] = np.append(gra[n][t][...,lcn], lcn[None, :], axis = -2)
                    gra[n] = ngra
                    
                grad[n].append(gra[n])
                
                    
        for n in range(Nin):
            grad[n] = np.array(grad[n])
            
            
    else:
        start = time.time()
        for i in range(x.size(dim = 0)):
            xi = torch.clone(x[[i]])
            xi.requires_grad = True
            gra = []
            pred = model.predict(xi, enable_grad = True)
            if isinstance(pred, list):
                pred = torch.cat(pred, axis = 1)
            for t, tr in enumerate(tracks):
                pred[0, tr].backward(retain_graph = True)
                gr = xi.grad.clone().cpu().numpy()
                gra.append(gr)
                xi.grad.zero_()
            grad.append(np.concatenate(gra,axis = 0))
        grad = np.array(grad)
        end = time.time()
        if correct_from_neutral:
            grad = correct_by_mean(grad)
        print('TISM time for', x.size(dim = 0), len(tracks),  end-start)
    return grad



# "scales" deeplift output to the effective change from baseline to observed sequence
def correct_deeplift(dlvec, freq):
    if len(np.shape(freq)) < len(np.shape(dlvec)):
        freq = np.ones_like(dlvec) * freq[:,None] # for each base, we have to "take step" from the frequency in baseline to 1.
    # This whole procedure is similar to ISM, where we remove the reference base and set it to zero, so take a step of -1, and then go a step of one in the direction of the alternative base. DeepLift returns the regression coefficients, which need to be scaled to the steps that are taken to arrive at the base we're looking at.
    negmul = np.sum(dlvec * freq, axis = -2)
    dlvec = dlvec - negmul[...,None,:]
    return dlvec

# use a different way to correct_deeplift, and allow for multiple dlvec and mulitple freq
# use deeplift in a different way, give it list of output tracks and multiple sequences at once. 
    
                
def deeplift(x, model, tracks, deepshap = False, basefreq = None, batchsize = None, effective_attributions = True):
    from captum.attr import DeepLift, DeepLiftShap
    #GradientShap,
    #DeepLift,
    #DeepLiftShap,
    #IntegratedGradients)
    if deepshap:
        dl = DeepLiftShap(model.eval(), multiply_by_inputs=False)
    else:
        dl = DeepLift(model.eval(), multiply_by_inputs=False, eps=1e-6)
    
    x = torch.Tensor(x)
    # when shift_sequence in my model is not None, inputs will be padded
    if model.shift_sequence is not None:
        x = torch.nn.functional.pad(x, (model.shift_sequence, model.shift_sequence), mode = 'constant', value = 0.25)
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
        print('Deltas for track', tr, 'min', np.amin(delt), '5%', np.percentile(delt, 5), '10%',np.percentile(delt, 10), 'median',np.median(delt), '90%',np.percentile(delt, 90),'95%',np.percentile(delt, 95),'max', np.amax(delt))
    # return array if shape = (Ndatapoints, Ntracks, Nbase, Lseq)
    grad = np.transpose(np.array(grad), axes = (1,0,2,3))
    if model.shift_sequence is not None:
        grad = grad[..., model.shift_sequence: np.shape(grad)[-1]-model.shift_sequence]
    # correct the deeplift output
    if effective_attributions:
        grad = correct_deeplift(grad, basefreq.detach().numpy())
    print(np.shape(grad))
    return grad            


def kernel_assessment(model, X, Y, testclasses = None, onlyppms = True, genewise = False, ppmparal = False, stppm= None, Nppm = None, respwm = 200000): 
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
        seq_seq = genseq(model.l_kernels, respwm) # generate 200000 random sequences with length l_kernels
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
                if ppmparal: 
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
                # generate motifs from aligned sequences with activation over 0.9 of the maximum activation
                pwms.append(pwms_from_seqs(seq_seq, seqactivations, 1.64))
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
        ppms = np.concatenate(ppms, axis = 0)
        biases = np.concatenate(biases, axis = 0)
        # create ppms directly from kernel matrix
        ppms = np.around(kernel_to_ppm(ppms[:,:,:], kernel_bias =biases),3)
        weights = np.around(kernelweight, 6)
        # generate pwms from most activated sequences
        pwms = np.around(np.concatenate(pwms, axis = 0),3)
        min_nonrand_freq = 0.3
        iupacmotifs = pfm2iupac(pwms, bk_freq = min_nonrand_freq)
    
        return ppms, pwms, weights, iupacmotifs, motifnames, importance, mseimportance, effect, abseffect, geneimportance, genetraincorr


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
    
def write_meme_file(pwm, pwmname, alphabet, output_file_path):
    """[summary]
    write the pwm to a meme file
    Args:
        pwm ([np.array]): n_filters * 4 * motif_length
        output_file_path ([type]): [description]
    """
    n_filters = len(pwm)
    print(n_filters)
    meme_file = open(output_file_path, "w")
    meme_file.write("MEME version 4 \n")
    meme_file.write("ALPHABET= "+alphabet+" \n")
    meme_file.write("strands: + -\n")

    print("Saved PWM File as : {}".format(output_file_path))

    for i in range(0, n_filters):
        if np.sum(np.absolute(pwm[i])) > 0:
            meme_file.write("\n")
            meme_file.write("MOTIF %s \n" % pwmname[i])
            meme_file.write(
                "letter-probability matrix: alength= "+str(len(alphabet))+" w= %d \n"
                % np.count_nonzero(np.sum(pwm[i], axis=0))
            )
        
        for j in range(0, np.shape(pwm[i])[-1]):
            #if np.sum(pwm[i][:, j]) > 0:
                for a in range(len(alphabet)):
                    if a < len(alphabet)-1:
                        meme_file.write(str(pwm[i][ a, j])+ "\t")
                    else:
                        meme_file.write(str(pwm[i][ a, j])+ "\n")

    meme_file.close()




