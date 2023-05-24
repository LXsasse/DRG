from functions import dist_measures
import numpy as np
import sys, os
from scipy.spatial.distance import cdist


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
        kernels += kernel_bias[:,None,None]
    if bk_freq is None:
        bk_freq = np.ones(n_input)*np.log2(1./float(n_input))
    elif isinstance(bk_freq, float) or isinstance(bk_freq, int):
        bk_freq = np.ones(n_input)*np.log2(1./float(bk_freq))
    kernels -= bk_freq[None,:,None]
    ppms = 2.**kernels
    ppms = ppms/np.sum(ppms, axis = 1)[:,None, :]
    return ppms

def pwms_from_seqs(ohseqs, activations, cut):
    minact = np.amin(activations, axis = 1)
    activations = activations - minact[:,None]
    maxact = np.amax(activations, axis = 1)
    activations = activations/maxact[:,None]
    seqs = np.where(activations >= cut)
    pwms = []
    for a, act in enumerate(activations):
        pwms.append(np.sum(ohseqs[seqs[1][seqs[0]==a]], axis = 0)/np.sum(ohseqs[seqs[1][seqs[0]==a]], axis = (0,1))[None, :])
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
        if np.sum(pwm[i]) > 0:
            meme_file.write("\n")
            meme_file.write("MOTIF %s \n" % pwmname[i])
            meme_file.write(
                "letter-probability matrix: alength= "+str(len(alphabet))+" w= %d \n"
                % np.count_nonzero(np.sum(pwm[i], axis=0))
            )
        
        for j in range(0, np.shape(pwm[i])[-1]):
            if np.sum(pwm[i][:, j]) > 0:
                for a in range(len(alphabet)):
                    if a < len(alphabet)-1:
                        meme_file.write(str(pwm[i][ a, j])+ "\t")
                    else:
                        meme_file.write(str(pwm[i][ a, j])+ "\n")

    meme_file.close()




