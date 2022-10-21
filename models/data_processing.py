import sys, os 
import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr, cosine
from scipy.spatial.distance import cdist
from collections import OrderedDict
from functools import reduce





def readin(inputfile, outputfile, delimiter = ' ', return_header = True, assign_region = True, n_features = 4, combinex = True, mirrorx = False):

    if ',' in inputfile:
        inputfiles = inputfile.split(',')
        X = []
        inputfeatures = []
        inputnames = []
        # determines if kmerfile or sequence one-hot encoding
        arekmers = True
        for i, inputfile in enumerate(inputfiles):
            if os.path.splitext(inputfile)[1] == '.npz':
                Xin = np.load(inputfile, allow_pickle = True)
                inpnames = Xin['genenames'].astype(str)
                sortn = np.argsort(inpnames)
                inputnames.append(inpnames[sortn])
                Xi = Xin['seqfeatures'][0][sortn]
                if len(np.shape(Xi)) > 2:
                    arekmers = False
                    inputfeatures.append([x+'_'+str(i) for x in Xin['seqfeatures'][1]])
                else:
                    inputfeatures.append(Xin['seqfeatures'][1])
                
                
                if mirrorx and not arekmers:
                    Xi = realign(Xi)
                X.append(Xi)

            else: # For fastafiles create onehot encoding 
                inpnames, inseqs = readinfasta(inputfile)
                Xin, Xinfeatures = quick_onehot(inseqs)
                inputfeatures.append([x+'_'+str(i) for x in Xinfeatures])
                if mirrorx:
                    Xin = realign(Xin)
                X.append(Xin)
                inputnames.append(inpnames)
                arekmers = False
        # transpose, and then introduce mirrorx

        comnames = reduce(np.intersect1d, inputnames)
        X = [X[i][np.isin(inputnames[i],comnames)] for i in range(len(X))]
        if arekmers:
            inputfeatures = np.concatenate(inputfeatures)
        else:
            inputfeatures = inputfeatures[0]
            if assign_region:
                lx = len(X)
                X = [np.append(X[i], np.ones((np.shape(X[i])[0], np.shape(X[i])[1], lx), dtype = np.int8)*(np.arange(lx)==i).astype(np.int8), axis = -1) for i in range(lx)]
                inputfeatures = np.append(inputfeatures, ['F'+str(i) for i in range(lx)])
        if combinex:
            X = np.concatenate(X, axis = 1)
        inputnames = comnames
    else:
        combinex = True
        if os.path.splitext(inputfile)[1] == '.npz':
            Xin = np.load(inputfile, allow_pickle = True)
            X, inputfeatures = Xin['seqfeatures']
            arekmers = len(np.shape(X)) <= 2
            inputnames = Xin['genenames']
            if assign_region == False and not arekmers:
                inputfeatures = inputfeatures[:n_features]
                X = X[:,:,:n_features]
            if mirrorx:
                X = realign(X)
        else: # For fastafiles create onehot encoding 
            arekmers = False
            inputnames, inseqs = readinfasta(inputfile)
            X, inputfeatures = quick_onehot(inseqs)
            if mirrorx:
                X = realign(X)
    if os.path.isfile(outputfile):
        if os.path.splitext(outputfile)[1] == '.npz':
            Yin = np.load(outputfile, allow_pickle = True)
            Y, outputnames = Yin['counts'], Yin['names'] # Y should of shape (nexamples, nclasses, l_seq/n_resolution)
        else:
            Yin = np.genfromtxt(outputfile, dtype = str, delimiter = delimiter)
            Y, outputnames = Yin[:, 1:].astype(float), Yin[:,0]
        hasoutput = True
    else:
        print(outputfile, 'not a file')
        hasoutput = False
        Y, outputnames = None, None
    #eliminate data points with no features
    if arekmers and combinex:
        Xmask = np.sum(X*X, axis = 1) > 0
        X, inputnames = X[Xmask], inputnames[Xmask]
    
    
    sortx = np.argsort(inputnames)
    if hasoutput:
        sortx = sortx[np.isin(np.sort(inputnames), outputnames)]
        sorty = np.argsort(outputnames)[np.isin(np.sort(outputnames), inputnames)]
        
    if combinex:
        X, inputnames = X[sortx], inputnames[sortx]
    else:
        X, inputnames = [x[sortx] for x in X], inputnames[sortx]
    if hasoutput:
        Y, outputnames = Y[sorty], outputnames[sorty]
    
    if return_header and hasoutput:
        if os.path.splitext(outputfile)[1] == '.npz':
            if 'celltypes' in Yin.files:
                header = Yin['celltypes']
            else:
                header = ['C'+str(i) for i in range(np.shape(Y)[1])]
        else:
            header = open(outputfile, 'r').readline()
            if '#' in header:
                header = header.strip('#').strip().split(delimiter)
            else:
                header = ['C'+str(i) for i in range(np.shape(Y)[1])]
        header = np.array(header)
    else:
        header  = None
    
    if not arekmers:
        if combinex:
            X = np.transpose(X, axes = [0,2,1])
        else:
            X = [np.transpose(x, axes = [0,2,1]) for x in X]
    if combinex:
        print('Input shapes', np.shape(X), np.shape(Y))
    else:
        print('Input shapes', [np.shape(x) for x in X], np.shape(Y))
    
    
    return X, Y, inputnames, inputfeatures, header

def realign(X):
    end_of_seq = np.sum(X,axis = (1,2)).astype(int)
    Xmirror = []
    lx = np.shape(X)[-2]
    for s, es in enumerate(end_of_seq):
        xmir = np.zeros(np.shape(X[s]), dtype = np.int8)
        xmir[lx-es:] = X[s,:es]
        Xmirror.append(xmir)
    return np.append(X, np.array(Xmirror), axis = -2)

def readinfasta(fastafile, minlen = 10, upper = True):
    obj = open(fastafile, 'r').readlines()
    genes = []
    sequences = []
    for l, line in enumerate(obj):
        if line[0] == '>':
            sequence = obj[l+1].strip()
            if sequence != 'Sequence unavailable' and len(sequence) > minlen:
                genes.append(line[1:].strip())
                if upper:
                    sequence = sequence.upper()
                sequences.append(sequence)
    sortgen = np.argsort(genes)
    genes, sequences = np.array(genes)[sortgen], np.array(sequences)[sortgen]
    return genes, sequences

def seqlen(arrayofseqs):
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

def read_mutationfile(mutfile,X,Y,names,experiments):
    ## Read in the mutation file and use mean of expression from all sequences with same sequence instead of having multiple duplications loaded into memory. Use weighting instead
    ### Add sequences with mutations as additional examples to the 1-d X
    mutation = np.genfromtxt(mutfile, dtype = str)
    mutation = mutation.T
    mutations = []
    for mutate in mutation:
        mut = []
        for m in mutate:
            mut.append(m.split(':'))
        mutations.append(mut)
    mutations = np.array(mutations)
    mutnames = np.array(open(mutfile).readline().strip('#').strip().split())
    nindividuals = len(mutnames)
    if np.array_equal(mutnames, experiments):
        modifs = np.array([np.concatenate(mutations[:, :, 0].astype(int)), (np.ones((np.shape(mutations)[1], np.shape(mutations)[0]), dtype = int)*np.arange(len(mutations),dtype = int)).T.flatten()]).T
        modseqs = np.unique(mutations[:,:,0].astype(int))
        modpos = [np.concatenate(mutations[:,:,1]).astype(int), np.concatenate(mutations[:,:,2])]
        kept_seq = np.delete(np.arange(len(X), dtype = int), modseqs)
        Yn = Y[:,0]
        # For all sequences that do not have a single mutation in any individual, take the mean over all individuals as the expression for that sequence.
        # Note: does not have to be Gaussian mean, can be other distribution
        Yn[kept_seq] = np.mean(Y[kept_seq],axis = 1) # Assumption of Gassian Noise
        weights = np.ones(len(Yn)+len(modifs))
        keep = list(kept_seq)
        nts = np.array(['A','C','G','T'])
        for m , mod in enumerate(modifs):
            #if m%100 == 0:
                #print(m, len(modifs))
            Yn = np.append(Yn, Y[mod[0], mod[1]])
            names = np.append(names, [names[mod[0]]+'_'+experiments[mod[1]]])
            weights[len(Yn)-1] = 1./float(nindividuals)
            Xn = np.copy(X[mod[0]])
            Xn[:, int(modpos[0][m])] = nts == modpos[1][m]
            X = np.append(X, [Xn], axis = 0)
            origexp = np.delete(np.arange(nindividuals, dtype = int), modifs[modifs[:,0] == mod[0],1])
            keep.append(len(Yn)-1)
            if len(origexp) > 0:
                Yn[mod[0]] = np.mean(Y[mod[0]][origexp])
                weights[mod[0]] = float(len(origexp))/float(nindividuals) + 1e-8
                keep.append(mod[0])
    
    keep = np.unique(keep)
    Y = Yn[keep].reshape(-1,1)
    X = X[keep]
    names = names[keep]
    weights = weights[keep]
    experiments = ['SNPenriched']
    return X,Y,names,experiments,weights



## Create random train, test and validation sets using permutations
# Samples can be split into different classes: each set will have same fraction of each class
def create_sets(n_samples, folds, fold, seed = 1010, Yclass = None, genenames = None):
    if isinstance(folds, int):
        if Yclass is None:
            Yclass = np.ones(n_samples)
        np.random.seed(seed)
        permut = np.random.permutation(n_samples)
        classes = np.unique(Yclass)
        testset, valset, trainset = [],[],[]
        valfold = fold -1
        if valfold == -1:
            valfold = folds -1
        for cla in classes:
            inclass = Yclass[permut] == cla
            totinclass = np.sum(inclass)
            modulos = totinclass%folds
            addedstart = fold * int(fold<modulos)
            startfold = int((fold/folds)*np.sum(Yclass== cla)) +addedstart
            endfold = int(((fold+1)/folds)*np.sum(Yclass== cla))+ addedstart + int(fold+1 < modulos)
            valaddedstart = valfold * int(valfold<modulos)
            valstartfold = int((valfold/folds)*np.sum(Yclass== cla)) +valaddedstart
            valendfold = int(((valfold+1)/folds)*np.sum(Yclass== cla))+ valaddedstart + int(valfold+1 < modulos)
            testset.append(permut[inclass][startfold:endfold])
            valset.append(permut[inclass][valstartfold:valendfold])
        testset, valset = np.concatenate(testset), np.concatenate(valset)
        trainset = np.delete(np.arange(n_samples, dtype = int), np.append(testset, valset))
    else:
        lines = open(folds, 'r').readlines()
        sets = []
        for l, line in enumerate(lines):
            if line[0] != '#':
                line = line.strip().split()
                sets.append(np.where(np.isin(genenames,line))[0])
        testset = sets[(fold+1)%len(sets)]
        valset = sets[fold]
        trainset = np.delete(np.arange(len(genenames), dtype = int),np.append(testset,valset))
    return trainset, testset, valset



# Combine filenames to a new output file name, removing text that is redundant in both filenames    
def create_outname(name1, name2, lword = 'on'):
    name1s = os.path.split(name1)[1].replace('.dat','').replace('.hmot','').replace('.txt','').replace('.npz','').replace('.list','').replace('.csv','').replace('.tsv','').replace('.tab','').replace('-', "_").replace('.fasta','').replace('.fa','').split('_')
    name2s = name2.replace('-', "_").split('_')
    diffmask = np.ones(len(name1s)) == 1
    for n, na1 in enumerate(name1s):
        for m, na2 in enumerate(name2s):
            if na1 in na2:
                diffmask[n] = False
    diff = np.array(name1s)[diffmask]
    outname = os.path.split(name2)[1].replace('.dat','').replace('.hmot','').replace('.txt','').replace('.npz','').replace('.list','').replace('.csv','').replace('.tsv','').replace('.tab','').replace('.fasta','').replace('.fa','')
    if len(diff) > 0:
        outname+=lword+'_'.join(diff)
    return outname


# check if number can be converted to a float
def isfloat(number):
    try:
        float(number)
    except:
        return False
    else:
        return True
    
# check if string can be integer or float
def numbertype(inbool):
    try:
        int(inbool)
    except:
        pass
    else:
        return int(inbool)
    try:
        float(inbool)
    except:
        pass
    else:
        return float(inbool)
    return inbool

# check if str is boolean, a list, or a number, otherwise return string back
import ast
def check(inbool):
    if inbool == 'True' or inbool == 'TRUE' or inbool == 'true':
        return True
    elif inbool == 'False' or inbool == 'FALSE' or inbool == 'false':
        return False
    elif inbool == 'None' or inbool == 'NONE' or inbool == 'none':
        return None
    elif "[" in inbool or "(" in inbool:
        return ast.literal_eval(inbool)
    else:
        inbool = numbertype(inbool)
    return inbool
'''
# Read text files with PWMs
def read_pwm(pwmlist):
    names = []
    pwms = []
    pwm = []
    obj = open(pwmlist, 'r').readlines()
    for l, line in enumerate(obj):
        line = line.strip().split()
        if len(line) > 0:
            if line[0] == 'Motif':
                names.append(line[1])
                pwm = []
            elif line[0] == 'Pos':
                nts = line[1:]
            elif isinstance(numbertype(line[0]), int):
                pwm.append(line[1:])
        elif len(line) == 0 and len(pwm) > 0:
            pwm = np.array(pwm, dtype = float)
            pwms.append(np.array(pwm).T)
            pwm = []
    return pwms, names
'''
# Read text files with PWMs
def read_pwm(pwmlist, nameline = 'Motif'):
    names = []
    pwms = []
    pwm = []
    obj = open(pwmlist, 'r').readlines()
    for l, line in enumerate(obj):
        line = line.strip().split('\t')
        if ((len(line) == 0) or (line[0] == '')) and len(pwm) > 0:
            pwm = np.array(pwm, dtype = float)
            pwms.append(np.array(pwm))
            pwm = []
            names.append(name)
        elif len(line) > 0:
            if line[0] == nameline:
                name = line[1]
                pwm = []
            elif line[0] == 'Pos':
                nts = line[1:]
            elif isinstance(numbertype(line[0]), int):
                pwm.append(line[1:])
        
    return pwms, names

def rescale_pwm(pfms, infcont = False, psam = False, norm = False):
    pwms = []
    for p, pwm in enumerate(pfms):
        if infcont:
            pwm = np.log2((pwm+0.001)*float(len(pwm)))
            pwm[pwm<0] = 0
        if psam:
            pnorm = np.amax(pwm, axis = 0)
            pnorm[pnorm == 0] = 1
            pwm = pwm/pnorm
        pwms.append(pwm)
    if norm:
        len_pwms = [np.sum(pwm) for pwm in pwms]
        pwms = [pwm/(2.*np.sqrt(np.amax(len_pwms))) for pwm in pwms]
    return pwms


def manipulate_input(X, features, sysargv):
    if '--select_features' in sysargv:
        selected_feat = np.genfromtxt(sysargv[sysargv.index('--select_features')+1], dtype = str)
        if len(features[0]) == len(selected_feat[0]):
            featmask = np.isin(features, selected_feat)
        elif len(features[0]) < len(selected_feat[0]):
            selected_feat = ','.join(selected_feat)
            s_feat = []
            for feat in feature:
                if feat in selected_feat:
                    s_feat.append(feat)
            featmask = np.isin(features, s_feat)
        elif len(features[0]) > len(select_features[0]):
            selected_feat = ','.join(selected_feat)
            s_feat = []
            for feat in feature:
                if feat in selected_feat:
                    s_feat.append(feat)
            featmask = np.isin(features, s_feat)
        features, X = np.array(features)[featmask], X[:, featmask]
        print('X reduced to', np.shape(X))
        outname+= '_featsel'+str(len(X[0]))
        
        
    if '--centerfeature' in sysargv:
        outname += '-cenfeat'
        X = X - np.mean(X, axis = 0)
    
    elif '--centerdata' in sysargv:
        outname += '-cendata'
        X = X - np.mean(X, axis = 1)[:, None]


    if '--norm2feature' in sysargv:
        outname += '-n2feat'
        norm = np.sqrt(np.sum(X*X, axis = 0))
        norm[norm == 0] = 1.
        X = X/norm
        
    elif '--norm1feature' in sysargv:
        outname += '-n1feat'
        norm =np.sum(np.absolute(X), axis = 0)
        norm[norm == 0] = 1.
        X = X/norm
    
    if '--norm2data' in sysargv:
        outname += '-n2data'
        norm =np.sqrt(np.sum(X*X, axis = 1))[:, None] 
        X = X/norm
        
    elif '--norm1data' in sysargv:
        outname += '-n1data'
        X = X/np.sum(np.absolute(X), axis = 1)[:,None]
        
    return X, features


