#simple_simulation.py
# Simple simulation that simulates random mutations in sequences for different individuals
# Major chalenge is to handle amount of sequences that need to go into model
import numpy as np
import sys, os
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from scipy.stats import pearsonr

n_motifs = 100 
l_motifs = 7

n_seqs = 10000
l_seqs = 1000

noiseratio = 0.25
nindividuals = 1000
# each individual has its personal noise added
mutation_rate = 100000. # on average every x bp is mutated in the individuals sequence

if len(sys.argv) >= 8:
    n_motifs = int(sys.argv[1])
    l_motifs = int(sys.argv[2])

    n_seqs = int(sys.argv[3])
    l_seqs = int(sys.argv[4])

    noiseratio = float(sys.argv[5])
    nindividuals = float(sys.argv[6])
    mutation_rate = float(sys.argv[7])


np.random.seed(101)

outname = 'Simulated_sequences'+str(n_seqs)+'_len'+str(l_seqs)+'_nmot'+str(n_motifs)+'lm'+str(l_motifs)+'noise'+str(noiseratio)+'_for'+str(nindividuals)+'inds_atrate'+str(mutation_rate)


motifs = []
motif_effect = np.random.random(n_motifs)-0.5
i = 0
while i < n_motifs:
    nmotif = ''.join(np.random.choice(['A', 'C', 'G', 'T',], size = l_motifs))
    if nmotif not in motifs:
        motifs.append(nmotif)
        i += 1
motifs = np.sort(motifs)

    
sequences = []
for s in range(n_seqs):
    sequences.append(''.join(np.random.choice(['A', 'C', 'G', 'T',], size = l_seqs)))

sequence_features = np.zeros((n_seqs, n_motifs))
featuressets = []
shortfeaturesets = []
longfeaturesets = []
for s, seq in enumerate(sequences):
    sfeat = []
    ssfeat = []
    lsfeat = []
    for l in range(l_seqs-l_motifs+1):
        sfeat.append(seq[l:l+l_motifs])
        ssfeat.append(seq[l:l+l_motifs-1])
        if l < l_seqs-l_motifs:
            lsfeat.append(seq[l:l+l_motifs+1])
    ssfeat.append(seq[l+1:l+l_motifs])
    featuressets.append(sfeat)
    shortfeaturesets.append(ssfeat)
    longfeaturesets.append(lsfeat)
    sfeat, nfeat = np.unique(sfeat, return_counts = True)
    sequence_features[s][np.isin(motifs, sfeat)] = nfeat[np.isin(sfeat, motifs)]
    
# Generate full feature set to check performance without prior knowledge
full_features = np.unique(np.concatenate(featuressets))
full_short_features = np.unique(np.concatenate(shortfeaturesets))
full_long_features = np.unique(np.concatenate(longfeaturesets))

full_sequence_features = np.zeros((n_seqs, len(full_features)))
full_sequence_shortfeatures = np.zeros((n_seqs, len(full_short_features)))
full_sequence_longfeatures = np.zeros((n_seqs, len(full_long_features)))
for s, sfeat in enumerate(featuressets):
    sfeat, sfnum = np.unique(sfeat, return_counts = True)
    full_sequence_features[s, np.isin(full_features, sfeat)] = sfnum
    sfeat, sfnum = np.unique(shortfeaturesets[s], return_counts = True)
    full_sequence_shortfeatures[s, np.isin(full_short_features, sfeat)] = sfnum
    sfeat, sfnum = np.unique(longfeaturesets[s], return_counts = True)
    full_sequence_longfeatures[s, np.isin(full_long_features, sfeat)] = sfnum
    
expression = np.sum(sequence_features*motif_effect, axis = 1)

sequences = np.array([list(seq) for seq in sequences])

indexpression = np.ones((n_seqs, nindividuals)) * expression[:, None]
mutations = []
mutation_features = []
for i in range(nindividuals):
    
    randpos = np.random.randint(n_seqs*l_seqs, size = int(n_seqs*l_seqs/mutation_rate))
    randseq = (randpos/l_seqs).astype(int)
    randpos = randpos%l_seqs
    randbase = sequences[randseq, randpos]
    randbases2 = sequences[randseq, randpos]
    bases = np.array(['A', 'C', 'G', 'T'])
    for base in bases:
        randbase[randbases2 == base] = np.random.choice(bases[bases!= base], size = int(np.sum(randbases2 == base)))
    mutations.append(np.array([randseq,randpos,randbase]).T)
    # change expression according to variation
        # Generate new feature vector:
    #print randpos
    #print randseq
    #print randbase
    #print randbases2
    mut_feat = []
    for z, rs in enumerate(zip(randseq, randpos)):
        rs, r = rs
        oldset = []
        newset = []
        oldseq = ''.join(sequences[rs][max(0,r-l_motifs+1): min(r+l_motifs,l_seqs)])
        newseq = np.copy(sequences[rs])
        newseq[r] = randbase[z]
        newseq = ''.join(newseq[max(0,r-l_motifs+1): min(r+l_motifs,l_seqs)])
        #print r, max(0,r-l_motifs+1), min(r+l_motifs,l_seqs)
        #print oldseq
        #print newseq
        for k in range(len(newseq)):
            oldset.append(oldseq[k:k+l_motifs])
            newset.append(newseq[k:k+l_motifs])
        oldset, on = np.unique(oldset, return_counts = True)
        newset, nn = np.unique(newset, return_counts = True)
        featvec = np.copy(sequence_features[rs])
        
        if np.isin(motifs,oldset).any():
            #print 'old', motifs[np.isin(motifs,oldset)]
            #print np.nonzero(featvec)[0]
            #print indexpression[rs, i]
            indexpression[rs, i] -= np.sum(on[np.isin(oldset,motifs)]*motif_effect[np.isin(motifs, oldset)])
            #print indexpression[rs, i]
            #print motif_effect[np.isin(motifs, oldset)]
            featvec[np.isin(motifs,oldset)] -= on[np.isin(oldset,motifs)]
            #print np.nonzero(featvec)[0]
        if np.isin(motifs,newset).any():
            #print "new", motifs[np.isin(motifs,newset)]
            #print np.nonzero(featvec)[0]
            #print indexpression[rs, i]
            #print motif_effect[np.isin(motifs, newset)]
            indexpression[rs, i] += np.sum(nn[np.isin(newset,motifs)]*motif_effect[np.isin(motifs, newset)])
            #print indexpression[rs, i]
            featvec[np.isin(motifs,newset)] += nn[np.isin(newset,motifs)]
            #print np.nonzero(featvec)[0]
        mut_feat.append(featvec)
    mutation_features.append(np.array(mut_feat))
            

ntimesnonlin = None
if '--non_linearities' in sys.argv or '--all_non_linearities' in sys.argv:
    if '--non_linearities' in sys.argv:
        i = sys.argv.index('--non_linearities')+1
    else:
        i = sys.argv.index('--all_non_linearities')+1
    ntimesnonlin = float(sys.argv[i])
    if '--all_non_linearities' in sys.argv:
        outname += '_allnonlinear'+str(ntimesnonlin)
    else:
        outname += '_nonlinear'+str(ntimesnonlin)
    n_nonlinear = int(n_motifs*ntimesnonlin)
    nlin_features = np.zeros((n_seqs, n_nonlinear))
    nlin_activation = np.random.random(n_nonlinear)-0.5
    for n in range(n_nonlinear):
        randfeat = np.random.randint(n_motifs, size = 2)
        nlin_features[:, n] = sequence_features[:, randfeat[0]] * sequence_features[:, randfeat[1]]
        for i in range(nindividuals):
            nonlin_mutationfeat = mutation_features[i][:,randfeat[0]] * mutation_features[i][:,randfeat[1]]
            infeat = np.copy(nlin_features[:,n])
            infeat[mutations[i][:,0].astype(int)] = nonlin_mutationfeat
            indexpression[:,i] += infeat * nlin_activation[n]
            if '--all_non_linearities' in sys.argv:
            # non-linearities override linearities in sequence
            # remove linearities if they occur in seqwuence with non_linearity
                indvfeat = np.copy(sequence_features)
                indvfeat[mutations[i][:,0].astype(int)] = mutation_features[i]
                indvfeat = indvfeat[infeat > 0]
                indexpression[:,i][infeat>0] -= np.dot(indvfeat[:,randfeat], motif_effect[randfeat])
            
            
    
    if '--print_stats':
        n_feat_inseq = np.sum(nlin_features > 0, axis = 0)
        n_seq_infeat = np.sum(nlin_features > 0, axis = 1)
        print 'N nonlinear features in all sequences:'
        print 'mean', np.mean(n_feat_inseq), "meadia", np.median(n_feat_inseq), '+-', np.std(n_feat_inseq)
        print 'N nonlinear features per sequence:'
        print 'mean', np.mean(n_seq_infeat), "meadia", np.median(n_seq_infeat), '+-', np.std(n_seq_infeat)

expression = indexpression
sequences = [''.join(sequence) for sequence in sequences]
        
expeffect = False
if '--exponential_effect' in sys.argv:
    expeffect = True
    expression = np.exp(expression)
    outname += '_exponential-effect'

print n_motifs, l_motifs, n_seqs, l_seqs, noiseratio, ntimesnonlin, expeffect, nindividuals, mutation_rate

mean_expression = np.mean(expression)
var_expression = np.std(expression)

expression += np.random.normal(loc = 0, scale = var_expression*noiseratio, size = (n_seqs,nindividuals))


if '--repress_output' not in sys.argv:
    genenames = np.array(['G_'+str(i) for i in range(n_seqs)])
    obj = open(outname+'_sequences.fasta', 'w')
    for i in range(n_seqs):
        obj.write('>'+genenames[i]+'\n'+sequences[i]+'\n')
    header = np.array(['C'+str(i) for i in range(nindividuals)])
    np.savetxt(outname+'_expression.dat', np.append(genenames.reshape(-1,1),expression, axis = 1), fmt = '%s', header = ' '.join(header))
    np.savetxt(outname+'_motifset.dat', motifs, fmt = '%s')
    muttext = []
    for mutation in mutations:
        muttext.append([':'.join(mut.astype(str)) for mut in mutation])
    muttext = np.array(muttext).T
    np.savetxt(outname+'_snps.dat', muttext, fmt = '%s', header = ' '.join(header))


def teststats(name, real, pred):
    print name, np.around(pearsonr(real, pred)[0], 2), np.around(np.mean((real-pred)**2),4)  
    
# instead of fitting nindividuals*nseqs data points, we can find the mean expression of an unchanged sequence by fitting a distribution to it and using the mean of that distribution as our value for the sequence which will only be inserted once.

mutations = np.array(mutations)[:,:,:2].astype(int)
modifs = np.array(zip(np.concatenate(mutations[:, :, 0]), (np.ones((np.shape(mutations)[1], np.shape(mutations)[0]), dtype = int)*np.arange(len(mutations),dtype = int)).T.flatten()))
modseqs = np.unique(modifs[:,0])
kept_seq = np.delete(np.arange(len(sequences), dtype = int), modseqs)
summedexpression = expression[:,0]
summedexpression[kept_seq] = np.mean(expression[kept_seq],axis = 1) # Assumption of Gassian Noise
weights = np.ones(len(summedexpression)+len(modifs))
keep = []
for m , mod in enumerate(modifs):
    summedexpression = np.append(summedexpression, expression[mod[0], mod[1]])
    weights[len(summedexpression)-1] = 1./float(nindividuals)
    sequence_features = np.append(sequence_features, mutation_features[mod[1]][[m-len(np.where(modifs[:,1]<mod[1])[0])]], axis = 0)
    origexp = np.delete(np.arange(nindividuals, dtype = int), modifs[modifs[:,0] == mod[0],1])
    keep.append(len(summedexpression)-1)
    if len(origexp) > 0:
        summedexpression[mod[0]] = np.mean(expression[mod[0]][origexp])
        weights[mod[0]] = float(len(origexp))/float(nindividuals) + 1e-8
        keep.append(mod[0])

#So that gives us a theoretical bound given by the mutation rate in the regulatory sequence, like the promoter
# The higher the frequency, the more sequence variability we get. With length 200 every 50 sequence is effected by mutation. That means we get 20 more examples for each person we measure with mutation. Now we just have to find out how many sequences we need for sufficient fit. 


keep = np.sort(keep)
expression = summedexpression[keep]
sequence_features = sequence_features[keep]
weights = weights[keep]

#print np.sum(sequence_features, axis = 0)
#print expression
#print weights

# Quicktest
testsets = np.random.permutation(len(sequence_features))
trainset = testsets[:int(float(len(testsets))*0.7)]
testset = testsets[int(float(len(testsets))*0.7):]

sequence_features = sequence_features[:, (np.sum(sequence_features[trainset],axis = 0)>1) &(np.sum(sequence_features[testset],axis = 0)>0)]



# linear Regression with the correct features
lr = LinearRegression( fit_intercept = True)
lr.fit(sequence_features[trainset], expression[trainset], weights[trainset] )
expred = lr.predict(sequence_features[testset])

teststats('LRknowledge', expression[testset].flatten(), expred.flatten())


if '--assess_model' in sys.argv:
    def findbest_hyper(model, x, y, xval, yval, start = 1., inc = 1./2.):
        model.set_params(alpha = start, fit_intercept = True)
        model.fit(x,y)
        pred = model.predict(xval)
        perform = 1. - pearsonr(pred.flatten(),yval.flatten())[0]
        if np.isnan(perform):
            perform = 2.
        #print 'Start performance', perform
        alpha = np.copy(start)*inc
        j = 0
        while True:
            model.set_params(alpha = alpha, fit_intercept = True)
            model.fit(x,y)
            pred = model.predict(xval)
            nperform = 1. - pearsonr(pred.flatten(),yval.flatten())[0] #np.mean((pred-yval)**2)
            if np.isnan(nperform):
                nperform = 2.
            #print alpha, nperform, pearsonr(pred.flatten(),yval.flatten())[0]
            if np.std(pred)/np.std(y) < 1e-8:
                alpha = alpha*inc
                perform = np.copy(nperform)
                j = 0
            elif nperform < perform:
                alpha = alpha*inc
                perform = np.copy(nperform)
            elif nperform >= perform and j == 0:
                inc = 2.
                alpha = np.copy(start)*inc
            elif nperform >= perform and j >= 1:
                alpha = alpha/inc
                model = model.set_params(alpha = alpha, fit_intercept = True)
                model.fit(x,y)
                pred = model.predict(xval)
                nperform = 1. - pearsonr(pred.flatten(),yval.flatten())[0]
                #print alpha, nperform, pearsonr(pred.flatten(),yval.flatten())[0]
                return pred, alpha
            j+=1

        

    # linear Regression with all features
    lr = LinearRegression(fit_intercept = True)
    lr.fit(full_sequence_features[trainset], expression[trainset])
    expred = lr.predict(full_sequence_features[testset])
    teststats('LR', expression[testset].flatten(), expred.flatten())

    # linear Regression with short features
    lr = LinearRegression(fit_intercept = True)
    lr.fit(full_sequence_shortfeatures[trainset], expression[trainset])
    expred = lr.predict(full_sequence_shortfeatures[testset])
    teststats('LRshort', expression[testset].flatten(), expred.flatten())

    # linear Regression with long features
    lr = LinearRegression(fit_intercept = True)
    lr.fit(full_sequence_longfeatures[trainset], expression[trainset])
    expred = lr.predict(full_sequence_longfeatures[testset])
    teststats('LRlong', expression[testset].flatten(), expred.flatten())

    # linear Regression with mixed features
    full_sequence_shortfeatures = np.concatenate([full_sequence_shortfeatures, full_sequence_features, full_sequence_longfeatures], axis = 1)
    lr = LinearRegression(fit_intercept = True)
    lr.fit(full_sequence_shortfeatures[trainset], expression[trainset])
    expred = lr.predict(full_sequence_shortfeatures[testset])
    teststats('LRmixed', expression[testset].flatten(), expred.flatten())


    # Lasso regression
    lr = Lasso()
    expred, apha = findbest_hyper(lr, full_sequence_features[trainset], expression[trainset], full_sequence_features[testset], expression[testset])
    teststats('Lasso_'+str(apha), expression[testset].flatten(), expred.flatten())

    # Lasso with mixed features
    lr = Lasso()
    expred, apha = findbest_hyper(lr, full_sequence_shortfeatures[trainset], expression[trainset], full_sequence_shortfeatures[testset], expression[testset])
    teststats('Lassomixed_'+str(apha), expression[testset].flatten(), expred.flatten())


    # Ridge regression
    lr = Ridge()
    expred, apha = findbest_hyper(lr, full_sequence_features[trainset], expression[trainset], full_sequence_features[testset], expression[testset])
    teststats('Ridge_'+str(apha), expression[testset].flatten(), expred.flatten())


    if '--non_linearities' in sys.argv:
        lr.fit(np.append(sequence_features[trainset], nlin_features[trainset], axis = 1), expression[trainset])
        expred = lr.predict(np.append(sequence_features[testset], nlin_features[testset], axis = 1))
        
        teststats('LRknowledgenonlinear', expression[testset].flatten(), expred.flatten())
        
        # Lasso regression
        lr = Lasso()
        nonlin_full_sequence_features = []
        for k  in range(len(full_sequence_features[0])):
            for l in range(k+1, len(full_sequence_features[0])):
                nonlin_full_sequence_features.append(full_sequence_features[:,k]*full_sequence_features[:,l])
                
        nonlin_full_sequence_features = np.append(full_sequence_features, np.array(nonlin_full_sequence_features).T, axis = 1)
        expred, apha = findbest_hyper(lr, nonlin_full_sequence_features[trainset], expression[trainset], nonlin_full_sequence_features[testset], expression[testset])
        teststats('Lassononlinear_'+str(apha), expression[testset].flatten(), expred.flatten())









    
    
    
