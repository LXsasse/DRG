#simple_simulation.py
import numpy as np
import sys, os
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

n_motifs = 100 
l_motifs = 7

n_seqs = 1000
l_seqs = 400

noiseratio = 0.25

if len(sys.argv) >= 6 and '--' not in sys.argv[1]:
    n_motifs = int(sys.argv[1])
    l_motifs = int(sys.argv[2])

    n_seqs = int(sys.argv[3])
    l_seqs = int(sys.argv[4])

    noiseratio = float(sys.argv[5])



np.random.seed(101)

outname = 'Simulated_sequences'+str(n_seqs)+'_len'+str(l_seqs)+'_nmotifs'+str(n_motifs)+'lm'+str(l_motifs)+'noise'+str(noiseratio)


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
    
if '--print_stats':
    n_feat_inseq = np.sum(sequence_features > 0, axis = 0)
    n_seq_infeat = np.sum(sequence_features > 0, axis = 1)
    print 'N features in all sequences:'
    print 'mean', np.mean(n_feat_inseq), "meadia", np.median(n_feat_inseq), '+-', np.std(n_feat_inseq)
    print 'N features per sequence:'
    print 'mean', np.mean(n_seq_infeat), "meadia", np.median(n_seq_infeat), '+-', np.std(n_seq_infeat)
    


expression = np.sum(sequence_features*motif_effect, axis = 1)

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
    for n in range(n_nonlinear):
        randfeat = np.random.randint(n_motifs, size = 2)
        bexp = np.copy(expression)
        nlin_features[:, n] = sequence_features[:, randfeat[0]] * sequence_features[:, randfeat[1]]
        nocc = np.sum(nlin_features > 0, axis = 0)
        if '--all_non_linearities' in sys.argv:
            # non-linearities override linearities in sequence
            # remove linearities if they occur in seqwuence with non_linearity
            expression[nlin_features[:,n]>0] -= np.dot(sequence_features[nlin_features[:,n]>0][:,randfeat], motif_effect[randfeat])
            
    nlin_activation = np.random.random(n_nonlinear)-0.5
    expression += np.dot(nlin_features, nlin_activation)
    
    if '--print_stats':
        n_feat_inseq = np.sum(nlin_features > 0, axis = 0)
        n_seq_infeat = np.sum(nlin_features > 0, axis = 1)
        print 'N nonlinear features in all sequences:'
        print 'mean', np.mean(n_feat_inseq), "meadia", np.median(n_feat_inseq), '+-', np.std(n_feat_inseq)
        print 'N nonlinear features per sequence:'
        print 'mean', np.mean(n_seq_infeat), "meadia", np.median(n_seq_infeat), '+-', np.std(n_seq_infeat)
            
            
    
        
expeffect = False
if '--exponential_effect' in sys.argv:
    expeffect = True
    expression = np.exp(expression)
    outname += '_exponential-effect'

print n_motifs, l_motifs, n_seqs, l_seqs, noiseratio, ntimesnonlin, expeffect

mean_expression = np.mean(expression)
var_expression = np.std(expression)

expression += np.random.normal(loc = 0, scale = var_expression*noiseratio, size = n_seqs)

expression = expression.reshape(-1,1)

if '--repress_output' not in sys.argv:
    genenames = np.array(['G_'+str(i) for i in range(n_seqs)])
    obj = open(outname+'_sequences.fasta', 'w')
    for i in range(n_seqs):
        obj.write('>'+genenames[i]+'\n'+sequences[i]+'\n')
    np.savetxt(outname+'_expression.dat', np.append(genenames.reshape(-1,1),expression, axis = 1, header = 'C0'), fmt = '%s')
    np.savetxt(outname+'_motifset.dat', motifs, fmt = '%s')


# Quicktest
trainset = np.arange(int(n_seqs*0.7), dtype = int)
testset = np.arange(int(n_seqs*0.7), n_seqs, dtype = int)

def teststats(name, real, pred):
    print name, np.around(pearsonr(real, pred)[0], 2), np.around(np.mean((real-pred)**2),4)  
    

# linear Regression with the correct features
lr = LinearRegression( fit_intercept = True)
lr.fit(sequence_features[trainset], expression[trainset])
expred = lr.predict(sequence_features[testset])
teststats('LRknowledge', expression[testset].flatten(), expred.flatten())

if '--plot_residual' in sys.argv:
    # could add confidence interval, either bootstrapping or compute posterior somehow
    fig = plt.figure(figsize = [9,9])
    residual = expression[testset].flatten()-expred.flatten()

    ax0 = fig.add_subplot(221)
    bins = np.linspace(min(np.amin(expred),np.amin(expression[testset])), max(np.amax(expred),np.amax(expression[testset])), 20)
    ax0.hist(expression[testset].flatten(), bins = bins, alpha = 0.5, label = 'Real')
    ax0.hist(expred.flatten(), bins = bins, alpha = 0.5, label = 'Predicted')
    ax0.set_yscale('log')
    ax0.set_ylabel('Number of genes')
    ax0.set_xlabel('Expression value')
    ax0.legend()
    
    ax2 = fig.add_subplot(222)
    x = [[np.amin(expred.flatten())], [np.amax(expred.flatten())]]
    ax2.scatter(expred.flatten(), expression[testset].flatten(), label = 'R '+str(np.around(pearsonr(expred.flatten(), expression[testset].flatten())[0],2)))
    lr = LinearRegression(fit_intercept = True).fit(expred.reshape(-1,1), expression[testset].flatten())
    y = lr.predict(x)
    ax2.plot(x,y,c='r', label = 'mx_y: '+str(np.around(lr.coef_[0],3)))
    lr = LinearRegression(fit_intercept = True).fit(expression[testset].reshape(-1,1), expred.flatten())
    y = [[np.amin(expression[testset])], [np.amax(expression[testset])]]
    x = lr.predict(y)
    ax2.plot(x,y,c='r', label = 'my_x: '+str(np.around(lr.coef_[0],3)))
    
    ax2.set_ylabel('Real')
    ax2.set_xlabel('Predicted')
    ax2.legend()
    
    ax = fig.add_subplot(223)
    x = [[np.amin(expred.flatten())], [np.amax(expred.flatten())]]
    ax.plot(x,[0,0], c = 'grey')
    ax.set_ylabel('Residual')
    ax.set_xlabel('Predicted')
    ax.scatter(expred, residual)
    lr = LinearRegression( fit_intercept = True).fit(expred.reshape(-1,1), residual)
    y = lr.predict(x)
    ax.plot(x,y,c='r', label = 'mx_y: '+str(round(lr.coef_[0],3)))
    lr = LinearRegression( fit_intercept = True).fit(residual.reshape(-1,1), expred.flatten())
    y = [[np.amin(residual)], [np.amax(residual)]]
    x = lr.predict(y)
    ax.plot(x,y,c='r', label = 'my_x: '+str(round(lr.coef_[0],3)))
    ax.legend()

    ax1 = fig.add_subplot(224)
    x = [[np.amin(expression[testset].flatten())], [np.amax(expression[testset].flatten())]]
    ax1.plot(x,[0,0], c = 'grey')
    ax1.set_ylabel('Residual')
    ax1.set_xlabel('Real')
    ax1.scatter(expression[testset].flatten(), residual)
    lr = LinearRegression( fit_intercept = True).fit(expression[testset].reshape(-1,1), residual)
    y = lr.predict(x)
    ax1.plot(x,y,c='r', label = 'm: '+str(np.around(lr.coef_[0],3)))
    lr = LinearRegression( fit_intercept = True).fit(residual.reshape(-1,1), expression[testset].flatten())
    y = [[np.amin(residual)], [np.amax(residual)]]
    x = lr.predict(y)
    ax1.plot(x,y,c='r', label = 'my_x: '+str(round(lr.coef_[0],3)))
    
    ax1.legend()
    fig.savefig(outname+'_residual_plot.jpg', dpi = 200, bbox_inches = 'tight')
    

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









    
    
    
