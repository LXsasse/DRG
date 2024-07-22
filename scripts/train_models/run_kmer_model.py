import sys, os 
import numpy as np
import scipy.stats as stats
from functools import reduce

from drg_tools.io_utils import readin, check, numbertype, create_outname
from drg_tools.model_training import create_sets
from drg_tools.data_processing import manipulate_input
from drg_tools.model_output import print_averages, save_performance
from drg_tools.plotlib import plot_scatter

from drg_tools.torch_regression import torch_Regression
from drg_tools.classical_models import logistic_regression, sk_regression 




    

if __name__ == '__main__':
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    
    
    delimiter = ','
    if '--delimiter' in sys.argv:
        delimiter = str(sys.argv[sys.argv.index('--delimiter')+1])
        print(delimiter)
    X, Y, names, features, experiments = readin(inputfile, outputfile, delimiter = delimiter,return_header = True)
    
    if ',' in inputfile:
        inputfiles = inputfile.split(',')
        inputfile = inputfiles[0]
        for inp in inputfiles[1:]:
            inputfile = create_outname(inp, inputfile, lword = 'and')
            print(inputfile)
    outname = create_outname(inputfile, outputfile) 
    print(outname)
    
    
    if '--outdir' in sys.argv:
        outname = sys.argv[sys.argv.index('--outdir')+1] + os.path.split(outname)[1]

    
    X, features = manipulate_input(X, features, sys.argv)
    
    # use a random number of samples from the entire data set to test model
    if '--testrandom' in sys.argv:
        subsamp = float(sys.argv[sys.argv.index('--testrandom')+1])
        if subsamp < 1:
            subsamp = int(len(X)*subsamp)
        subsamp = int(subsamp)
        outname +='ss'+str(subsamp)
        sub = np.random.permutation(len(X))[:subsamp]
        names, X, Y = names[sub], X[sub], Y[sub]
    
    if '--crossvalidation' in sys.argv:
        folds = check(sys.argv[sys.argv.index('--crossvalidation')+1])
        fold = int(sys.argv[sys.argv.index('--crossvalidation')+2])
        if '--significant_genes' in sys.argv:
            siggenes = np.genfromtxt(sys.argv[sys.argv.index('--significant_genes')+1], dtype = str)
            Yclass = np.isin(names, siggenes).astype(int)
        elif '--gene_classes' in sys.argv:
            Yclass = []
            for n, name in enumerate(names):
                Yclass.append(name.split(':')[0])
            Yclass = np.array(Yclass)
        else:    
            cutoff = float(sys.argv[sys.argv.index('--crossvalidation')+3])
            Yclass = (np.sum(np.absolute(Y)>=cutoff, axis = 1) > 0).astype(int)
    else:
        folds, fold, Yclass = 10, 0, None
        
    if isinstance(folds,int):
        outname += '-cv'+str(folds)+'-'+str(fold)
    else:
        outname += '-cv'+str(int(cutoff))+'-'+str(fold)
    trainset, testset, valset = create_sets(len(X), folds, fold, Yclass = Yclass, genenames = names)
    
    print('Train', len(trainset))
    print('Test', len(testset), testset)
    print('Val', len(valset))
    
    
    if '--norm2output' in sys.argv:
        print ('ATTENTION: output has been normalized along data points')
        outname += '-n2out'
        outnorm =np.sqrt(np.sum(Y*Y, axis = 1))[:, None] 
        Y = Y/outnorm
        
    elif '--norm2outputclass' in sys.argv:
        print('ATTENTION: output has been normalized along data classess')
        outname += '-n2outclass'
        outnorm =np.sqrt(np.sum(Y*Y, axis = 0))
        Y = Y/outnorm

    weights = None
    if '--sample_weights' in sys.argv:
        weightfile = np.genfromtxt(sys.argv[sys.argv.index('--sample_weights')+1], dtype = str)
        outname += '_sweigh'
        weight_names, weights = weightfile[:,0], weightfile[:,1].astype(float)
        sortweight = np.argsort(weight_names)[np.isin(np.sort(weight_names), names)]
        weight_names, weights = weight_names[sortweight], weights[sortweight]
        if not np.array_equal(weight_names, names):
            print("Weights cannot be used")
            sys.exit()
        weights = weights[trainset]
    
    #alpha=1.0, fit_intercept=True, max_iter=None, tol=None, solver='auto', positive=False, random_state=None, penalty = None, pca = None, center = False, optimize_alpha = True, change_alpha =.6, validation_axis = 1, alpha_search = 'independent', normalize = False, full_nonlinear = False, logistic = False, warm_start = False, n_jobs = None
    param_names = {'alpha':'l1reg_last', 'fit_intercept':'kernel_bias', 'penalty':'loss_function', 'max_iter':'epochs', 'tol':'patience', 'solver':'optimizer', 'positive':'kernel_function', 'random_state':'seed', 'pca': 'num_kernels', 'center': 'center', 'optimize_alpha': 'init_adjust', 'change_alpha':'kernel_lr', 'validation_axis': 'optim_params', 'alpha_search': 'adjust_lr', 'normalize': 'batch_norm', 'full_nonlinear': 'interaction_layer', 'logistic': 'outclass', 'warm_start': 'hot_start', 'n_jobs': 'device'}
    
    if '--skregression' in sys.argv:
        fit_intercept = check(sys.argv[sys.argv.index('--skregression')+1])
        penalty = sys.argv[sys.argv.index('--skregression')+2]
        alpha = float(sys.argv[sys.argv.index('--skregression')+3])

        outname += '_reglr'+penalty+'-'+str(alpha)+'fi'+str(fit_intercept)[0]
        
        params = {}
        if len(sys.argv) > sys.argv.index('--skregression')+4:
            if '--' not in sys.argv[sys.argv.index('--skregression')+4]:
                if '+' in sys.argv[sys.argv.index('--skregression')+4]:
                    parameters = sys.argv[sys.argv.index('--skregression')+4].split('+')
                else:
                    parameters = [sys.argv[sys.argv.index('--skregression')+4]]
                for p in parameters:
                    if ':' in p and '=' in p:
                        p = p.split('=',1)
                    elif ':' in p:
                        p = p.split(':',1)
                    elif '=' in p:
                        p = p.split('=',1)
                    params[p[0]] = check(p[1])
                    outname += p[0][:2]+p[0][max(2,len(p[0])-2):]+str(p[1])
        
        model = sk_regression(fit_intercept = fit_intercept, alpha = alpha, penalty = penalty, **params)
        
        obj= open(outname+'_model_params.dat', 'w')
        obj.write('outname : '+outname+'\n'+'kernel_bias : '+str(fit_intercept)+'\n'+'loss_function : '+str(penalty)+'\n'+'l1reg_last : '+str(alpha)+'\n')
        if len(params) > 0:
            for par in params:
                obj.write(param_names[par] + ' : '+str(params[par])+'\n')
                
        print(outname+'_model_params.dat')
        
        
    print('Fit')
    model.fit(X[trainset], Y[trainset], XYval = [X[valset], Y[valset]], weights = weights)
    Y_pred = model.predict(X[testset])
    
    if '--norm2output' in sys.argv:
        Y *= outnorm
        Y_pred *= outnorm[testset]
        
    elif '--norm2outputclass' in sys.argv:
        Y *= outnorm
        Y_pred *= outnorm
    
    if '--random_model' in sys.argv:
        could_add_modelthatlearns_on_random_data = 0
    
    if '--save_predictions' in sys.argv:
        np.savetxt(outname+'_pred.txt', np.append(names[testset][:, None], Y_pred, axis = 1), header = ' '.join(experiments), fmt = '%s')
        print('SAVED', outname+'_pred.txt')    

    if '--split_outclasses' in sys.argv:
        testclasses = np.genfromtxt(sys.argv[sys.argv.index('--split_outclasses')+1], dtype = str)
        tsort = []
        for exp in experiments:
            tsort.append(list(testclasses[:,0]).index(exp))
        testclasses = testclasses[tsort][:,1]
    else:
        testclasses = np.zeros(len(Y[0]), dtype = np.int8).astype(str)
    
    if '--significant_genes' in sys.argv:
        siggenes = np.genfromtxt(sys.argv[sys.argv.index('--significant_genes')+1], dtype = str)
        tlen = len(testset)
        tsetmask = np.isin(names[testset], siggenes)
        testset = testset[tsetmask]
        Y_pred = Y_pred[tsetmask]
        print('Testset length reduced to significant genes from', tlen, 'to', len(testset))

    # USE: --aurocaverage --auprcaverage --mseaverage --correlationaverage
    print_averages(Y_pred, Y[testset], testclasses, sys.argv)
    
    # USE: --save_correlation_perclass --save_auroc_perclass --save_auprc_perclass --save_mse_perclass --save_correlation_pergene '--save_mse_pergene --save_auroc_pergene --save_auprc_pergene --save_topdowncorrelation_perclass
    save_performance(Y_pred, Y[testset], testclasses, experiments, names[testset], outname, sys.argv, compare_random = True)
    
    # reducing X to features that
    features = np.array(features)
    if model.Xmask is not None:
        X, features = X[:, model.Xmask], np.array(features)[model.Xmask]
        model.Xmask = None
    
    if '--feature_weights' in sys.argv:
        coef_mask = model.reduced_coef()
        np.savetxt(outname+'_features.dat', np.append(features[coef_mask].reshape(-1,1),np.around(model.coef_.T[coef_mask],5), axis = 1).astype(str), fmt = '%s', header = ' '.join(experiments))
    
    if '--feature_zscores' in sys.argv:
        # saves sign(coef)*log10(pvalues)
        logpvalues, coef_mask = model.statistical_weight(X[trainset], Y[trainset], comp_pvalues = False)
        np.savetxt(outname+'_feature_zscores.dat', np.append(features[coef_mask].reshape(-1,1),np.around(logpvalues.T,2), axis = 1).astype(str), fmt = '%s', header = ' '.join(experiments))
    
    if '--feature_statistics' in sys.argv:
        # saves sign(coef)*log10(pvalues)
        logpvalues, coef_mask = model.statistical_weight(X[trainset], Y[trainset], compute_new = False)
        np.savetxt(outname+'_feature_stats.dat', np.append(features[coef_mask].reshape(-1,1),np.around(logpvalues.T,2), axis = 1).astype(str), fmt = '%s', header = ' '.join(experiments))
    
        
        # plots scatter plot for each output class
    if '--plot_correlation_perclass' in sys.argv:
        plot_scatter(Y[testset], Y_pred, xlabel = 'Measured', ylabel = 'Predicted', titles = experiments, outname = outname + '_class_scatter.jpg')
        
    # plots scatter plot fo n_genes that within the n_genes quartile
    if '--plot_correlation_pergene' in sys.argv:
        n_genes = int(sys.argv['--plot_correlation_pergene'])
        for tclass in np.unique(testclasses):
            correlation_genes = correlation(Ytest[np.random.permutation(len(Ytest))][:,consider], Y_pred[:,consider], axis = 1)
            sort = np.argsort(correlation_genes)
            posi = np.linspace(0,len(correlation_genes), n_genes).astype(int)
            i = sort[posi] 
            plot_scatter(Y[test][i].T, Ypred[i].T, xlabel = 'Measured', ylabel = 'Predicted', titles = names[testset][i], outname = outname + '_gene_scatter.jpg')








