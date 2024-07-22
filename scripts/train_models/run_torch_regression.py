import sys, os 
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from drg_tools.torch_regression import torch_Regression

from drg_tools.io_utils import readin, check, numbertype, create_outname
from drg_tools.model_training import create_sets
from drg_tools.data_processing import manipulate_input
from drg_tools.model_output import print_averages, save_performance
    

if __name__ == '__main__':
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    outname = create_outname(inputfile, outputfile) 
    
    if '--outdir' in sys.argv:
        outname = sys.argv[sys.argv.index('--outdir')+1] + os.path.split(outname)[1]

    delimiter = ','
    if '--delimiter' in sys.argv:
        delimiter = str(sys.argv[sys.argv.index('--delimiter')+1])
        print(delimiter)
    X, Y, names, features, experiments = readin(inputfile, outputfile, delimiter = delimiter,return_header = True)
    
    X, features = manipulate_input(X, features, sys.argv)
    
    # use a random number of samples from the entire data set
    if '--sub_sample' in sys.argv:
        subsamp = float(sys.argv[sys.argv.index('--sub_sample')+1])
        if subsamp < 1:
            subsamp = int(len(X)*subsamp)
        subsamp = int(subsamp)
        outname +='ss'+str(subsamp)
        sub = np.random.permutation(len(X))[:subsamp]
        names, X, Y = names[sub], X[sub], Y[sub]
    
    if '--crossvalidation' in sys.argv:
        folds = int(sys.argv[sys.argv.index('--crossvalidation')+1])
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
        
    outname += '-cv'+str(folds)+'-'+str(fold)
    trainset, testset, valset = create_sets(len(X), folds, fold, Yclass = Yclass)
    
    
    if '--norm2output' in sys.argv:
        outname += '-n2out'
        norm =np.sqrt(np.sum(Y*Y, axis = 1))[:, None] 
        Y = Y/norm
        
    elif '--norm2outputclass' in sys.argv:
        outname += '-n2outclass'
        norm =np.sqrt(np.sum(Y*Y, axis = 0))
        Y = Y/norm

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
    
    if '--regression' in sys.argv:
        params = {}
        params['device'] = get_device()
        print('Device', params['device'])
        if len(sys.argv) > sys.argv.index('--regression') + 1:
            parameters = sys.argv[sys.argv.index('--regression')+1]
            if '+' in parameters:
                parameters = parameters.split('+')
            else:
                parameters = [parameters]
            for p in parameters:
                if ':' in p and '=' in p:
                    p = p.split('=',1)
                elif ':' in p:
                    p = p.split(':',1)
                elif '=' in p:
                    p = p.split('=',1)
                params[p[0]] = check(p[1])
        params['outname'] = outname
        
        model = torch_Regression(**params)
    
    model.fit(X[trainset], Y[trainset], XYval = [X[valset], Y[valset]], weights = weights)
    Y_pred = model.predict(X[testset])
    
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
    
    if '--feature_statistics' in sys.argv:
        # saves sign(coef)*log10(pvalues)
        logpvalues = model.statistical_weight(X[trainset], Y[trainset])
        np.savetxt(outname+'_feature_stats.dat', np.append(np.array(features).reshape(-1,1),np.around(logpvalues.T,2), axis = 1).astype(str), fmt = '%s', header = ' '.join(experiments))
    
    if '--feature_weights' in sys.argv:
        np.savetxt(outname+'_features.dat', np.append(np.array(features).reshape(-1,1),np.around(model.coef_.T[:-1],5), axis = 1).astype(str), fmt = '%s', header = ' '.join(experiments))
        
        
    








