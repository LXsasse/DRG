# hyperparameter_search.py

#### Need to find a way to send models for training on gpu with joblib
#### joblib will wait until one process is done before starting next one, like used before
#### However, we need an updating list of gpus that are free
#### and we need a wrapper, so that we can put it all in one line for using processing, 
#### The wrapper initiates the model, trains it and updates the list of free gpus, 
#### Maybe the we don't need a list, we just use i%n_gpus

import numpy as np
import sys, os
import itertools
import time 
import torch
import torch.nn as nn
import cnn_model
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, cosine
from output import save_performance

def compute_memory(model, batchsize, n_feataures, l_sequence):
    input_ = torch.rand(batchsize, n_features, l_sequence)
    totalmemory = batchsize*n_features*l_sequence*32
    for param_tensor in model.state_dict():
        totalmemory += np.prod(model.state_dict()[param_tensor].size())*32
    
    all_layers = []
    def remove_sequential(network):
        for layer in network.children():
            if type(layer) == nn.Sequential: # if sequential layer, apply recursively to layers in sequential layer
                remove_sequential(layer)
            if list(layer.children()) == []: # if leaf node, add it to list
                all_layers.append(layer)       
                
    remove_sequential(model)
    for layer in all_layers:
        input_size = input_.size()
        layer_dict = layer.state_dict()
        if 'weight' in layer_dict:
            required_input_size = layer_dict['weight'].size(dim = 1)
        else:
            required_input_size = input_size[1]
        if required_input_size != input_size[1]:
            input_ = torch.flatten(input_, start_dim = 1)
            
        input_ = layer(input_)
        out_size = np.array(input_.size())
        totalmemory += np.prod(out_size)*32*2
    totalmemory/=8e6
    return totalmemory




# Assess space on gpu and whether another model can be trained on it
def get_free_gpu(model):
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_used = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Total >tmp')
    memory_total = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])
    memory_available = memory_total - memory_used - 6000 
    required_memory = compute_memory(model, model.batchsize, model.n_features, model.l_seqs)
    for m, avail in enumerate(memory_available):
        if required_memory < avail:
            return m, True
    return None, False


# Returns all i long combinations of n numbers
def combinations(i, n):
    out = [[j] for j in range(n)]
    if i == 1:
        return out
    while True:
        nout = []
        for o, ou in enumerate(out):
            for s in range(ou[-1]+1, n):
                no = ou.copy()
                no.append(s)
                nout.append(no)
        out = nout
        if len(out[0]) == i:
            break
    return out





if __name__ == '__main__':
    
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    
    delimiter = ','
    if '--delimiter' in sys.argv:
        delimiter = sys.argv[sys.argv.index('--delimiter')+1]
    
    aregion = True
    if '--regionless' in sys.argv:
        aregion = False
    
    X, Y, names, features, experiments = cnn_model.readin(inputfile, outputfile, delimiter = delimiter,return_header = True, assign_region = aregion)
    
    if ',' in inputfile:
        inputfiles = inputfile.split(',')
        inputfile = inputfiles[0]
        for inp in inputfiles[1:]:
            inputfile = cnn_model.create_outname(inp, inputfile, lword = 'and')
    
    outname = cnn_model.create_outname(inputfile, outputfile) 
    if '--regionless' in sys.argv:
        outname += '-rgls'
    print('Filenames', outname)
    
    # Parameter dictionary for initializing the cnn
    params = {}
    # Define the gpu that all the models will be trained on
    cudanode = sys.argv[3]
    params['device'] = cudanode
    
    n_varying_parameter = int(sys.argv[4]) # number of parameter combinations that should be varied: For example 2 means that all duplets of parameters will be changed
    
    outname += '_'+cudanode.replace(':','')+str(n_varying_parameter)
    
    if '--outdir' in sys.argv:
        outname = sys.argv[sys.argv.index('--outdir')+1] + os.path.split(outname)[1]
    
    if '--addname' in sys.argv:
        outname += '_'+sys.argv[sys.argv.index('--addname')+1]
    
    
    weights = None
    if '--mutation_file' in sys.argv:
        mutfile = sys.argv[sys.argv.index('--mutation_file')+1]
        X,Y,names,experiments,weights = cnn_model.read_mutationfile(mutfile,X,Y,names,experiments)
        if sys.argv[sys.argv.index('--mutation_file')+2] != 'weighted':
            weights = None
        outname = cnn_model.create_outname(mutfile, outname+'.dat', lword = 'with')
        print(outname)

    if '--crossvalidation' in sys.argv:
        folds = cnn_model.check(sys.argv[sys.argv.index('--crossvalidation')+1])
        fold = int(sys.argv[sys.argv.index('--crossvalidation')+2])
        if '--significant_genes' in sys.argv:
            siggenes = np.genfromtxt(sys.argv[sys.argv.index('--significant_genes')+1], dtype = str)
            Yclass = np.isin(names, siggenes).astype(int)
        else:    
            cutoff = float(sys.argv[sys.argv.index('--crossvalidation')+3])
            Yclass = (np.sum(np.absolute(Y)>=cutoff, axis = 1) > 0).astype(int)
    else:
        folds, fold, Yclass = 10, 0, None
    
    if isinstance(folds,int):
        outname += '-cv'+str(folds)+'-'+str(fold)
    else:
        outname += '-cv'+str(int(cutoff))+'-'+str(fold)
    trainset, testset, valset = cnn_model.create_sets(len(X), folds, fold, Yclass = Yclass, genenames = names)
    print('Train', len(trainset))
    print('Test', len(testset))
    print('Val', len(valset))

    if '--hyperparamter_seach' in sys.argv:
        trainset = np.append(trainset, valset)
        valset = testset
        
    
    outnorm = np.ones(np.shape(Y)[-1])
    if '--norm2output' in sys.argv:
        print ('ATTENTION: output has been normalized along data points')
        outname += '-n2out'
        outnorm =np.sqrt(np.sum(Y*Y, axis = 1))[:, None] 
        Y = Y/outnorm
        outnorm = outnorm[testset]
        
    elif '--norm2outputclass' in sys.argv:
        print('ATTENTION: output has been normalized along data classess')
        outname += '-n2outc'
        outnorm =np.sqrt(np.sum(Y*Y, axis = 0))
        Y = Y/outnorm
    
    pwmoutname = ''
    params['fixed_kernels'] = None
    if '--list_of_pwms' in sys.argv:
        psam = False
        if '--psam' in sys.argv:
            psam = True
        infcont = False
        if '--infocont' in sys.argv:
            infcont = True
        pwmnameset = np.array(os.path.splitext(os.path.split(sys.argv[sys.argv.index('--list_of_pwms')+1])[1])[0].split('_'))
        pwmoutname = '_pwms'+'_'.join(pwmnameset[~np.isin(pwmnameset, outname.split('_'))]) + 'ps'+str(psam)[0]+'ic'+str(infcont)[0]
        motcut = None
        if '--motif_cutoff' in sys.argv:
            motcut = float(sys.argv[sys.argv.index('--motif_cutoff')+1])
            pwmoutname += 'mc'+str(motcut)
            params['motif_cutoff'] = motcut
        pwms, rbpnames = cnn_model.read_pwm(sys.argv[sys.argv.index('--list_of_pwms')+1])
        pwms = cnn_model.rescale_pwm(pwms, psam = psam, infcont = infcont, norm = True)
   
   
   
   
   
    # If we split on different classes, such as p_values and foldchanges but train on both, one can split the assessment between them
    if '--split_outclasses' in sys.argv:
        testclasses = np.genfromtxt(sys.argv[sys.argv.index('--split_outclasses')+1], dtype = str)
        tsort = []
        for exp in experiments:
            tsort.append(list(testclasses[:,0]).index(exp))
        testclasses = testclasses[tsort][:,1]
    else:
        testclasses = np.zeros(len(Y[0]), dtype = np.int8).astype(str)
   
    
    keepmodels = False
    if '--keep_modelparams' in sys.argv:
        keepmodels = True
   
    if '--cnn' in sys.argv:
        parameters = sys.argv[sys.argv.index('--cnn')+1].split('+')
        for p in parameters:
            if ':' in p and '=' in p:
                p = p.split('=',1)
            elif ':' in p:
                p = p.split(':',1)
            elif '=' in p:
                p = p.split('=',1)
            if p[0] == 'fixed_kernels' and p[1] != 'None':
                params[p[0]] = pwms
            else:
                params[p[0]] = cnn_model.check(p[1])
            
        params['n_features'], params['l_seqs'], params['n_classes']= np.shape(X)[-2], np.shape(X)[-1], np.shape(Y)[-1]
    
    # The hyperparamters in the list will be mixed with each other and then
    if '--hyper_parameter' in sys.argv:
        hyperparameter = {}
        n_hyper = []
        hparameters = sys.argv[sys.argv.index('--hyper_parameter')+1].split('+')
        for p in hparameters:
            if ':' in p and '=' in p:
                p = p.split('=',1)
            elif ':' in p:
                p = p.split(':',1)
            elif '=' in p:
                p = p.split('=',1)
            hvalue = cnn_model.check(p[1])
            if isinstance(hvalue, list):
                hyperparameter[p[0]] = hvalue
            else:
                hyperparameter[p[0]] = [hvalue]
            n_hyper.append(len(hyperparameter[p[0]]))
        n_hyper = np.array(n_hyper)
    elif '--hyper_parameter_list' in sys.argv:
        # iterate over len of lists (lists must have same length) and use ith combination to run model
        hyperkeys = []
        hypervalues = []
        hypercomb
    
    
    ctrlfile = open(outname + '_hyperpar-ctrl.txt', 'w')
    
    for key in params:
        ctrlfile.write(key+' : '+str(params[key])+'\n')
    
    # Train the reference model
    
    modelparams = params.copy()
    if modelparams['fixed_kernels'] is None:
        modelparams['outname'] = outname
    else:
        modelparams['outname'] = outname + pwmoutname
    
    try:
        model = cnn_model.cnn(**modelparams)
        modelname = model.outname
    except Exception as err:
        print(err, type(err))
        err = str(err)
        ctrlfile.write(err+'\n\n')
        if os.path.isfile(modelname+'_model_params.dat'):
            os.remove(modelname+'_model_params.dat')
        success = False
    else:
        success = True
    modelname = model.outname
    if success:
        print('Started')
        try:
            model.fit(X[trainset], Y[trainset], XYval = [X[valset], Y[valset]], sample_weights = weights)
        except Exception as err:
            print(err, type(err))
            err = str(err)
            ctrlfile.write(err+'\n\n')
            if os.path.isfile(modelname+'_model_params.dat'):
                os.remove(modelname+'_model_params.dat')
            if os.path.isfile(modelname+'_parameter.pth'):
                os.remove(modelname+'_parameter.pth')
            if os.path.isfile(modelname+'_params0.pth'):
                os.remove(modelname+'_params0.pth')
            success = False
        else:
            success = True
        if success:
            ctrlfile.write('Results:'+'\n'+str(model.validation_loss)+'(val) '+model.validation_loss+'(train) '+model.loss_function+'(val) '+model.loss_function+ '(train) Epochs\n'+' '.join(np.around(model.saveloss,3).astype(str))+'\n\n')
            Y_pred = model.predict(X[testset])
            save_performance(Y_pred, Y[testset], testclasses, experiments, names[testset], model.outname, sys.argv)
            #outwrite.write(Y_pred, Y[testset], names[testset], experiments, model.outname, testclasses)
    
    #model = cnn_model.cnn(**modelparams)
    #model.fit(X[trainset], Y[trainset], XYval = [X[valset], Y[valset]], sample_weights = weights)
    
    #ctrlfile.write('Results:'+'\n'+model.validation_loss+'(val) '+model.loss_function+'(val) '+model.validation_loss+'(train) '+model.loss_function+ '(train) Epochs\n'+' '.join(np.around(model.saveloss,3).astype(str))+'\n\n')
    
    #Y_pred = model.predict(X[testset])
    #outwrite.write(Y_pred, Y[testset], names[testset], experiments, model.outname, testclasses)
    
    for search_along in range(1, n_varying_parameter+1):
        hypercomb = combinations(search_along, len(n_hyper))
        gsize = 0
        for hyp in hypercomb:
            gsize += np.prod(n_hyper[hyp])
        print('Total number of models for gridsearch of size', search_along, 'is', gsize)
        hyperkeys = np.array(list(hyperparameter.keys()))
        hypervalues = list(hyperparameter.values())
        comb = 0
        ac = 0
        #clear = 0
        while True:
            hyp = hypercomb[comb]
            choskeys = hyperkeys[hyp]
            print('\nSearch along', choskeys)
            allcombos = list(itertools.product(*[hypervalues[h] for h in hyp]))
            acomb = allcombos[ac]
            modelparams = params.copy()
            modelparams['outname'] = outname
            for h in range(len(acomb)):
                modelparams[choskeys[h]] = acomb[h]
                print(choskeys[h], ':', acomb[h])
                ctrlfile.write(choskeys[h]+' : '+str(acomb[h])+'\n')   
            
            for h in range(len(acomb)):
                if choskeys[h] == 'fixed_kernels' and acomb[h] is not None:
                    modelparams[choskeys[h]] = pwms
                    modelparams['outname'] = modelparams['outname'] + pwmoutname
            try:
                model = cnn_model.cnn(**modelparams)
                modelname = model.outname
            except Exception as err:
                print(err, type(err))
                err = str(err)
                ctrlfile.write(err+'\n\n')
                if os.path.isfile(modelname+'_model_params.dat'):
                    os.remove(modelname+'_model_params.dat')
                success = False
            else:
                success = True
            modelname = model.outname
            if success:
                print('Started')
                try:
                    model.fit(X[trainset], Y[trainset], XYval = [X[valset], Y[valset]], sample_weights = weights)
                except Exception as err:
                    print(err, type(err))
                    err = str(err)
                    ctrlfile.write(err+'\n\n')
                    if os.path.isfile(modelname+'_model_params.dat'):
                        os.remove(modelname+'_model_params.dat')
                    if os.path.isfile(modelname+'_parameter.pth'):
                        os.remove(modelname+'_parameter.pth')
                    if os.path.isfile(modelname+'_params0.pth'):
                        os.remove(modelname+'_params0.pth')
                    success = False
                else:
                    success = True
                if success:
                    ctrlfile.write('Results:'+'\n'+model.validation_loss+'(val) '+model.validation_loss+'(train) '+model.loss_function+'(val) '+model.loss_function+ '(train) Epochs\n'+' '.join(np.around(model.saveloss,3).astype(str))+'\n\n')
                    Y_pred = model.predict(X[testset])
                    save_performance(Y_pred, Y[testset], testclasses, experiments, names[testset], model.outname, sys.argv)
                    #outwrite.write(Y_pred*outnorm, Y[testset]*outnorm, names[testset], experiments, model.outname, testclasses)
            
            ac += 1
            if ac == len(allcombos):
                ac = 0
                comb += 1
            if comb == len(hypercomb):
                break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



