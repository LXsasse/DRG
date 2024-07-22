
import sys, os 
import numpy as np
import torch.nn as nn
import torch

from drg_tools.io_utils import readin, check, numbertype, isfloat, create_outname, write_meme_file
from drg_tools.model_training import create_sets
from drg_tools.data_processing import manipulate_input
from drg_tools.model_output import print_averages, save_performance
from drg_tools.plotlib import plot_scatter
from drg_tools.model_output import add_params_to_outname
from drg_tools.motif_analysis import pfm2iupac
from drg_tools.interpret_cnn import takegrad, ism, deeplift, indiv_network_contribution, kernel_assessment
from drg_tools.bpcnn_model import bpcnn


if __name__ == '__main__':
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    
    # Whether to assign a different regional channel for each input file
    aregion = True
    if '--regionless' in sys.argv:
        aregion = False
    
    # Whether sequences in each region should be aligned to right side, for example if sequences have different lengths and there could be a positional signal at the end of sequences
    mirror = False
    if '--realign_input' in sys.argv:
        mirror = True
    
    # select output tracks to refine network or only train on a specific track from the beginning
    select_track = None
    if '--select_tracks' in sys.argv:
        select_track = sys.argv[sys.argv.index('--select_tracks')+1]
        select_track, select_test = read_separated(select_track)
        
    X, Y, names, features, experiments = readin(inputfile, outputfile, return_header = True, assign_region = aregion, mirrorx = mirror)

    reverse_complement = False
    if '--reverse_complement' in sys.argv:
        reverse_complement = False
         
    # filter counts with 0 entries
    if Y is not None:
        ymask = np.sum(Y,axis = (1,2)) >0
        X, Y, names = X[ymask], Y[ymask], names[ymask]

    if '--testrandom' in sys.argv:
        trand = int(sys.argv[sys.argv.index('--testrandom')+1])
        mask = np.random.permutation(len(X))[:trand]
        X, names = X[mask], names[mask]
        if Y is not None:
            Y = Y[mask]

    outname = create_outname(inputfile, outputfile) 

    
    # Parameter dictionary for initializing the cnn
    params = {}
    
    input_is_output = False
    if '--maskinput' in sys.argv:
        maxmask = int(sys.argv[sys.argv.index('--maskinput')+1])# max number of basepairs that are masked
        minmask = int(sys.argv[sys.argv.index('--maskinput')+2])# min number of basepairs taht are masked
        nmasks = int(sys.argv[sys.argv.index('--maskinput')+3])# average number of masked duplicates per seqeuence in training, validation and testing
        outname += 'mask'+str(minmask)+'-'+str(maxmask)+'n'+str(nmasks)
        allmaskloc = np.concatenate([np.array([np.arange(0,np.shape(X)[-1]-m+1),np.arange(m,np.shape(X)[-1]+1)]).T for m in range(minmask, maxmask+1)], axis = 0)
        allmasks = torch.ones(([len(allmaskloc)] + list(np.shape(X)[1:])))
        for l, loc in enumerate(allmaskloc):
            allmasks[l][...,loc[0]:loc[1]] = 0
        allmasks = allmasks == 0
        print('INPUT sequences will be masked during training with', maxmask, minmask, nmasks)
        params['masks'], params['nmasks'] = allmasks, nmasks
        # input masks can be provided to train a sequence representation by predicting masked sequence values
        # or do the same to augment training of another objective in parallel
        if '--augment_training' in sys.argv:
            augment_representation = sys.argv[sys.argv.index('--augment_training')+1] # position of embedding that will be given to the deconvolution layer for predicting masked sequences
            aug_kernel_size = int(sys.argv[sys.argv.index('--augment_training')+2]) # size of the kernels in final convolutional layer for masked sequence predictions
            aug_loss = sys.argv[sys.argv.index('--augment_training')+3] # Loss for sequence self learning
            aug_loss_mix = float(sys.argv[sys.argv.index('--augment_training')+4]) # value between 0 and 1 that mixes the objective loss with the aug_loss_mix, or if larger than 1, aug_loss is simply multiplies by this value
            params['augment_representation'], params['aug_kernel_size'], params['aug_loss'], params['aug_loss_mix'] = augment_representation, aug_kernel_size, aug_loss, aug_loss_mix
        else:
            print('INPUT == OUTPUT')
            input_is_output = True
            Y = np.copy(X)
            outname = os.path.splitext(inputfile)[0] + '_self-trainmask'+str(minmask)+'-'+str(maxmask)+'n'+str(nmasks)
        
    
    
    if '--regionless' in sys.argv:
        outname += '-rgls'
    if '--realign_input' in sys.argv:
        outname += 'mirx'
    if '--reverse_complement' in sys.argv:
        outname += 'rcomp'
    
    if '--outdir' in sys.argv:
        outname = sys.argv[sys.argv.index('--outdir')+1] + os.path.split(outname)[1]
    
    if '--addname' in sys.argv:
        outname += '_'+sys.argv[sys.argv.index('--addname')+1]
    
    if '--outname' in sys.argv:
        outname = sys.argv[sys.argv.index('--outname')+1]
    
    # reads in file that had locations and type of mutations, currently not able to handle mirror=True or multiple inputs, needs to be adjusted to common vcf file format

    if '--crossvalidation' in sys.argv:
        folds = check(sys.argv[sys.argv.index('--crossvalidation')+1])
        fold = int(sys.argv[sys.argv.index('--crossvalidation')+2])
        if '--significant_genes' in sys.argv:
            siggenes = np.genfromtxt(sys.argv[sys.argv.index('--significant_genes')+1], dtype = str)
            Yclass = np.isin(names, siggenes).astype(int)
        else:    
            cutoff = float(sys.argv[sys.argv.index('--crossvalidation')+3])
            Yclass = (np.sum(np.absolute(Y)>=cutoff, axis = 1) > 0).astype(int)
            cvs = True
    elif '--predictnew' in sys.argv:
        cvs = False
        trainset, testset, valset = [], np.arange(len(X), dtype = int), []
    else:
        folds, fold, Yclass = 10, 0, None
        cvs = True
    
    if cvs:
        if isinstance(folds,int):
            outname += '-cv'+str(folds)+'-'+str(fold)
        else:
            outname += '-cv'+str(int(cutoff))+'-'+str(fold)
        trainset, testset, valset = create_sets(len(X), folds, fold, Yclass = Yclass, genenames = names)
        print('Train', len(trainset))
        print('Test', len(testset))
        print('Val', len(valset))
    
    
    if '--maketestvalset' in sys.argv:
        testset = valset
        print('Set testset to validation set')
    
    if '--RANDOMIZE' in sys.argv:
        # randomly permute data and see if model can learn anything
        outname +='_RNDM'
        permute = np.permute(len(X))
        Y = Y[permute]

    if '--npfloat16' in sys.argv:
        scaletype = np.float16
    else:
        scaletype = np.float32
        

    if '--norm_center' in sys.argv:
        ncenter = int(sys.argv[sys.argv.index('--norm_center')+1])
        Cm = np.sum(Y[...,int((np.shape(Y)[-1]+ncenter)/2)-ncenter: int((np.shape(Y)[-1]+ncenter)/2)], axis = -1)
        Cy = np.sum(Y, axis = -1)
        Cy[Cy == 0] = 1
        C = Cm/Cy
        Y = Y.astype(scaletype)*C[...,None].astype(scaletype)
        outname += 'nc'+str(ncenter)

    if '--quantile_norm' in sys.argv:
        Cy = np.sum(Y, axis = -1)
        M = np.mean(np.sort(Cy, axis = 0), axis = 1)
        Cm = M[np.argsort(np.argsort(Cy, axis = 0), axis = 0)]
        Cy[Cy==0] = 1
        C = Cm/Cy
        Y = Y.astype(scaletype)*C[...,None].astype(scaletype)
        outname += 'qtln'
    
    if '--logcounts' in sys.argv:
        Y = np.log(1+Y)
        outname += 'lg'
    
    if '--log2counts' in sys.argv:
        Y = np.log2(2+Y)
        outname += 'lg2'
    
    if '--norm2output' in sys.argv:
        print ('ATTENTION: output has been normalized along data points')
        outname += '-n2out'
        outnorm =np.sqrt(np.sum(np.sum(Y*Y, axis = (1,2))))[:, None, None] 
        Y = Y/outnorm
        
    elif '--norm2outputclass' in sys.argv:
        print('ATTENTION: output has been normalized along data classess')
        outname += '-n2outc'
        outnorm =np.sqrt(np.sum(Y*Y, axis = (0,2)))
        Y = Y/outnorm
    
    elif '--norm1outputbins' in sys.argv:
        print('ATTENTION: output has been normalized along bins')
        outname += '-n1outb'
        outnorm =np.sum(Y, axis = 2)[...,None]
        Y = Y/outnorm
    
    if '--npfloat16' in sys.argv:
        # should round before using this, have to compute how many decimals based on largest number
        # should also warn if largest number is too big
        Y = Y.astype(np.float16)
    
    if '--npfloat32' in sys.argv:
        Y = Y.astype(np.float32)
    
    pfms = None
    pwmusage = 'none'
    if '--list_of_pwms' in sys.argv:
        list_of_pwms = sys.argv[sys.argv.index('--list_of_pwms')+1]
        pwmusage = sys.argv[sys.argv.index('--list_of_pwms')+2]
        
        psam = False
        if '--psam' in sys.argv:
            psam = True
        infcont = False
        if '--infocont' in sys.argv:
            infcont = True
        
        pwmnameset = np.array(os.path.splitext(os.path.split(list_of_pwms)[1])[0].split('_'))
        
        outname += '_pwms'+'_'.join(pwmnameset[~np.isin(pwmnameset, outname.split('_'))]) + 'ps'+str(psam)[0]+'ic'+str(infcont)[0]
        motcut = None
        
        if '--motif_cutoff' in sys.argv:
            motcut = float(sys.argv[sys.argv.index('--motif_cutoff')+1])
            outname += 'mc'+str(motcut)
        
        pfms, rbpnames = read_pwm(list_of_pwms)
        pwms = rescale_pwm(pfms, psam = psam, infcont = infcont, norm = True)
        if pwmusage != 'initialize': 
            params['fixed_kernels'] = pwms
            params['motif_cutoff'] = motcut
        # elif initialize: pwms are trimmmed to the kernel length and used as initialization for the kernels

    
    
    if '--cnn' in sys.argv:
        parameterfile = False
        train_model = True
        parameters = sys.argv[sys.argv.index('--cnn')+1]
        params['device'] = get_device()
        if '+' in parameters:
            parameters = parameters.split('+')
            params['n_classes'] = np.shape(Y)[-2]
            params['l_out'] = np.shape(Y)[-1]
        
        elif os.path.isfile(parameters):
            parameterfile = parameters.replace('model_params.dat', 'parameter.pth')
            obj = open(parameters,'r').readlines()
            parameters = []
            for l, line in enumerate(obj):
                if line[0] != '_' and line[:7] != 'outname':
                    parameters.append(line.strip().replace(' ', ''))
                    print(line.strip().replace(' ', ''))
                if line[:7] == 'outname':
                    outname += 'from'+os.path.split(line.strip().split(' : ')[1])[1]
                    #outname += 'from'+os.path.split(parameterfile.replace('_parameter.pth', ''))[1]
            params['add_outname'] = False
            params['generate_paramfile'] = False
            train_model = False
            if ':' in sys.argv[sys.argv.index('--cnn')+2] or '=' in sys.argv[sys.argv.index('--cnn')+2]:
                if '+' in sys.argv[sys.argv.index('--cnn')+2]:
                    for a in sys.argv[sys.argv.index('--cnn')+2].split('+'):
                        print(a)
                        parameter.append(a)
                else:
                    parameters.append(sys.argv[sys.argv.index('--cnn')+2])
        else:
            parameters = [parameters]
            params['n_classes'] = np.shape(Y)[-2]
            params['l_out'] = np.shape(Y)[-1]
        
        for p in parameters:
            if ':' in p and '=' in p:
                p = p.split('=',1)
            elif ':' in p:
                p = p.split(':',1)
            elif '=' in p:
                p = p.split('=',1)
            params[p[0]] = check(p[1])
        params['outname'] = outname
        print('Device', params['device'])
        params['n_features'], params['l_seqs'], params['reverse_complement'] = np.shape(X)[-2], np.shape(X)[-1], reverse_complement
        model = bpcnn(**params)
       
    weights = None
    if '--sample_weights' in sys.argv:
        weights = np.genfromtxt(sys.argv[sys.argv.index('--sample_weights')+1], dtype = str)
        weights = weights[np.sort(weights[:,0])]
        weights = weights[np.isin(weights[:,0], names), 1].astype(float)
        weights = weights[trainset]
            
    
    translate_dict = None
    exclude_dict = []
    allow_reduction = False
    include = False
    if '--load_parameters' in sys.argv:
        parameterfile = sys.argv[sys.argv.index('--load_parameters')+1]
        translate_params = sys.argv[sys.argv.index('--load_parameters')+2] # dictionary with names from current and alternatie models
        exclude_params = sys.argv[sys.argv.index('--load_parameters')+3] # list with names from loaded model that should be ignored when loading
        include = sys.argv[sys.argv.index('--load_parameters')+4] == 'True' # if include then exclude list is list of parameters that will be included
        allow_reduction = sys.argv[sys.argv.index('--load_parameters')+5] == 'True' # if include then exclude list is list of parameters that will be included
        model.outname += sys.argv[sys.argv.index('--load_parameters')+6]
        if ":" in translate_params:
            if ',' in tranlate_params:
                translate_params = translate_params.split(',')
            else:
                translate_params = [translate_params]
            translate_dict = {}
            for tp in tranlate_params:
                translate_dict[tp.split(":")[0]] = tp.split(":")[1]
        
        if '.' in exclude_params:
            if ',' in exclude_params:
                exclude_dict = exclude_params.split(',')
            else:
                exclude_dict = [exclude_params]
        
        
        
    # parameterfile is loaded into the model
    if parameterfile:
        load_parameters(model, parameterfile, translate_dict = translate_dict, exclude = exclude_dict, include = include, allow_reduction = allow_reduction)
        if model.generate_paramfile:
            obj = open(model.outname+'_model_params.dat', 'a')
            obj.write('param_file : '+ os.path.split(parameterfile)[1] +'\n')
            obj.close()
    
    if select_track is not None:
        if select_track[0] in experiments:
            select_track = [list(experiments).index(st) for st in select_track]
                
        model.classifier.Linear.weight = nn.Parameter(model.classifier.Linear.weight[select_track])
        model.classifier.Linear.bias = nn.Parameter(model.classifier.Linear.bias[select_track])
        model.n_classes = len(select_track)
        Y = Y[:,select_track]
        experiments = experiments[select_track]
        print(len(select_track), '('+','.join(experiments)+')', 'tracks selected')
        
    # pre-loaded pwms are loaded into the model as convolutional weights
    if pwmusage == 'initialize':
        init_kernels = [pwmset(pwm.T, params['l_kernels'], shift_long = False) for pwm in pwms]
        init_kernels = np.concatenate(init_kernels, axis = 0)
        print('INIT kernels', len(init_kernels))
        if model.generate_paramfile:
            obj = open(model.outname+'_model_params.dat', 'a')
            obj.write('param_file : '+ os.path.split(list_of_pwms)[1] +'\n')
            obj.close()
        init_kernels = OrderedDict({ 'convolutions.weight': torch.Tensor(init_kernels[np.argsort(-np.sum(init_kernels, axis = (1,2)))])})
        load_parameters(model, init_kernels, allow_reduction = True)
    
    
    if train_model:
        model.fit(X[trainset], Y[trainset], XYval = [X[valset], Y[valset]], sample_weights = weights)
    
    
    if input_is_output and nmasks is not None:
        ntsus = np.array(list('ACGT'))
        countmat = np.zeros(np.shape(Y[testset]), dtype = np.int16)
        Y_predmasked = np.zeros(np.shape(Y[testset]))
        Y_predunmasked = np.zeros(np.shape(Y[testset]))
        for nma in range(nmasks):
            cmasks = allmasks[np.random.choice(len(allmasks), len(testset))]
            Xmin = np.copy(X[testset])
            Xmin[cmasks] = 0.25
            countmat[cmasks] += 1
            Yi_pred = model.predict(Xmin)
            Y_predmasked[cmasks] += Yi_pred[cmasks]
            Y_predunmasked[~cmasks] += Yi_pred[~cmasks]
        Y_pred = Y_predmasked/countmat
        Y_pred[countmat == 0] = Y_predunmasked[countmat == 0]/nmasks
        Y_pred = np.nan_to_num(Y_pred)
        #print(''.join(ntsus[np.argmax(Y_pred[0], axis = 0)]))
        #print(''.join(ntsus[np.argmax(Y[testset][0], axis = 0)]))
    else:
        Y_pred = model.predict(X[testset])
    
    if Y is not None:
        Ytest = Y[testset]
    if '--npfloat16' in sys.argv:
        Ytest = Ytest.astype(np.float32)
    
    
    if '--norm2output' in sys.argv:
        Y *= outnorm
        Y_pred *= outnorm[testset]
        
    elif '--norm2outputclass' in sys.argv:
        Y *= outnorm
        Y_pred *= outnorm
    
    
    
    if model.outname is not None and model.outname != outname:
        outname = model.outname
    print(outname)
    
    
    if Y is not None:
            
        # If we split on different classes, such as p_values and foldchanges but train on both, one can split the assessment between them
        if '--split_outclasses' in sys.argv:
            testclasses = np.genfromtxt(sys.argv[sys.argv.index('--split_outclasses')+1], dtype = str)
            tsort = []
            for exp in experiments:
                tsort.append(list(testclasses[:,0]).index(exp))
            testclasses = testclasses[tsort][:,1]
        else:
            testclasses = np.zeros(len(Y[0]), dtype = np.int8).astype(str)
            
        # Sometimes we're not interested in the reconstruction for all genes in the training set and we can define a set of genes that we're most interested in
        if '--significant_genes' in sys.argv:
            siggenes = np.genfromtxt(sys.argv[sys.argv.index('--significant_genes')+1], dtype = str)
            tlen = len(testset)
            tsetmask = np.isin(names[testset], siggenes)
            testset = testset[tsetmask]
            Y_pred = Y_pred[tsetmask]
            print('Testset length reduced to significant genes from', tlen, 'to', len(testset))

        # USE: --aurocaverage --auprcaverage --mseaverage --correlationaverage
        print_averages(Y_pred, Ytest, testclasses, sys.argv)
        
        # USE: --save_correlation_perclass --save_auroc_perclass --save_auprc_perclass --save_mse_perclass --save_correlation_pergene '--save_mse_pergene --save_auroc_pergene --save_auprc_pergene --save_topdowncorrelation_perclass
        save_performance(Y_pred, Ytest, testclasses, experiments, names[testset], outname, sys.argv, compare_random = True)
        
    if '--save_predictions' in sys.argv:
        print('SAVED', outname+'_pred.npz')
        np.savez_compressed(outname+'_pred.npz', names = names[testset], counts = Y_pred)
    
    if '--save_embeddings' in sys.argv:
        emb_loc = sys.argv[sys.argv.index('--save_embeddings')+1]
        Y_emb = []
        for b in range(max(1,int(len(testset)/model.batchsize) + int(len(testset)%model.batchsize > 0))):
            yemb = model.forward(torch.Tensor(X[testset[b*model.batchsize:(b+1)*model.batchsize]]).to(model.device), location = emb_loc)
            Y_emb.append(yemb.detach().cpu().numpy())
        Y_emb = np.concatenate(Y_emb, axis = 0)
        np.savez_compressed(outname+'_embedding'+emb_loc+'.npz', names = names[testset], counts = Y_emb)
        print('SAVED', outname+'_embedding'+emb_loc+'.npz', 'with shape', np.shape(Y_emb))
    
    if '--save_trainpredictions' in sys.argv:
        print('SAVED training', outname+'_trainpred.txt')
        Y_predtrain = model.predict(X[trainset])
        np.savez_compressed(outname+'_trainpred.npz', names = names[trainset], counts = Y_predtrain)
    
    
    # Generate PPMs for kernels
    if '--convertedkernel_ppms' in sys.argv:
        lpwms = 4
        if len(sys.argv) > sys.argv.index('--convertedkernel_ppms')+1:
            if '--' not in sys.argv[sys.argv.index('--convertedkernel_ppms')+1]:
                activation_measure = sys.argv[sys.argv.index('--convertedkernel_ppms')+1]
            else:
                activation_measure = 'euclidean'
        else:
            activation_measure = 'euclidean'
        ppms = model.convolutions.weight.detach().cpu().numpy()
        
        motifnames = np.array(['filter'+str(i) for i in range(len(ppms))])
        
        # maybe don't add  biasses? But makes mathematically sense
        biases = None
        if model.kernel_bias:
            biases = model.convolutions.bias.detach().cpu().numpy()
        
        if np.shape(ppms)[1] > lpwms:
            ppmscontext = ppms[:,lpwms:,:]
            ppmscontext = kernel_to_ppm(ppmscontext, kernel_bias = biases)
            write_meme_file(ppmscontext, motifnames, ''.join(np.array([f[0] for f in features[lpwms:]])), outname+'_kernelcontext_ppms.meme')
        
        ppms = kernel_to_ppm(ppms[:,:lpwms,:], kernel_bias = biases)
        
        iupacmotifs = pfm2iupac(ppms, bk_freq = 0.3)
        pwm_in = None
        if pfms is not None:
            iupacmotifs = np.append(iupacmotifs, pfm2iupac(pfms, bk_freq = 0.3))
            motifnames = np.append(motifnames, rbpnames)
            pwm_in = pwm_scan(X[testset], pwms, targetlen = model.l_kernels, activation = 'max', motif_cutoff = model.motif_cutoff, set_to = 0., verbose = False)
            ppms = list(ppms) + list(pfms)
        
        write_meme_file(ppms, motifnames, ''.join(np.array([f[0] for f in features[:lpwms]])), outname+'_kernel_ppms.meme')
        
        #### enable this to generate values per cell type and per gene when different activation_measure given
        
        motifimpact, impact_direction = compute_importance(model, X[trainset], Y[trainset], activation_measure = activation_measure, pwm_in = pwm_in, normalize = False)
        
        np.savetxt(outname+'_kernel_importance.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), motifimpact], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(experiments))
        np.savetxt(outname+'_kernel_impact.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), impact_direction], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(experiments))
        
        # Due to early stopping the importance scores from training and testing should be the same
        # However training importance provide impacts of patterns from overfitting
        # Test set on the other hand could lack the statistical power
        if '--testset_importance' in sys.argv:
            motifimpact, impact_direction = compute_importance(model, X[testset], Ytest, activation_measure = activation_measure, pwm_in = model.pwm_out, normalize = False)
            np.savetxt(outname+'_kernel_importance_test.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), motifimpact], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(experiments))
            np.savetxt(outname+'_kernel_impact_test.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), impact_direction], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(experiments))
            
    

    
    
    








