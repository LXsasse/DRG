import sys, os 
import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr, cosine
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
from collections import OrderedDict
from sklearn.linear_model import LinearRegression, Lasso
from torch import Tensor
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
from functools import reduce
from torch_regression import torch_Regression
from data_processing import readin, read_mutationfile, create_sets, create_outname, rescale_pwm, read_pwm, check, numbertype, isfloat, manipulate_input
from functions import mse, correlation, dist_measures
from output import print_averages, save_performance, plot_scatter, add_params_to_outname
from functions import dist_measures
from interpret_cnn import write_meme_file, pfm2iupac, kernel_to_ppm, compute_importance 
from init import get_device, MyDataset, kmer_from_pwm, pwm_from_kmer, kmer_count, kernel_hotstart, load_parameters
from modules import parallel_module, gap_conv, interaction_module, pooling_layer, correlation_loss, correlation_both, cosine_loss, cosine_both, zero_loss, Complex, Expanding_linear, Res_FullyConnect, Residual_convolution, Res_Conv1d, MyAttention_layer, Kernel_linear, loss_dict, func_dict, simple_multi_output
from train import pwmset, pwm_scan, batched_predict
from train import fit_model
from compare_expression_distribution import read_separated
from cnn_model import cnn

def separate_sys(sysin, delimiter = ',', delimiter_1 = None):
    if delimiter in sysin:
        sysin = sysin.split(delimiter)
    else:
        sysin = [sysin]
    if delimiter_1 is not None:
        for s, sin in enumerate(sysin):
            if delimiter_1 in sin:
                sysin[s] = sin.split(delimiter_1)
            else:
                sysin[s] = [sin]
    return sysin
        
    
def genseq(lseq, nseq):
    seqs = np.zeros((nseq,4,lseq))
    pos = np.random.randint(0,4,lseq*nseq)
    pos0 = (np.arange(lseq*nseq,dtype=int)/lseq).astype(int)
    pos1 = np.arange(lseq*nseq,dtype=int)%lseq
    seqs[pos0,pos,pos1] = 1
    return seqs

class combined_network(nn.Module):
    def __init__(self, loss_function = 'MSE', validation_loss = None, n_features = None, n_classes = 1, l_seqs = None, reverse_complement = None, cnn_embedding = 516, n_combine_layers = 3, combine_function = 'GELU', combine_widening = 1.1, combine_residual = 0, shift_sequence = None, random_shift = False, smooth_onehot = 0, reverse_sign = False, dropout = 0, batch_norm = False, epochs = 1000, lr = 1e-2, kernel_lr = None, cnn_lr = None, batchsize = None, patience = 50, outclass = 'Linear', outlog = 2, outlogoffset = 2, outname = None, optimizer = 'Adam', optim_params = None, verbose = True, checkval = True, init_epochs = 3, writeloss = True, write_steps = 1, device = 'cpu', load_previous = True, init_adjust = True, seed = 101010, keepmodel = False, generate_paramfile = True, add_outname = True, restart = False, **kwargs):
        super(combined_network, self).__init__()
        
        torch.manual_seed(seed)
        
        self.loss_function = loss_function
        self.validation_loss = validation_loss
        self.n_features = n_features
        self.n_classes = n_classes
        self.l_seqs = l_seqs
        self.n_inputs = len(l_seqs)
        self.cnn_embedding = cnn_embedding
        self.n_combine_layers = n_combine_layers
        self.combine_function = combine_function
        self.combine_widening = combine_widening
        self.combine_residual = combine_residual
        self.shift_sequence = shift_sequence
        self.reverse_sign = reverse_sign
        self.smooth_onehot = smooth_onehot
        self.epochs = epochs 
        self.lr = lr
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.kernel_lr = kernel_lr
        self.cnn_lr = cnn_lr
        self.batchsize = batchsize 
        self.patience = patience
        self.outclass = outclass
        self.outlog = outlog
        self.outlogoffset = outlogoffset
        self.outname = outname
        self.optimizer = optimizer
        self.optim_params = optim_params
        self.verbose = verbose
        self.checkval = checkval
        self.init_epochs = init_epochs
        self.writeloss = writeloss
        self.write_steps = write_steps
        self.device = device 
        self.load_previous = load_previous
        self.init_adjust = init_adjust
        self.seed = seed
        self.keepmodel = keepmodel
        self.generate_paramfile = generate_paramfile
        self.add_outname = add_outname
        self.restart = restart
        self.random_shift = random_shift
        
        if reverse_complement is None:
            reverse_complement = [False for l in range(len(l_seqs))]
        self.__dict__['reverse_complement'] = any(reverse_complement)
        
        for kw in kwargs:
            self.__dict__[str(kw)] = kwargs[kw]
        refresh_dict = cnn(n_features = n_features, n_classes = cnn_embedding, l_seqs = l_seqs[0], seed = self.seed, dropout = dropout, batch_norm = batch_norm, add_outname = False, generate_paramfile = False, keepmodel = False, verbose = False, **kwargs).__dict__
        for rw in refresh_dict:
            if rw not in self.__dict__.keys():
                self.__dict__[str(rw)] = refresh_dict[rw]
        
        self.use_nnout = True
        if (('FRACTION' in outclass.upper()) or ('DIFFERENCE' in outclass.upper())) and len(l_seqs) == 2:
            self.cnn_embedding = cnn_embedding = n_classes
            self.n_combine_layers = 0
            self.use_nnout = False
            
        
        if add_outname:
            ### Generate file name from all settings
            if outname is None:
                self.outname = 'CNNmodel'  # add all the other parameters
            else:
                self.outname = outname
            
            self.outname = add_params_to_outname(self.outname, self.__dict__)
            if self.verbose:
                print('ALL file names', self.outname)
    
        if generate_paramfile:
            obj = open(self.outname+'_model_params.dat', 'w')
            for key in self.__dict__:
                if str(key) == 'fixed_kernels' and self.__dict__[key] is not None:
                    obj.write(key+' : '+str(len(self.__dict__[key]))+'\n')
                else:
                    obj.write(key+' : '+str(self.__dict__[key])+'\n')
            obj.close()
        self.generate_paramfile = generate_paramfile
        
        
        self.cnns = nn.ModuleDict()
        currdim = 0
        for l, lseq in enumerate(l_seqs):
            self.cnns['CNN'+str(l)] = cnn(n_features = n_features, n_classes = cnn_embedding, l_seqs = lseq, seed = self.seed +l, dropout = dropout, batch_norm = batch_norm, add_outname = False, generate_paramfile = False, keepmodel = False, verbose = verbose, outclass = combine_function, reverse_complement = reverse_complement[l], shift_sequence = shift_sequence, **kwargs)
            currdim += cnn_embedding
        
        # difference can be used to fit log(expression) = log(kt/kd) = log(kt) - log(kd) or expression = kt/kd 
        # However, DO NOT use difference if values are log(a+expression) since they cannot be represented as difference
        if self.use_nnout == False:
            print('MAKE SURE that processing of the log is identical to what is known about the data.\nCurrently:',outclass, outlogoffset, outlog)
            prefunc = None
            if 'FRACTION' in outclass.upper():
                prefunc = 'ReLU'
            self.classifier = simple_multi_output(out_relation = outclass, pre_function = 'ReLU', log = outlog, logoffset = outlogoffset)
        
        else:
            if self.n_combine_layers > 0:
                self.nclayers = Res_FullyConnect(currdim, outdim = currdim, n_classes = None, n_layers = self.n_combine_layers, layer_widening = combine_widening, batch_norm = self.batch_norm, dropout = self.dropout, activation_function = combine_function, residual_after = combine_residual, bias = True)
            
            classifier = OrderedDict()
            classifier['Linear'] = nn.Linear(currdim, n_classes)
            
            if self.outclass == 'Class':
                classifier['Sigmoid'] = nn.Sigmoid()
            elif self.outclass == 'Multi_class':
                classifier['Softmax'] = nn.Softmax()
            elif self.outclass == 'Complex':
                classifier['Complex'] = Complex(n_classes)
            
            self.classifier = nn.Sequential(classifier)
        
            
            self.kwargs = kwargs
            # set learning_rate reduce or increase learning rate for kernels by hand
            if self.kernel_lr is None:
                self.kernel_lr = lr
        
    def forward(self, x, mask = None, mask_value = 0, location = 'None', **kwargs):
        # Forward pass through all the initialized layers
        pred = []
        for c, cn in enumerate(self.cnns.values()):
            if mask is None:
                pred.append(cn(x[c]))
            elif isinstance(mask, int):
                pred.append(mask_value.unsqueeze(0).repeat(x[c].size(dim = 0),1))
            elif len(mask) == 2:
                if mask[0] == c:
                    pred.append(cn(x[c], mask = mask[1], mask_value = mask_value))
                else:
                    pred.append(cn(x[c]))
        
        # return representation of single CNNx
        if location == '0':
            return pred[kwargs['cnn']]

        # return concatenated representations from all CNNs
        if location == '1' or (self.n_combine_layers == 0 and location == '-1'):
            return torch.cat(pred, dim = -1)
        
        if self.use_nnout:
            pred = torch.cat(pred, dim = -1)
        
        if self.n_combine_layers > 0:
            pred = self.nclayers(pred)
            if location == '-1' or location == '2':
                return pred
        
        pred = self.classifier(pred)
        
        return pred
    
    def predict(self, X, mask = None, mask_value = 0, device = None):
        if device is None:
            device = self.device
        predout = batched_predict(self, X, mask = mask, mask_value = mask_value, device = device, batchsize = self.batchsize, shift_sequence = self.shift_sequence, random_shift = self.random_shift)
        return predout

    def fit(self, X, Y, XYval = None, sample_weights = None):
        self.saveloss = fit_model(self, X, Y, XYval = XYval, sample_weights = sample_weights, loss_function = self.loss_function, validation_loss = self.validation_loss, batchsize = self.batchsize, device = self.device, optimizer = self.optimizer, optim_params = self.optim_params, verbose = self.verbose, lr = self.lr, kernel_lr = self.kernel_lr, hot_start = self.hot_start, warm_start = self.warm_start, outname = self.outname, adjust_lr = self.adjust_lr, patience = self.patience, init_adjust = self.init_adjust, keepmodel = self.keepmodel, load_previous = self.load_previous, write_steps = self.write_steps, checkval = self.checkval, writeloss = self.writeloss, init_epochs = self.init_epochs, epochs = self.epochs, l1reg_last = self.l1reg_last, l2_reg_last = self.l2reg_last, l1_kernel = self.l1_kernel, reverse_sign = self.reverse_sign, shift_back = self.shift_sequence, random_shift = self.random_shift, smooth_onehot = self.smooth_onehot, restart = self.restart, multiple_input = True, **self.kwargs)
    






if __name__ == '__main__':
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    
    # Delimiter for values in output file
    delimiter = ','
    if '--delimiter' in sys.argv:
        delimiter = sys.argv[sys.argv.index('--delimiter')+1]
    
    
    # Whether sequences in each region should be aligned to right side, for example if sequences have different lengths and there could be a positional signal at the end of sequences
    mirror = False
    if '--realign_input' in sys.argv:
        mirror = True
    
    combinput = False
    
    # select output tracks to refine network or only train on a specific track from the beginning
    select_track = None
    if '--select_tracks' in sys.argv:
        select_track = sys.argv[sys.argv.index('--select_tracks')+1]
        select_track, select_test = read_separated(select_track)
        
    X, Y, names, features, experiments = readin(inputfile, outputfile, delimiter = delimiter,return_header = True, assign_region = False, mirrorx = mirror, combinex = combinput)
    
    if '--testrandom' in sys.argv:
        trand = int(sys.argv[sys.argv.index('--testrandom')+1])
        mask = np.random.permutation(len(Y))[:trand]
        X, names = [x[mask] for x in X], names[mask]
        if Y is not None:
            Y = Y[mask]
    
    
    if '--remove_allzero' in sys.argv and Y is not None:
        mask = np.sum(Y, axis = 1) != 0
        Y = Y[mask]
        X, names = [x[mask] for x in X], names[mask]
    
    if '--remove_allzerovar' in sys.argv and Y is not None:
        mask = np.std(Y, axis = 1) != 0
        Y = Y[mask]
        X, names = [x[mask] for x in X], names[mask]
    

    if '--adjust_allzerovar' in sys.argv and Y is not None:
        mask = np.std(Y, axis = 1) == 0
        rand = np.random.normal(loc = 0, scale = 1e-4, size = np.shape(Y[mask]))
        Y[mask] += rand
        
    # make X with 8 rows for ACGTACGT to capture the reverse complement\
    # Replace with Convolutional module that generates reverse complement during operation
    reverse_complement = [False for i in range(len(X))]
    if '--reverse_complement' in sys.argv:
        reverse_complement = np.array(sys.argv[sys.argv.index('--reverse_complement')+1].split(',')) == 'True'
        if 'True' not in sys.argv[sys.argv.index('--reverse_complement')+1]:
            print('Define which cnn uses the reverse complement with True,False,...')
            sys.exit()
    
    if ',' in inputfile:
        inputfiles = inputfile.split(',')
        inputfile = inputfiles[0]
        for inp in inputfiles[1:]:
            inputfile = create_outname(inp, inputfile, lword = 'and')
            
    outname = create_outname(inputfile, outputfile) 
    if '--regionless' in sys.argv:
        outname += '-rgls'
    if '--realign_input' in sys.argv:
        outname += 'mirx'
    if '--reverse_complement' in sys.argv:
        outname += 'rcomp'
    # Parameter dictionary for initializing the cnn
    params = {}
    
    if '--outdir' in sys.argv:
        outname = sys.argv[sys.argv.index('--outdir')+1] + os.path.split(outname)[1]
    
    if '--addname' in sys.argv:
        outname += '_'+sys.argv[sys.argv.index('--addname')+1]
    
    if '--outname' in sys.argv:
        outname = sys.argv[sys.argv.index('--outname')+1]
    
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
        trainset, testset, valset = [], np.arange(len(Y), dtype = int), []
    else:
        folds, fold, Yclass = 10, 0, None
        cvs = True
    
    if cvs:
        if isinstance(folds,int):
            outname += '-cv'+str(folds)+'-'+str(fold)
        else:
            outname += '-cv'+str(int(cutoff))+'-'+str(fold)
        trainset, testset, valset = create_sets(len(Y), folds, fold, Yclass = Yclass, genenames = names)
        print('Train', len(trainset))
        print('Test', len(testset))
        print('Val', len(valset))
    
    if '--RANDOMIZE' in sys.argv:
        # randomly permute data and see if model can learn anything
        outname +='_RNDM'
        permute = np.permute(len(X))
        Y = Y[permute]
        
    
    if '--norm2output' in sys.argv:
        print ('ATTENTION: output has been normalized along data points')
        outname += '-n2out'
        outnorm =np.sqrt(np.sum(Y*Y, axis = 1))[:, None] 
        Y = Y/outnorm
        
    elif '--norm2outputclass' in sys.argv:
        print('ATTENTION: output has been normalized along data classess')
        outname += '-n2outc'
        outnorm =np.sqrt(np.sum(Y*Y, axis = 0))
        Y = Y/outnorm
    
    pwms = None
    if '--list_of_pwms' in sys.argv:
        list_of_pwms = sys.argv[sys.argv.index('--list_of_pwms')+1] # files with pwms
        pwmusage = sys.argv[sys.argv.index('--list_of_pwms')+2] # names of model assignments
        
        pwmusage = separate_sys(pwmusage)
        list_of_pwms = separate_sys(list_of_pwms)
        
        psam = False
        if '--psam' in sys.argv:
            psam = True
        infcont = False
        if '--infocont' in sys.argv:
            infcont = True
        
        pwmnameset = list_of_pwms[0]
        for inp in list_of_pwms[1:]:
            pwmnameset = create_outname(inp, pwmnameset, lword = 'and')
            
        
        outname += pwmnameset+ 'ps'+str(psam)[0]+'ic'+str(infcont)[0]
        
        pwms, rbpnames = [], []
        
        for pwmsli in list_of_pwms:
            pfms_, rbpnames_ = read_pwm(pwmsli)
            pwms_ = rescale_pwm(pfms_, psam = psam, infcont = infcont, norm = True)
            pwms.append(pwms_)
            rbpnames.append(rbpnames_)
            
        
    if '--cnn' in sys.argv:
        parameterfile = False
        train_model = True
        parameters = sys.argv[sys.argv.index('--cnn')+1]
        params['device'] = get_device()
        if '+' in parameters:
            parameters = parameters.split('+')
            params['n_classes'] = np.shape(Y)[-1]
        
        elif os.path.isfile(parameters):
            parameterfile = [parameters.replace('model_params.dat', 'parameter.pth')]
            obj = open(parameters,'r').readlines()
            parameters = []
            for l, line in enumerate(obj):
                if line[0] != '_' and line[:7] != 'outname':
                    parameters.append(line.strip().replace(' ', ''))
                if line[:7] == 'outname':
                    outname += os.path.split(line.strip().split(' : ')[1])[1]
            parameters.append('add_outname=False')
            parameters.append('generate_paramfile=False')
            train_model = False
            if ':' in sys.argv[sys.argv.index('--cnn')+2] or '=' in sys.argv[sys.argv.index('--cnn')+2]:
                if '+' in sys.argv[sys.argv.index('--cnn')+2]:
                    for a in sys.argv[sys.argv.index('--cnn')+2].split('+'):
                        parameters.append(a)
                else:
                    parameters.append(sys.argv[sys.argv.index('--cnn')+2])
        else:
            parameters = [parameters]
            params['n_classes'] = np.shape(Y)[-1]
        
        params['outname'] = outname
        for p in parameters:
            if ':' in p and '=' in p:
                p = p.split('=',1)
            elif ':' in p:
                p = p.split(':',1)
            elif '=' in p:
                p = p.split('=',1)
            params[p[0]] = check(p[1])
        
        print('Device', params['device'])
        params['n_features'], params['l_seqs'], params['reverse_complement'] = np.shape(X[0])[-2], [np.shape(x)[-1] for x in X], reverse_complement
        model = combined_network(**params)
    
    translate_dict = None
    exclude_dict = None
    allow_reduction = False
    include = False
    if '--load_parameters' in sys.argv:
        parameterfile = separate_sys(sys.argv[sys.argv.index('--load_parameters')+1])
        translate_params = sys.argv[sys.argv.index('--load_parameters')+2] # dictionary with names from current and alternatie models
        exclude_params = sys.argv[sys.argv.index('--load_parameters')+3] # list with names from loaded model that should be ignored when loading
        include = sys.argv[sys.argv.index('--load_parameters')+4] == 'True' # if include then exclude list is list of parameters that will be included
        allow_reduction = sys.argv[sys.argv.index('--load_parameters')+5] == 'True' # if include then exclude list is list of parameters that will be included
        outname += sys.argv[sys.argv.index('--load_parameters')+6]
        if ":" in translate_params:
            translate_params = separate_sys(translate_params, delimiter_1 = '-')
            translate_dict = [{} for t in range(len(translate_params))]
            for t, tp in enumerate(translate_params):
                for tle in tp:
                    translate_dict[t][tle.split(":")[0]] = tle.split(":")[1]
        if '.' in exclude_params:
            exclude_dict = separate_sys(exclude_params)
        
            
    # parameterfile is loaded into the model
    if parameterfile:
        if translate_dict is None:
            translate_dict = [None for i in range(len(parameterfile))]
        if exclude_dict is None:
            exclude_dict = [[] for i in range(len(parameterfile))]
            
        writeparm = ''
        for p, paramfile in enumerate(parameterfile):
            if os.path.isfile(paramfile):
                load_parameters(model, paramfile, translate_dict = translate_dict[p], exclude = exclude_dict[p], include = include, allow_reduction = allow_reduction)
            writeparm += os.path.splitext(paramfile)[1]+','
        if model.generate_paramfile:
            obj = open(model.outname+'_model_params.dat', 'a')
            obj.write('param_file : '+ writeparm +'\n')
            obj.close()
    
    if select_track is not None:
        if select_track[0] in list(experiments):
            select_track = [list(experiments).index(st) for st in select_track]
        model.classifier.Linear.weight = nn.Parameter(model.classifier.Linear.weight[select_track])
        model.classifier.Linear.bias = nn.Parameter(model.classifier.Linear.bias[select_track])
        model.n_classes = len(select_track)
        Y = Y[:,select_track]
        experiments = experiments[select_track]
        print(len(select_track), '('+','.join(experiments)+')', 'tracks selected')
        
    # pre-loaded pwms are loaded into the model as convolutional weights
    if pwms is not None:
        init_kernels = OrderedDict()
        for p, pwms_ in enumerate(pwms):
            init_k_ = np.concatenate([pwmset(pwm.T, params['l_kernels'], shift_long = False) for pwm in pwms_], axis = 0)
            init_kernels[pwmusage[p]] = torch.Tensor(init_k_[np.argsort(-np.sum(init_k_, axis = (1,2)))])
            print('INIT kernels', pwmusage[p], len(init_kernels[pwmusage[p]]))
        load_parameters(model, init_kernels, allow_reduction = True)
    
    
    if train_model:
        model.fit([x[trainset] for x in X], Y[trainset], XYval = [[x[valset] for x in X], Y[valset]])
    
    Y_pred = model.predict([x[testset] for x in X])
    
    if '--norm2output' in sys.argv:
        Y *= outnorm
        Y_pred *= outnorm[testset]
        
    elif '--norm2outputclass' in sys.argv:
        Y *= outnorm
        Y_pred *= outnorm
    
    if model.outname is not None and model.outname != outname:
        outname = model.outname
    
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
        print_averages(Y_pred, Y[testset], testclasses, sys.argv)
        
        # USE: --save_correlation_perclass --save_auroc_perclass --save_auprc_perclass --save_mse_perclass --save_correlation_pergene '--save_mse_pergene --save_auroc_pergene --save_auprc_pergene --save_topdowncorrelation_perclass
        save_performance(Y_pred, Y[testset], testclasses, experiments, names[testset], outname, sys.argv, compare_random = True)
        
    if '--save_predictions' in sys.argv:
        print('SAVED', outname+'_pred.txt')
        np.savetxt(outname+'_pred.txt', np.append(names[testset][:, None], Y_pred, axis = 1), header = ' '.join(experiments), fmt = '%s')
    
    if '--test_individual_network' in sys.argv:
        randomseqs = trainset[np.random.permutation(len(trainset))][:2500]
        genecont = []
        header= ['MSE', 'Corr']
        Ypredtrain = model.predict([x[trainset] for x in X])
        trainmse = mse(Ypredtrain,Y[trainset] ,axis =1)
        traincorr = correlation(Ypredtrain,Y[trainset] ,axis =1)
        for i in range(len(X)):
            header.append('CNN'+str(i)+'MSE')
            header.append('CNN'+str(i)+'Corr')
            mean_rep = []
            for r in range(0,len(randomseqs), 100):
                rand = randomseqs[r:r+100]
                mean_rep.append(model.forward([torch.Tensor(x[rand]) for x in X], location = '0', cnn = i).detach().cpu().numpy())
            mean_rep = np.mean(np.concatenate(mean_rep, axis = 0),axis = 0)
            Ymask = model.predict([x[trainset] for x in X], mask = i, mask_value = torch.Tensor(mean_rep))
            MSEdif = mse(Ymask,Y[trainset],axis = 1) - trainmse
            genecont.append(MSEdif)
            Corrdif = correlation(Ymask,Y[trainset],axis = 1) - traincorr
            genecont.append(Corrdif)
        
        np.savetxt(outname+'_netimportance.dat', np.concatenate([[names[trainset]],np.around([trainmse,traincorr],2), np.around(np.array(genecont),3)],axis = 0).T, header = 'Gene '+' '.join(np.array(header)), fmt = '%s')
            
            
            
    # Generate PPMs for kernels
    if '--convertedkernel_ppms' in sys.argv:
        if len(sys.argv) > sys.argv.index('--convertedkernel_ppms')+1:
            if '--' not in sys.argv[sys.argv.index('--convertedkernel_ppms')+1]:
                activation_measure = sys.argv[sys.argv.index('--convertedkernel_ppms')+1]
            else:
                activation_measure = 'euclidean'
        else:
            activation_measure = 'euclidean'
        
        ppms = []
        biases = []
        motifnames = []
        motifmeans = []
        importance = []
        effect = []
        seq_seq = genseq(model.l_kernels, 10000) # generate 10000 random sequences
        i =0
        Ypredtrain = model.predict([x[trainset] for x in X])
        traincorr = correlation(Ypredtrain,Y[trainset], axis = 0)
        for namep, parms in model.named_parameters():
            if namep.split('.')[0] == 'cnns' and namep.split('.')[2] == 'convolutions' and namep.rsplit('.')[-1] == 'weight':
                ppms.append(parms.detach().cpu().numpy())
                bias = model.state_dict()[namep.replace('weight', 'bias')].detach().cpu().numpy()
                biases.append(bias)
                motifnames.append(np.array(['filter'+str(i)+'_'+namep.split('.')[1] for i in range(len(ppms[-1]))]))
                motifmeans.append(np.mean(np.sum(ppms[-1][:,None]*seq_seq[None,...],axis = (2,3)) + bias[:,None] , axis = 1))
                for m in range(len(ppms[-1])):
                    print(i, m)
                    Ymask = model.predict([x[trainset] for x in X], mask = [i,m], mask_value = motifmeans[-1][m])
                    importance.append(correlation(Ymask,Y[trainset], axis = 0) - traincorr)
                    effect.append(np.sum((Ypredtrain-Ymask)**3/np.sum((Ypredtrain-Ymask)**2,axis = 1)[:,None],axis = 0))
                i += 1
        
        ppms = np.concatenate(ppms, axis = 0)
        motifnames = np.concatenate(motifnames)
        importance = np.array(importance)
        effect = np.array(effect)
        biases = np.concatenate(biases, axis = 0)
        ppms = kernel_to_ppm(ppms[:,:,:], kernel_bias = biases)
        iupacmotifs = pfm2iupac(ppms, bk_freq = 0.3)
        
        write_meme_file(ppms, motifnames, 'ACGT', outname+'_kernel_ppms.meme')
        
        np.savetxt(outname+'_kernel_importance.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), importance], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(experiments))
        np.savetxt(outname+'_kernel_impact.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), effect], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(experiments))
        
        
            


    # plots scatter plot for each output class
    if '--plot_correlation_perclass' in sys.argv:
        plot_scatter(Y[testset], Y_pred, xlabel = 'Measured', ylabel = 'Predicted', titles = experiments, include_lr = False, outname = outname + '_class_scatter.jpg')
        
    # plots scatter plot fo n_genes that within the n_genes quartile
    if '--plot_correlation_pergene' in sys.argv:
        n_genes = int(sys.argv['--plot_correlation_pergene'])
        for tclass in np.unique(testclasses):
            correlation_genes = correlation(Ytest[np.random.permutation(len(Ytest))][:,consider], Y_pred[:,consider], axis = 1)
            sort = np.argsort(correlation_genes)
            posi = np.linspace(0,len(correlation_genes), n_genes).astype(int)
            i = sort[posi] 
            plot_scatter(Y[test][i].T, Ypred[i].T, xlabel = 'Measured', ylabel = 'Predicted', titles = names[testset][i], outname = outname + '_gene_scatter.jpg')
         

    
    
    








