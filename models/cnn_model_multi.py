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
from interpret_cnn import write_meme_file, pfm2iupac, kernel_to_ppm, compute_importance, pwms_from_seqs, genseq
from init import get_device, MyDataset, kmer_from_pwm, pwm_from_kmer, kmer_count, kernel_hotstart, load_parameters
from modules import parallel_module, gap_conv, interaction_module, pooling_layer, correlation_loss, correlation_both, cosine_loss, cosine_both, zero_loss, Complex, Expanding_linear, Res_FullyConnect, Residual_convolution, Res_Conv1d, MyAttention_layer, Kernel_linear, loss_dict, func_dict, simple_multi_output, PredictionHead
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
        


class combined_network(nn.Module):
    def __init__(self, loss_function = 'MSE', validation_loss = None, loss_weights = 1, val_loss_weights = 1, n_features = None, n_classes = 1, l_seqs = None, reverse_complement = None, cnn_embedding = 512, n_combine_layers = 3, combine_function = 'GELU', combine_widening = 1.1, combine_residual = 0, shift_sequence = None, random_shift = False, smooth_onehot = 0, reverse_sign = False, dropout = 0, batch_norm = False, epochs = 1000, lr = 1e-2, kernel_lr = None, cnn_lr = None, batchsize = None, patience = 50, outclass = 'Linear', input_to_combinefunc = 'ALL', outlog = 2, outlogoffset = 2, outname = None, shared_cnns = False, optimizer = 'Adam', optim_params = None, optim_weight_decay = None, verbose = True, checkval = True, init_epochs = 0, writeloss = True, write_steps = 1, device = 'cpu', load_previous = True, init_adjust = True, seed = 101010, keepmodel = False, generate_paramfile = True, add_outname = True, restart = False, **kwargs):
        super(combined_network, self).__init__()
        # set manual seed to replicate results with same seed
        # may not work if dataloader is outside this script and seed is not given to it.
        torch.manual_seed(seed)
        
        self.loss_function = loss_function
        self.validation_loss = validation_loss # validation loss can be a list of losses for each output: None means that a specific output is not considered in the validation loss and therefore does not influence the stopping criteria.
        self.loss_weights = loss_weights # weights for different losses of different data sets
        self.val_loss_weights = val_loss_weights # weights for different losses of validation loss, can be zero instead of using None as validatino loss
        self.n_features = n_features 
        self.n_classes = n_classes # is a list or an integer that determines how many columns are in each Y
        self.l_seqs = l_seqs # list with length of sequences as input
        self.n_inputs = len(l_seqs)
        self.shared_cnns = shared_cnns # if shared_cnns then the cnns are shared between inputs, Otherwise a different CNN is used for each input, can only be used if inputs have same sequence length
        self.cnn_embedding = cnn_embedding # dimension of final output of each CNN that is then concatented and given to n_combine_layers for final prediction from all inputs
        self.n_combine_layers = n_combine_layers
        self.combine_function = combine_function
        self.combine_widening = combine_widening
        self.combine_residual = combine_residual
        self.shift_sequence = shift_sequence # shift the sequence by +- a random integer 
        self.reverse_sign = reverse_sign # perform predictions from one-hot encoding with -1 to outputs that are negative
        self.smooth_onehot = smooth_onehot # add a random noise to the one-hot encoding to learn robust predictor
        self.epochs = epochs 
        self.lr = lr
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.kernel_lr = kernel_lr
        self.cnn_lr = cnn_lr
        self.batchsize = batchsize 
        self.patience = patience
        self.outclass = outclass # specific output functions to chose from: FRACTION or DIFFERENCE, EXPDIFFERENCE, LOGXPLUSFRACTION, LOGXPLUSEXPDIFFERENCE
        self.outlog = outlog # if specific function is given, and then predicted output is sent through the LOG with base outlog
        self.outlogoffset = outlogoffset # of specific function is chosen, and LOG addeed to the function, then outlogoffset is the offset before the log is taken. 
        self.input_to_combinefunc = input_to_combinefunc # if 'ALL', every output uses the embedding of all inputs, else use the assigned inputs in the list of lists, f.e if two input sequences are provided for three output modalities, create list of three lists [[0,1],[0],[1]]
        self.outname = outname
        self.optimizer = optimizer
        self.optim_params = optim_params
        self.optim_weight_decay = optim_weight_decay
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
        
        # if reverse complement for any of the sequences, the sequence and the reverse complement are stacked onto each other, and the convolution is run in both directions on the respective strand, and a max-pooling is performed for each kernel at each position for the two possible values.
        if reverse_complement is None:
            reverse_complement = [False for l in range(len(l_seqs))]
        # __dict__ is used to make a parameter file that contains all the model parameters and can be used for reloading the model or just checking the parameters that were used
        self.__dict__['reverse_complement'] = any(reverse_complement)

        if input_to_combinefunc == 'ALL':
            inplist = [i for i in range(self.n_inputs)]
            self.input_to_combinefunc = []
            if isinstance(self.n_classes, list):
                for i in range(len(self.n_classes)):
                    self.input_to_combinefunc.append(inplist)
            else:
                self.input_to_combinefunc.append(inplist)
        
        for kw in kwargs:
            self.__dict__[str(kw)] = kwargs[kw]

        # get parameters from single cnn layers
        refresh_dict = cnn(n_features = n_features, n_classes = cnn_embedding, l_seqs = l_seqs[0], seed = self.seed, dropout = dropout, batch_norm = batch_norm, add_outname = False, generate_paramfile = False, keepmodel = False, verbose = False, **kwargs).__dict__
        for rw in refresh_dict:
            if rw not in self.__dict__.keys():
                self.__dict__[str(rw)] = refresh_dict[rw]
        
        self.use_nnout = True # Determines whether to use a neural network on concatenated embedding from single cnns or to use a specifc function to combine the outputs of two sequences
        # If several outputs are given
        if isinstance(self.n_classes, list):
            if isinstance(self.outclass, list):
                for co, cout in enumerate(outclass):
                    if (('FRACTION' in cout.upper()) or ('DIFFERENCE' in cout.upper())or ('DIRECT' in cout.upper())) and len(l_seqs) == 2:
                        self.use_nnout = False
            else:
                # if n_classes is list but outclass is the same for all
                self.outclass = [outclass for n in self.n_classes]
            print(self.outclass)
            
            if self.use_nnout == False:
                self.cnn_embedding, self.n_combine_layers= [], [] #if a specific function is chosen, the individual cnns produce several outputs that are given individually to different combining functions for each data modularity. 
                for co, cout in enumerate(self.outclass):
                    if (('FRACTION' in cout.upper()) or ('DIFFERENCE' in cout.upper())or ('DIRECT' in cout.upper())) and len(l_seqs) == 2:
                        # if specific function is chosen the output of the individual CNNs is equal to the number of n_classes
                        self.cnn_embedding.append(n_classes[co])
                        self.n_combine_layers.append(0)
                    else:
                        # if not a specific function, then use a NN for the specific output
                        if isinstance(cnn_embedding, list):
                            self.cnn_embedding.append(cnn_embedding[co])
                        else:
                            self.cnn_embedding.append(cnn_embedding)
                        if isinstance(n_combine_layers, list):
                            self.n_combine_layers.append(n_combine_layers[co])
                        else:
                            self.n_combine_layers.append(n_combine_layers)
            
            elif not isinstance(self.n_combine_layers, list): # if n_combine_layers is integer but n_classes is list then generate list 
                self.n_combine_layers = [n_combine_layers for i in range(len(self.n_classes))]
                    
                    
                    
        elif (('FRACTION' in outclass.upper()) or ('DIFFERENCE' in outclass.upper())or ('DIRECT' in cout.upper())) and len(l_seqs) == 2:
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
    
        if generate_paramfile: # generate a file that contains the selected parameters of the model
            obj = open(self.outname+'_model_params.dat', 'w')
            for key in self.__dict__:
                if str(key) == 'fixed_kernels' and self.__dict__[key] is not None:
                    obj.write(key+' : '+str(len(self.__dict__[key]))+'\n')
                else:
                    obj.write(key+' : '+str(self.__dict__[key])+'\n')
            obj.close()
        self.generate_paramfile = generate_paramfile
        
        # initialize the separate cnns for the different input sequences
        self.cnns = nn.ModuleDict()
        currdim = 0
        for l, lseq in enumerate(l_seqs):
            if l > 0 and lseq == l_seqs[0] and self.shared_cnns:
                self.cnns['CNN'+str(l)] = self.cnns['CNN0']
            else:
                self.cnns['CNN'+str(l)] = cnn(n_features = n_features, n_classes = self.cnn_embedding, l_seqs = lseq, seed = self.seed +l, dropout = dropout, batch_norm = batch_norm, add_outname = False, generate_paramfile = False, keepmodel = False, verbose = verbose, outclass = combine_function, reverse_complement = reverse_complement[l], shift_sequence = shift_sequence, **kwargs)
        
        if isinstance(n_classes, list):
            self.nclayers = nn.ModuleList()
            self.classifier = nn.ModuleList()
            for co, cout in enumerate(self.outclass): # if one of the outclasses is not a NN then the individual CNNs produce individual outputs for each output NN.
                if ('FRACTION' in cout.upper()) or ('DIFFERENCE' in cout.upper()) or ('DIRECT' in cout.upper()):
                    prefunc = None
                    if 'FRACTION' in cout.upper():
                        prefunc = 'ReLU'
                    self.nclayers.append(nn.Identity())
                    if isinstance(outlog, list):
                        ool = outlog[co]
                        loff = outlogoffset[co]
                    else:
                        ool = outlog
                        loff = outlogoffset
                    self.classifier.append(simple_multi_output(out_relation = cout, pre_function = prefunc, log = ool, logoffset = loff))
        
                else:
                    currdim = len(self.input_to_combinefunc[co]) * cnn_embedding # count the size of the input to the combining network from the concatenated outputs, is only used if no specific function was chosen as outclass
                    # Even if a specific function is chosen, then each individual CNN returns one array for the function and one with size cnn_embedding            for the NN predicted modularity. The sum of there is the currdim 
                    if self.n_combine_layers[co] > 0:
                        self.nclayers.append(Res_FullyConnect(currdim, outdim = currdim, n_classes = None, n_layers = self.n_combine_layers[co], layer_widening = combine_widening, batch_norm = self.batch_norm, dropout = self.dropout, activation_function = combine_function, residual_after = combine_residual, bias = True))
                    else:
                        self.nclayers.append(nn.Identity())
                    self.classifier.append(PredictionHead(currdim, n_classes[co], cout, dropout = self.dropout, batch_norm = self.batch_norm))
        
        # difference can be used to fit log(expression) = log(kt/kd) = log(kt) - log(kd) or expression = kt/kd 
        # However, DO NOT use difference if values are log(a+expression) since they cannot be represented as difference
        elif self.use_nnout == False:
            print('MAKE SURE that processing of the log is identical to what is known about the data.\nCurrently:', outclass, outlogoffset, outlog)
            prefunc = None
            if 'FRACTION' in outclass.upper():
                prefunc = 'ReLU'
            self.classifier = simple_multi_output(out_relation = outclass, pre_function = prefunc, log = outlog, logoffset = outlogoffset)
        
        else:
            currdim = len(self.input_to_combinefunc[0]) * cnn_embedding
            ## add option to have list of n_combine_layers and then split the classifier as before
            if self.n_combine_layers > 0:
                self.nclayers = Res_FullyConnect(currdim, outdim = currdim, n_classes = None, n_layers = self.n_combine_layers, layer_widening = combine_widening, batch_norm = self.batch_norm, dropout = self.dropout, activation_function = combine_function, residual_after = combine_residual, bias = True)
            self.classifier = PredictionHead(currdim, n_classes, self.outclass, dropout = self.dropout, batch_norm = self.batch_norm)
            
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
                if mask == c:
                    pred.append(mask_value.unsqueeze(0).repeat(x[c].size(dim = 0),1))
                else:
                    pred.append(cn(x[c]))
            elif isinstance(mask,list) and len(mask) == 2:
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
        
        
        if isinstance(self.n_classes, list):
            multipred = []
            for s, ncl in enumerate(self.n_classes):
                if not self.use_nnout:
                    npred = [pred[t][s] for t in self.input_to_combinefunc[s]]
                    if ('FRACTION' in self.outclass[s].upper()) or ('DIFFERENCE' in self.outclass[s].upper()) or ('DIRECT' in self.outclass[s].upper()):
                        multipred.append(self.classifier[s](npred))
                    else:
                        npred = torch.cat(npred, dim = -1)
                        if self.n_combine_layers[s] > 0:
                            npred = self.nclayers[s](npred)
                        multipred.append(self.classifier[s](npred))
                    
                else:
                    npred = torch.cat([pred[t] for t in self.input_to_combinefunc[s]], dim = -1)
                    if self.n_combine_layers[s] > 0:
                        npred = self.nclayers[s](npred)
                    multipred.append(self.classifier[s](npred))
            pred = multipred
            
        
        else:
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
        self.saveloss = fit_model(self, X, Y, XYval = XYval, sample_weights = sample_weights, loss_function = self.loss_function, validation_loss = self.validation_loss, loss_weights = self.loss_weights, val_loss_weights = self.val_loss_weights, batchsize = self.batchsize, device = self.device, optimizer = self.optimizer, optim_params = self.optim_params, optim_weight_decay=self.optim_weight_decay, verbose = self.verbose, lr = self.lr, kernel_lr = self.kernel_lr, hot_start = self.hot_start, warm_start = self.warm_start, outname = self.outname, adjust_lr = self.adjust_lr, patience = self.patience, init_adjust = self.init_adjust, keepmodel = self.keepmodel, load_previous = self.load_previous, write_steps = self.write_steps, checkval = self.checkval, writeloss = self.writeloss, init_epochs = self.init_epochs, epochs = self.epochs, l1reg_last = self.l1reg_last, l2_reg_last = self.l2reg_last, l1_kernel = self.l1_kernel, reverse_sign = self.reverse_sign, shift_back = self.shift_sequence, random_shift = self.random_shift, smooth_onehot = self.smooth_onehot, restart = self.restart, multiple_input = True, **self.kwargs)
    






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
        mask = np.random.permutation(len(names))[:trand]
        X, names = [x[mask] for x in X], names[mask]
        if Y is not None:
            if isinstance(Y, list):
                Y = [y[mask] for y in Y]
            else:
                Y = Y[mask]
    
    if '--select_list' in sys.argv:
        sel = np.genfromtxt(sys.argv[sys.argv.index('--select_list')+1], dtype = str)
        mask = np.isin(names, sel)
        if np.sum(mask) < 1:
            print('Selected list names do not match the names in the data')
            sys.exit()
        X, names = X[mask], names[mask]
        if Y is not None:
            if isinstance(Y, list):
                Y = [y[mask] for y in Y]
            else:
                Y = Y[mask]
    
    
    # Don't use with multiple Y
    if '--remove_allzero' in sys.argv and Y is not None:
        mask = np.sum(Y, axis = 1) != 0
        Y = Y[mask]
        X, names = [x[mask] for x in X], names[mask]
    # Don't use with multiple Y
    if '--remove_allzerovar' in sys.argv and Y is not None:
        mask = np.std(Y, axis = 1) != 0
        Y = Y[mask]
        X, names = [x[mask] for x in X], names[mask]
    # Don't use with multiple Y
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
            inputfile = create_outname(inp, inputfile, lword = '')
    if ',' in outputfile:
        outputfiles = outputfile.split(',')
        outputfile = outputfiles[0]
        for inp in outputfiles[1:]:
            outputfile = create_outname(inp, outputfile, lword = '')       
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
            if isinstance(Y, list):
                Yclass = (np.sum(np.absolute(np.concatenate(Y,axis =1))>=cutoff, axis = 1) > 0).astype(int)
            else:
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
        trainset, testset, valset = create_sets(len(names), folds, fold, Yclass = Yclass, genenames = names)
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
        if isinstance(Y, list):
            Y = [y[permute] for y in Y]
        else:
            Y = Y[permute]
        
    
    if '--norm2output' in sys.argv:
        print ('ATTENTION: output has been normalized along data points')
        outname += '-n2out'
        if isinstance(Y, list):
            outnorm, Y = [], []
            for y in Y:
                outnorm.append(np.sqrt(np.sum(y*y, axis = 1))[:, None])
                Y.append(y/outnorm[-1])
        else:
            outnorm = np.sqrt(np.sum(Y*Y, axis = 1))[:, None] 
            Y = Y/outnorm
        
    elif '--norm2outputclass' in sys.argv:
        print('ATTENTION: output has been normalized along data classess')
        outname += '-n2outc'
        if isinstance(y, list):
            outnorm, Y = [], []
            for y in Y:
                outnorm.append(np.sqrt(np.sum(y*y, axis = 0)))
                Y.append(y/outnorm[-1])
        else:
            outnorm = np.sqrt(np.sum(Y*Y, axis = 0))
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
            if isinstance(Y, list):
                params['n_classes'] = [np.shape(y)[-1] for y in Y]
            else:
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
            if isinstance(Y, list):
                params['n_classes'] = [np.shape(y)[-1] for y in Y]
            else:
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
        model.classifier.classifier.Linear.weight = nn.Parameter(model.classifier.classifier.Linear.weight[select_track])
        model.classifier.classifier.Linear.bias = nn.Parameter(model.classifier.classifier.Linear.bias[select_track])
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
        if isinstance(Y, list):
            Ytraintomodel, Yvaltomodel = [y[trainset] for y in Y], [y[valset] for y in Y]
            
        else:
            Ytraintomodel, Yvaltomodel = Y[trainset], Y[valset]
        model.fit([x[trainset] for x in X], Ytraintomodel, XYval = [[x[valset] for x in X], Yvaltomodel])
    
    Y_pred = model.predict([x[testset] for x in X])
    
    if isinstance(Y,list):
        # Y_pred returns concatenated list
        Y = np.concatenate(Y, axis = 1)
        experiments = np.concatenate(experiments)
    
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
        print_averages(Y_pred, Y[testset], testclasses, sys.argv)
        
        # USE: --save_correlation_perclass --save_auroc_perclass --save_auprc_perclass --save_mse_perclass --save_correlation_pergene '--save_mse_pergene --save_auroc_pergene --save_auprc_pergene --save_topdowncorrelation_perclass
        save_performance(Y_pred, Y[testset], testclasses, experiments, names[testset], outname, sys.argv, compare_random = True)
        
    if '--save_predictions' in sys.argv:
        print('SAVED', outname+'_pred.txt')
        np.savetxt(outname+'_pred.txt', np.append(names[testset][:, None], Y_pred, axis = 1), header = ' '.join(experiments), fmt = '%s')
    
    if '--test_individual_network' in sys.argv:
        # pick a random set of input sequences to compute mean output of individual networks
        randomseqs = np.random.permutation(len(names))[:7000]
        # predict original predictions to compare losses to 
        Ypredtrain = model.predict(X)
        Ymask = []
        for i in range(len(X)):
            # get mean representations for individual networks
            mean_rep = []
            for r in range(0,len(randomseqs), 10):
                rand = randomseqs[r:r+10]
                mean_rep.append(model.forward([torch.Tensor(x[rand]).to(model.device) for x in X], location = '0', cnn = i).detach().cpu().numpy())
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
            header= ['MSE', 'Corr']
            # compute the performance for this class for the full predictions
            trainmse = mse(Ypredtrain[:,consider],Y[:,consider] ,axis =1)
            traincorr = correlation(Ypredtrain[:,consider],Y[:,consider] ,axis =1)
            #print(tclass)
            #print(traincorr)
            for i in range(len(X)):
                header.append('CNN'+str(i)+'MSE')
                header.append('CNN'+str(i)+'Corr')
                # compare the performance of the masked predictions with the full predicdtions
                MSEdif = mse(Ymask[i][:,consider],Y[:,consider],axis = 1) - trainmse
                genecont.append(MSEdif)
                Corrdif = correlation(Ymask[i][:,consider],Y[:,consider],axis = 1) - traincorr
                #print(i, Ymask[i][:,consider][0], Ypredtrain[0,consider])
                #print(Corrdif)
                genecont.append(Corrdif)
            np.savetxt(outname+'_netatt'+tclass+'.dat', np.concatenate([[names],np.around([trainmse,traincorr],4), np.around(np.array(genecont),6)],axis = 0).T, header = 'Gene '+' '.join(np.array(header)), fmt = '%s')
            
            
            
    # Generate PPMs for kernels
    # Generate PWMs from activation of sequences
    # Generate global importance for every output track
    # Generate mean direction of effect from kernel
    # Generate global importance of kernel for each gene and testclasses
    if '--convertedkernel_ppms' in sys.argv:
        ppms = []
        pwms = []
        biases = []
        motifnames = []
        motifmeans = []
        seqactivations = []
        importance = []
        geneimportance = []
        effect = []

        seq_seq = genseq(model.l_kernels, 100000) # generate 100000 random sequences
        i =0
        # predict original training predictions
        Ypredtrain = model.predict([x for x in X])
        # compute the correlation of between data points for each output track
        traincorr = correlation(Ypredtrain,Y, axis = 0)
        genetraincorr = []
        for t, testclass in enumerate(np.unique(testclasses)):
            consider = np.where(testclasses == testclass)[0]
            geneimportance.append([])
            if len(consider) > 1:
                genetraincorr.append(correlation(Ypredtrain[:,consider],Y[:,consider], axis = 1))
        for namep, parms in model.named_parameters():
            if namep.split('.')[0] == 'cnns' and namep.split('.')[2] == 'convolutions' and namep.rsplit('.')[-1] == 'weight':
                # collect the first layer convolution kernels
                ppms.append(parms.detach().cpu().numpy())
                # collect the biases if bias is not None
                if model.kernel_bias:
                    bias = model.state_dict()[namep.replace('weight', 'bias')].detach().cpu().numpy()
                else:
                    bias = np.zeros(len(ppms[-1]))
                biases.append(bias)
                # Generate names for all kernels
                motifnames.append(np.array(['filter'+str(i)+'_'+namep.split('.')[1] for i in range(len(ppms[-1]))]))
                # compute motif means from the activation of kernels with the random sequences.
                seqactivations = np.sum(ppms[-1][:,None]*seq_seq[None,...],axis = (2,3))
                # generate motifs from aligned sequences with activation over 0.9 of the maximum activation
                pwms.append(pwms_from_seqs(seq_seq, seqactivations, 0.9))
                # take the mean of these activations from all 100000 sequences as a mean actiavation value for each filter.
                motifmeans.append(np.mean(seqactivations + bias[:,None] , axis = 1))
                for m in range(len(ppms[-1])):
                    # make predictions from models with meaned activations from kernels
                    Ymask = model.predict([x for x in X], mask = [i,m], mask_value = motifmeans[-1][m])
                    # importance as difference in correlation between full model to train data and masked model to train data
                    importance.append(np.around(correlation(Ymask,Y, axis = 0) - traincorr,3))
                    # compute the importance of the kernel for every gene
                    for t, testclass in enumerate(np.unique(testclasses)):
                        consider = np.where(testclasses == testclass)[0]
                        if len(consider) > 1:
                            geneimportance[t].append(np.around(correlation(Ymask[:,consider],Y[:,consider], axis = 1) - genetraincorr[t],3))
                    # effect for a track shows the direction that the kernel causes on average over all genes
                    # it is weighted by how much it changes the mse of each gene
                    effect.append(np.around(np.sum((Ypredtrain-Ymask)**3/np.sum((Ypredtrain-Ymask)**2,axis = 1)[:,None],axis = 0),6))
                i += 1
        
        
        motifnames = np.concatenate(motifnames)
        importance = np.array(importance)
        geneimportance = np.array(geneimportance)
        effect = np.array(effect)
        
        ppms = np.concatenate(ppms, axis = 0)
        biases = np.concatenate(biases, axis = 0)
        # create ppms directly from kernel matrix
        ppms = kernel_to_ppm(ppms[:,:,:], kernel_bias = biases)
        
        
        # generate pwms from most activated sequences
        pwms = np.concatenate(pwms, axis = 0)
        iupacmotifs = pfm2iupac(pwms, bk_freq = 0.3)
        
        write_meme_file(ppms, motifnames, 'ACGT', outname+'_kernel_ppms.meme')
        write_meme_file(pwms, motifnames, 'ACGT', outname+'_kernel_pwms.meme')
        
        np.savetxt(outname+'_kernattrack.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), importance], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(experiments))
        for t, testclass in enumerate(np.unique(testclasses)):
            consider = np.where(testclasses == testclass)[0]
            if len(consider) > 1:
                np.savetxt(outname+'_kernatgene'+testclass+'.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), geneimportance[t]], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(names))
        np.savetxt(outname+'_kernimpact.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), effect], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(experiments))
        
        
            


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
         

    
    
    








