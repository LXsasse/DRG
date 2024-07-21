# cnn_model_multi.py 

'''
Contains torch.Module for cnn that can scan multiple sequences and predict multiple matrices from that

'''


import sys, os 
import numpy as np
import torch.nn as nn
import torch
from collections import OrderedDict


from .model_output import add_params_to_outname
from .modules import parallel_module, gap_conv, interaction_module, pooling_layer, correlation_loss, correlation_both, cosine_loss, cosine_both, zero_loss, Complex, Expanding_linear, Res_FullyConnect, Residual_convolution, Res_Conv1d, MyAttention_layer, Kernel_linear, loss_dict, func_dict, func_dict_single, simple_multi_output, PredictionHead
from .model_training import batched_predict, fit_model

from .cnn_model import cnn
import time
        


class combined_network(nn.Module):
    '''
    combines the outputs of several sequence cnn models to predict one, or several data modalities
    Data modalities can be predicted with Fully connected layers
    or with analytical functions from the outputs of the single sequence cnns
    '''
    
    def __init__(self, loss_function = 'MSE', validation_loss = None, loss_weights = 1, val_loss_weights = 1, n_features = None, n_classes = 1, l_seqs = None, reverse_complement = None, cnn_embedding = 512, n_combine_layers = 3, combine_function = 'GELU', shared_embedding = True, combine_widening = 1.1, combine_residual = 0, shift_sequence = None, random_shift = False, smooth_onehot = 0, reverse_sign = False, dropout = 0, batch_norm = False, epochs = 1000, lr = 1e-2, kernel_lr = None, cnn_lr = None, batchsize = None, patience = 50, outclass = 'Linear', input_to_combinefunc = 'ALL', outlog = 2, outlogoffset = 2, outname = None, shared_cnns = False, optimizer = 'Adam', optim_params = None, optim_weight_decay = None, verbose = True, checkval = True, init_epochs = 0, writeloss = True, write_steps = 1, device = 'cpu', load_previous = True, init_adjust = True, seed = 101010, keepmodel = False, generate_paramfile = True, add_outname = True, restart = False, **kwargs):
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
        self.combine_function = combine_function # activation function of the combining network. (For network set to GELU)
        self.shared_embedding = shared_embedding # if False: the individual sequence cnns return individual embeddings for len(n_classses) outputs, if True: returns only one embedding that is cloned for all. 
        #if output is direct or difference, combine_function is the outclass of the subnetwork. Therefore, if a value can be positive and negativbe, the GELU function will not be correct. (For direct or difference set to Linear)
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

        # if 'ALL', every output uses the embedding of all inputs, else use the assigned inputs in the list of lists, f.e if two input sequences are provided for three output modalities, create list of three lists [[0,1],[0],[1]]
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
            
            if not isinstance(self.combine_function, list) and shared_embedding == False:
                self.combine_function = [combine_function for n in self.n_classes]
                for c, cf in enumerate(self.combine_function):
                    if ('DIFFERENCE' in self.outclass[c].upper())or ('DIRECT' in  self.outclass[c].upper()):
                        self.combine_function[c] = 'Linear'
                
                    
            
            if self.use_nnout == False:
                self.cnn_embedding, self.n_combine_layers= [], [] #if a specific function is chosen, the individual cnns produce several outputs that are given individually to different combining functions for each data modularity. 
                if self.shared_embedding:
                    if (np.array(self.n_classes) == self.n_classes[0]).all():
                        self.cnn_embedding = n_classes[0]
                        if isinstance(self.combine_function, list):
                            if (np.array(self.combine_function) == self.combine_function[0]).all():
                                self.combine_function = self.combine_function[0]
                    else:
                        self.shared_embedding = False
                        print('shared_embedding set to False because n_classes or combine_function not all equal')
                else:
                    for co, cout in enumerate(self.outclass):
                        if (('FRACTION' in cout.upper()) or ('DIFFERENCE' in cout.upper())or ('DIRECT' in cout.upper())) and len(l_seqs) == 2:
                            # if specific function is chosen the output of the individual CNNs is equal to the number of n_classes
                            self.cnn_embedding.append(n_classes[co])
                        else:
                            # if not a specific function, then use a NN for the specific output
                            if isinstance(cnn_embedding, list):
                                self.cnn_embedding.append(cnn_embedding[co])
                            else:
                                self.cnn_embedding.append(cnn_embedding)

                for co, cout in enumerate(self.outclass):
                    if (('FRACTION' in cout.upper()) or ('DIFFERENCE' in cout.upper())or ('DIRECT' in cout.upper())) and len(l_seqs) == 2:
                        self.n_combine_layers.append(0)
                    else:
                        if isinstance(n_combine_layers, list):
                            self.n_combine_layers.append(n_combine_layers[co])
                        else:
                            self.n_combine_layers.append(n_combine_layers)

                    
            elif not isinstance(self.n_combine_layers, list): # if n_combine_layers is integer but n_classes is list then generate list 
                self.n_combine_layers = [n_combine_layers for i in range(len(self.n_classes))]
                    
                    
        elif (('FRACTION' in outclass.upper()) or ('DIFFERENCE' in outclass.upper())or ('DIRECT' in outclass.upper())) and len(l_seqs) == 2:
            self.cnn_embedding = cnn_embedding = n_classes
            self.n_combine_layers = 0
            self.use_nnout = False
            if ('DIFFERENCE' in outclass.upper())or ('DIRECT' in outclass.upper()):
                self.combine_function = 'Linear'
        
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
                self.cnns['CNN'+str(l)] = cnn(n_features = n_features, n_classes = self.cnn_embedding, l_seqs = lseq, seed = self.seed +l, dropout = dropout, batch_norm = batch_norm, add_outname = False, generate_paramfile = False, keepmodel = False, verbose = verbose, outclass = self.combine_function, reverse_complement = reverse_complement[l], shift_sequence = shift_sequence, **kwargs)
        
        if isinstance(n_classes, list):
            self.nclayers = nn.ModuleList()
            self.classifier = nn.ModuleList()
            if self.shared_embedding and not isinstance(self.combine_function, list):
                self.combine_function = [self.combine_function for co in self.outclass]
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
                        self.nclayers.append(Res_FullyConnect(currdim, outdim = currdim, n_classes = None, n_layers = self.n_combine_layers[co], layer_widening = combine_widening, batch_norm = self.batch_norm, dropout = self.dropout, activation_function = self.combine_function[co], residual_after = combine_residual, bias = True))
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
                self.nclayers = Res_FullyConnect(currdim, outdim = currdim, n_classes = None, n_layers = self.n_combine_layers, layer_widening = combine_widening, batch_norm = self.batch_norm, dropout = self.dropout, activation_function = self.combine_function, residual_after = combine_residual, bias = True)
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
                    if self.shared_embedding:
                        npred = [pred[t] for t in self.input_to_combinefunc[s]]
                    else:
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
    
    def predict(self, X, mask = None, mask_value = 0, device = None, enable_grad = False):
        if device is None:
            device = self.device
        predout = batched_predict(self, X, mask = mask, mask_value = mask_value, device = device, batchsize = self.batchsize, shift_sequence = self.shift_sequence, random_shift = self.random_shift, enable_grad = enable_grad)
        return predout

    def fit(self, X, Y, XYval = None, sample_weights = None):
        self.saveloss = fit_model(self, X, Y, XYval = XYval, sample_weights = sample_weights, loss_function = self.loss_function, validation_loss = self.validation_loss, loss_weights = self.loss_weights, val_loss_weights = self.val_loss_weights, batchsize = self.batchsize, device = self.device, optimizer = self.optimizer, optim_params = self.optim_params, optim_weight_decay=self.optim_weight_decay, verbose = self.verbose, lr = self.lr, kernel_lr = self.kernel_lr, hot_start = self.hot_start, warm_start = self.warm_start, outname = self.outname, adjust_lr = self.adjust_lr, patience = self.patience, init_adjust = self.init_adjust, keepmodel = self.keepmodel, load_previous = self.load_previous, write_steps = self.write_steps, checkval = self.checkval, writeloss = self.writeloss, init_epochs = self.init_epochs, epochs = self.epochs, l1reg_last = self.l1reg_last, l2_reg_last = self.l2reg_last, l1_kernel = self.l1_kernel, reverse_sign = self.reverse_sign, shift_back = self.shift_sequence, random_shift = self.random_shift, smooth_onehot = self.smooth_onehot, restart = self.restart, multiple_input = True, **self.kwargs)
    





    








