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
from output import print_averages, save_performance, plot_scatter
from functions import dist_measures
from interpret_cnn import write_meme_file, pfm2iupac, kernel_to_ppm, compute_importance 
from init import get_device, MyDataset, kmer_from_pwm, pwm_from_kmer, kmer_count, kernel_hotstart, load_parameters
from modules import parallel_module, gap_conv, interaction_module, pooling_layer, correlation_loss, correlation_both, cosine_loss, cosine_both, zero_loss, Complex, Expanding_linear, Res_FullyConnect, Residual_convolution, Res_Conv1d, MyAttention_layer, Kernel_linear, loss_dict, func_dict
from predict import pwmset, pwm_scan, batched_predict
from train import fit_model



# Include customized Convolutions: 
    # CNN network for each convolution, can be interpreted as one complex motif, should not sum over all positions but instead put them into a fully connected network and only sum at the end. So that this network creates outputs for each position 
    
# Problem of scales: 
    #Some Genes have high basis expression (reason are a bunch of TFs)
    # Changes in there will have higher impact than for small TFs, so we need:
        # We should train on several scalings across classes to help starting layers with feature extraction, log(1+c) and log(1+c/1+low_reference) or log(c/medium_reference)
        # This emphasizes changes in one output and the base-state in the other. This will help overall because TFs that are important in base state may be important in change state for another gene
        # INCLUDE all different normalizations to get best result for data that you're most interested in
    
    
    # How can we include ATTAC-seq into per gene expression:
        # could add attac signal to onehot encoded input
        # could add first few principal components of attac to signal (probably NMF)
        # Could split network at a point, into per gene prediction and per window prediction to predict attac signal
        
    # How to include other signals:
        # include further tracks
        # Similar to ATTAC-seq track: Need to split network after a few initial layers
        # at the point where initial inputs represent the region for the epigenetic mark
        


class combine_networks(nn.Module):
    def __init__(self, list_of_networks, list_of_inputs, output, combine_model):
        this_will = 0
    # provide a list of initialized models, their different inputs, the output, and a specific model that combines all these models, 
    # forward pass executes all models in the list with the x and then concatenates them and applies combine model
    # Will have to change/combine forward and embedding function use this with cnn()








# highly flexible Convolutional neural network architecture
class cnn(nn.Module):
    def __init__(self, loss_function = 'MSE', validation_loss = None, n_features = None, n_classes = 1, l_seqs = None, num_kernels = 0, kernel_bias = True, fixed_kernels = None, motif_cutoff = None, l_kernels = 7, kernel_function = 'GELU', warm_start = False, hot_start = False, hot_alpha=0.01, kernel_thresholding = 0, max_pooling = True, mean_pooling = False, pooling_size = None, pooling_steps = None, dilated_convolutions = 0, strides = 1, conv_increase = 1., dilations = 1, l_dilkernels = None, dilmax_pooling = None, dilmean_pooling = None, dilpooling_size = None, dilpooling_steps = None, dilpooling_residual = 1, gapped_convs = None, embedding_convs = 0, n_transformer = 0, n_attention = 0, n_distattention = 0, dim_distattention=2.5, dim_embattention = None, maxpool_attention = 0, sum_attention = False, transformer_convolutions = 0, trdilations = 1, trstrides = 1, l_trkernels = None, trconv_dim = None, trmax_pooling = False, trmean_pooling = False, trpooling_size = None, trpooling_steps = None, trpooling_residual = 1, nfc_layers = 0, nfc_residuals = 0, fc_function = None, layer_widening = 1.1, interaction_layer = False, neuralnetout = 0, dropout = 0., batch_norm = False, l1_kernel = 0, l2reg_last = 0., l1reg_last = 0., shift_sequence = 0, reverse_sign = False, epochs = 1000, lr = 1e-2, kernel_lr = None, adjust_lr = 'F', batchsize = None, patience = 25, outclass = 'Linear', outname = None, optimizer = 'Adam', optim_params = None, verbose = True, checkval = True, init_epochs = 250, writeloss = True, write_steps = 10, device = 'cpu', load_previous = True, init_adjust = False, seed = 101010, keepmodel = False, **kwargs):
        super(cnn, self).__init__()
        
        # Set seed for all random processes in the model: parameter init and other dataloader
        torch.manual_seed(seed)
        self.seed = seed
        self.verbose = verbose # if true prints out epochs and losses
        self.loss_function = loss_function # Either defined function or one None for 'mse'
        self.validation_loss = validation_loss
        
        # save kwargs for fitting
        self.kwargs = kwargs
        
        self.keepmodel = keepmodel # Determines if model parameters will be kept in pth file after training
        
        self.n_features = n_features # Number of features in one-hot coding
        self.l_seqs = l_seqs # length of padded sequences
        self.n_classes = n_classes # output classes to predict
        
        self.shift_sequence = shift_sequence # During training sequences that are shifted by 'shift_sequence' positions will be added
        self.reverse_sign = reverse_sign # During training, the sign of the input and the output will be shifted. This mirror image of the data can be helpful with training
        
        if self.n_features is None or self.l_seqs is None:
            print('n_features and l_seqs need to be defined')
            sys.exit()
        
        self.adjust_lr = adjust_lr # don't ignore kernel_lr but also adjust other lrs according to the number of parameters in the layer, the location of the layer and the relationship between kernel_lr and lr
        self.kernel_lr = kernel_lr
        self.num_kernels = num_kernels  # number of learnable kernels
        self.l_kernels = l_kernels # length of learnable kernels
        self.kernel_bias = kernel_bias # Turn on/off bias of kernels
        self.kernel_function = kernel_function # Non-linear function applied to kernel outputs
        self.kernel_thresholding = kernel_thresholding # Thresholding function a_i*k_i=bi for each kernel, important to use with pwms because they don't have any cutoffs
        self.fixed_kernels = fixed_kernels # set of fixed value kernels (pwms)
        self.motif_cutoff = motif_cutoff # when scanning with fixed kernels (pwms), all values below this cutoff are set to zero, creates sparser scanning matrix
        if self.fixed_kernels is None:
            self.motif_cutoff = None # set to default if no fixed kernels
        
        self.warm_start = warm_start # initialize kernels with kernel from 1layer cnn
        self.hot_start = hot_start # hot start initiates the kernels with the best k-mers from Lasso regression or from simple statistics
        self.hot_alpha = hot_alpha # starting regularization parameter for L1-regression 
        if not self.hot_start:
            self.hot_alpha = None
        
        self.max_pooling = max_pooling # If max pooling should be used
        self.mean_pooling = mean_pooling # If mean pooling should be used, if both False entire set is given to next layer
        self.pooling_size = pooling_size    # The size of the pooling window, Can span the entire sequence
        self.pooling_steps = pooling_steps # The step size of the pooling window, stepsizes smaller than the pooling window size create overlapping regions
        if self.max_pooling == False and self.mean_pooling == False:
            self.pooling_size = None
            self.pooling_steps = None
        elif self.pooling_size is None and self.pooling_steps is None:
            self.pooling_size = int((self.l_seqs - (self.l_kernels -1))/1.)
            self.pooling_steps = int((self.l_seqs - (self.l_kernels -1))/1.)
        elif self.pooling_steps is None:
            self.pooling_steps = self.pooling_size
        elif self.pooling_size is None:
            self.pooling_size = self.pooling_steps
        
        self.dilated_convolutions = dilated_convolutions # Number of additional dilated convolutions
        self.strides = strides #Strides of additional convolutions
        self.dilations = dilations # Dilations of additional convolutions
        self.conv_increase = conv_increase # Factor by which number of additional convolutions increases in each layer
        self.dilpooling_residual = dilpooling_residual # Number of convolutions before residual is added
        
        # Length of additional convolutional kernels
        if l_dilkernels is None:
            self.l_dilkernels = l_kernels
        else:
            self.l_dilkernels = l_dilkernels
        
        # Max pooling for additional convolutional layers
        if dilmax_pooling is None:
            self.dilmax_pooling = max_pooling
        else:
            self.dilmax_pooling = dilmax_pooling
        
        # Mean pooling for additional convolutional layers
        if dilmean_pooling is None:
            self.dilmean_pooling = mean_pooling
        else:
            self.dilmean_pooling = dilmean_pooling
        # Pooling sizes and step size for additional max pooling layers
        self.dilpooling_size = dilpooling_size
        self.dilpooling_steps = dilpooling_steps
        
        if self.dilmean_pooling == False and self.dilmax_pooling == False:
            self.dilpooling_size = None
            self.dilpooling_steps = None
        elif self.dilpooling_steps is None and self.dilpooling_size is None:
            self.dilpooling_steps = self.dilpooling_size = self.pooling_steps
        elif self.dilpooling_steps is None:
            self.dilpooling_steps = self.dilpooling_size
        elif self.dilpooling_size is None:
            self.dilpooling_size = self.dilpooling_steps
        
        # all to default if
        if self.dilated_convolutions == 0:
            self.strides, self.dilations = 1,1
            self.l_dilkernels, self.dilmax_pooling, self.dilmean_pooling,self.dilpooling_size, self.dilpooling_steps = None, None, None, None, None
        
        # If lists given, parallel gapped convolutions are initiated
        self.gapped_convs = gapped_convs # list of quadruples, first the size of the kernel left and right, second the gap, third the number of kernals, fourth the stride stride. Will generate several when parallel layers with different gapsizes if list is given.
        # Gapped convolutions are placed after maxpooling layer and then concatenated with output from previous maxpooling layer. 
        
        # reduce the dimensions of the output of the convolutional layer before giving it to the transformer layer
        self.embedding_convs = embedding_convs
        
        self.n_transformer = n_transformer
        self.n_attention = n_attention
        self.n_distattention = n_distattention # intializes the distance_attention with n heads
        self.dim_distattention = dim_distattention # multiplicative value by which dimension will be increased in embedding
        self.dim_embattention = dim_embattention # dimension of values
        self.sum_attention = sum_attention # if inputs are duplicated n_heads times then they will summed afterwards
        self.maxpool_attention = maxpool_attention # generate maxpool layer after attention layer to reduce length of input
        
        self.transformer_convolutions = transformer_convolutions # Number of additional convolutions afer transformer layer
        self.trpooling_residual = trpooling_residual # Number of layers that are scipped by residual layer
        self.trstrides = trstrides #Strides of additional convolutions
        self.trdilations = trdilations # Dilations of additional convolutions
        self.trconv_dim = trconv_dim # dimensions of convolutions after transformer
        
        # Length of additional convolutional kernels
        if l_trkernels is None:
            self.l_trkernels = l_kernels
        else:
            self.l_trkernels = l_trkernels
        
        # Pooling sizes and step size for additional max pooling layers
        self.trpooling_size = trpooling_size
        self.trpooling_steps = trpooling_steps
        
        self.trmean_pooling = trmean_pooling
        self.trmax_pooling = trmax_pooling
        
        if self.trmean_pooling == False and self.trmax_pooling == False:
            self.trpooling_size = None
            self.trpooling_steps = None
        elif self.trpooling_steps is None and self.trpooling_size is None:
            self.trpooling_steps = self.trpooling_size = self.pooling_steps
        elif self.trpooling_steps is None:
            self.trpooling_steps = self.trpooling_size
        elif self.trpooling_size is None:
            self.trpooling_size = self.trpooling_steps
        
        # all to default if
        if self.transformer_convolutions == 0:
            self.trstrides, self.trdilations, self.trconv_dim = None, None, None
            self.l_trkernels, self.trmax_pooling, self.trmean_pooling, self.trpooling_size, self.trpooling_steps = None, False, False, None, None
        if self.dilated_convolutions == 0 and self.transformer_convolutions == 0:
             self.conv_increase = 1   
        
        self.nfc_layers = nfc_layers # Number of fully connected ReLU layers after pooling before last layer
        self.nfc_residuals = nfc_residuals # Number of layers after which residuals should be added
        if fc_function is None:
            fc_function = kernel_function
        self.fc_function = fc_function # Non-linear transformation after each fully connected layer
        if self.nfc_layers == 0:
            self.fc_function = None
        self.layer_widening = layer_widening # Factor by which number of parameters are increased for each layer
        
        self.interaction_layer = interaction_layer # If true last layer multiplies the values of all features from previous layers with each other and weights them for classifation or prediction
        
        self.neuralnetout = neuralnetout # Determines the number of fully connected residual layers that are created for each output class
        
        self.l2reg_last = l2reg_last # L2 norm for last layer
        self.l1reg_last = l1reg_last # L1 norm for last layer
        self.l1_kernel = l1_kernel # L1 regularization for kernel parameters
        
        self.batch_norm = batch_norm # batch_norm True or False
        self.dropout = dropout # Fraction of dropout
            
        self.epochs = epochs # Max number of iterations
        self.lr = lr # stepsize for updates
        
        self.batchsize = batchsize # Number of data points that are included in one forward and backward step, if None, entire data set is used
        
        self.patience = patience # number of last validation loss values to look for improvement before ealry stopping is applied
        self.init_epochs = init_epochs # intial epochs in which lr can be adjusted and early stopping is not applied
        self.init_adjust = init_adjust # IF true reduce learning rate if loss of training data increases within init_epochs
        self.load_previous = load_previous # if an earlier model with better validation loss should be loaded when loop is stopped
        self.device = device # determine device for training
        self.checkval = checkval # If one should check the stop criterion on the validation loss for early stopping

        self.writeloss = writeloss # If true a file with the losses per epoch will be generated
        self.write_steps = write_steps # Number of steps before write out
        self.optimizer = optimizer # Choice of optimizer: Adam, SGD, Adagrad, see below for parameters
        
        self.optim_params = optim_params # Parameters given to the optimizer, For each optimizer can mean something else, lookg at fit() to see what they define.
        
        self.outclass = outclass # Class: sigmoid, Multi_class: Softmax, Complex: for non-linear scaling

        ### Generate file name from all settings
        if outname is None:
            self.outname = 'CNNmodel'  # add all the other parameters
        else:
            self.outname = outname
        
        self.outname += '_lf'+loss_function[:3]+loss_function[max(3,len(loss_function)-3):]+'_nk'+str(num_kernels)+'-'+str(l_kernels)+str(kernel_bias)[0]+'_max'+str(max_pooling)[0]+'_mean'+str(mean_pooling)[0]+str(pooling_size)+'-'+str(pooling_steps)+'-kf'+kernel_function+str(kernel_thresholding)
        if validation_loss != loss_function:
            outname +='vl'+validation_loss[:2]+validation_loss[max(2,len(loss_function)-2):]
        if self.hot_start:
            self.outname += '-hot'
        if self.warm_start:
            self.outname += '-warm'
        if self.shift_sequence > 0:
            self.outname += 'shs'+str(self.shift_sequence)
        if self.reverse_sign:
            self.outname += 'rs'
        
        if gapped_convs:
            self.outname += '_gp'+'-'.join(np.array(gapped_convs[0], dtype = str)).replace(',', '-').replace(' ', '').replace('[','').replace(']','').replace('(','').replace(')','')
            if len(gapped_convs) > 0:
                glist = ['k','g','n','s']
                for gl in range(1,len(gapped_convs)):
                    for g in range(4):
                        if gapped_convs[gl][g] != gapped_convs[0][g]:
                            self.outname += glist[g]+str(gapped_convs[gl][g])
        
        if self.dilated_convolutions > 0:
            self.outname += '_dc'+str(dilated_convolutions)+'i'+str(conv_increase).strip('0').strip('.')+'d'+str(dilations).replace(' ', '').replace(',', '-').replace('[','').replace(']','').replace('(','').replace(')','')+'s'+str(strides).replace(' ', '').replace(',', '-').replace('[','').replace(']','').replace('(','').replace(')','')

            if dilations is None:
                dilations = np.ones(self.dilated_convolutions,dtype = int)
            elif isinstance(dilations, int):
                dilations = np.ones(self.dilated_convolutions,dtype = int) * dilations
            else:
                dilations = np.array(dilations)

            if strides is None:
                strides = np.ones(self.dilated_convolutions,dtype = int)
            elif isinstance(strides, int):
                strides = np.ones(self.dilated_convolutions,dtype = int) * strides
            else:
                strides = np.array(strides)
            
            self.outname+='l'+str(self.l_dilkernels)
            if self.dilmax_pooling:
                self.outname += 'p'+str(self.dilmax_pooling)[0]+str(self.dilmean_pooling)[0]
            if self.dilpooling_size != self.pooling_size and self.dilpooling_size is not None:
                self.outname += str(self.dilpooling_size)
            if self.dilpooling_steps != self.pooling_steps and self.dilpooling_steps is not None:
                self.outname += str(self.dilpooling_steps)
            if self.dilpooling_residual > 0:
                self.outname += 'r'+str(self.dilpooling_residual)
        
        if self.embedding_convs > 0:
            self.outname += 'ec'+str(embedding_convs)
        
        if self.n_transformer >0:
            self.outname += 'trf'+str(sum_attention)[0]+str(n_transformer)+'h'+str(n_distattention)+'-'+str(dim_distattention)
        
        elif n_attention > 0:
            self.outname += 'at'+str(n_attention)+'h'+str(n_distattention)+'-'+str(dim_distattention)+'m'+str(maxpool_attention)
            if self.dim_embattention is not None:
                self.outname += 'v'+str(self.dim_embattention)
        
        if self.transformer_convolutions > 0:
            self.outname += '_tc'+str(transformer_convolutions)+'d'+str(trconv_dim)+'d'+str(trdilations).replace(' ', '').replace(',', '-').replace('[','').replace(']','').replace('(','').replace(')','')+'s'+str(trstrides).replace(' ', '').replace(',', '-').replace('[','').replace(']','').replace('(','').replace(')','')
            
            if trpooling_residual > 0:
                self.outname += 'r'+str(trpooling_residual)
            
            if trdilations is None:
                trdilations = np.ones(self.transformer_convolutions,dtype = int)
            elif isinstance(trdilations, int):
                trdilations = np.ones(self.transformer_convolutions,dtype = int) * trdilations
            else:
                trdilations = np.array(trdilations)

            if trstrides is None:
                trstrides = np.ones(self.transformer_convolutions,dtype = int)
            elif isinstance(strides, int):
                trstrides = np.ones(self.transformer_convolutions,dtype = int) * trstrides
            else:
                trstrides = np.array(trstrides)
        
            self.outname+='l'+str(self.l_trkernels)
            if self.trmax_pooling:
                self.outname += 'p'+str(self.trmax_pooling)[0]+str(self.trmean_pooling)[0]
            if self.trpooling_size != self.pooling_size and self.trpooling_size is not None:
                self.outname += str(self.dilpooling_size)
            if self.trpooling_steps != self.pooling_steps and self.trpooling_steps is not None:
                self.outname += str(self.trpooling_steps)
        
        
        if nfc_layers > 0:
            self.outname +='_nfc'+str(nfc_layers)+fc_function+'r'+str(nfc_residuals)
        if interaction_layer:
            self.outname += '_intl'+str(interaction_layer)[0]
        elif self.neuralnetout > 0:
            self.outname += 'nno'+str(neuralnetout)
        if l1reg_last > 0:
            self.outname += '_l1'+str(l1reg_last)
        if l2reg_last > 0:
            self.outname += '_l2'+str(l2reg_last)
        if l1_kernel > 0:
            self.outname += '_l1k'+str(l2reg_last)
        if dropout > 0.:
            self.outname += '_do'+str(dropout)
        if batch_norm:
            self.outname += '_bno'+str(batch_norm)[0]
        
        self.outname += '_tr'+str(lr)+'-bs'+str(batchsize)+optimizer+str(optim_params).replace('(', '-').replace(')', '').replace(',', '-').replace(' ', '')
        
        if kernel_lr != lr and kernel_lr is not None:
            self.outname+='-ka'+str(kernel_lr)
        if adjust_lr != 'None':
            self.outname+='-'+str(adjust_lr)[:3]
        
        print('ALL file names', self.outname)
        ### write all model settings into a file
        obj = open(self.outname+'_model_params.dat', 'w')
        for key in self.__dict__:
            if str(key) == 'fixed_kernels' and self.__dict__[key] is not None:
                obj.write(key+' : '+str(len(self.__dict__[key]))+'\n')
            else:
                obj.write(key+' : '+str(self.__dict__[key])+'\n')
        obj.close()
        
        # set learning_rate reduce or increase learning rate for kernels by hand
        if self.kernel_lr is None:
            self.kernel_lr = lr
        
        
        currdim = self.n_features
        print('In features', currdim, self.l_seqs)
        # initialize convolutional layer and compute new feature dimension and length of sequence
        if self.num_kernels > 0:
            self.convolutions = nn.Conv1d(self.n_features, self.num_kernels, kernel_size = self.l_kernels, bias = self.kernel_bias)
            currdim = np.copy(self.num_kernels)
        
        if self.fixed_kernels is not None:
            currdim += len(self.fixed_kernels)

        # Length of sequence is also changed if fixed_kernels are provided to the model
        # Model either needs to have convolutions in first layer or use the fixed kernels
        currlen = int((self.l_seqs - (self.l_kernels -1))/1.)
        print('Convolutions', currdim, currlen)

        ## a function that multiplies every kernel and fixed kernel with its own value and subtracts a bias, necessary for fixed kernels which do not come with a bias. Bias needs to be learned for sparsity    
        modellist = OrderedDict()
        if self.kernel_thresholding > 0:
            modellist['Kernelthresh'] = Kernel_linear(currdim, self.kernel_thresholding)
        
        # Non-linear conversion of kernel output
        #self.kernel_function = func_dict[kernel_function]
        modellist[kernel_function+'0'] = func_dict[kernel_function]
        
        # Max and mean pooling layers
        if self.max_pooling or self.mean_pooling:
            self.player = pooling_layer(max_pooling, mean_pooling, pooling_size=self.pooling_size, stride=pooling_steps)
            modellist['Pooling'] = self.player
            currlen = int(1. + (currlen - (self.pooling_size-1)-1)/self.pooling_steps)
            currdim = (int(self.max_pooling) + int(self.mean_pooling)) * currdim
            print('Pooling', currdim, currlen)
        
        # If dropout given, also introduce dropout after every layer
        if self.dropout > 0:
            modellist['Dropout_kernel'] = nn.Dropout(p=self.dropout)
        self.modelstart = nn.Sequential(modellist)
        
        # Initialize additional convolutional layers
        if self.dilated_convolutions > 0:
            if self.dilmax_pooling:
                dilmaxpooling_size = self.dilpooling_size
            else:
                dilmaxpooling_size = 0
            if self.dilmean_pooling:
                dilmeanpooling_size = self.dilpooling_size
            else:
                dilmeanpooling_size = 0
            self.convolution_layers = Res_Conv1d(currdim, currlen, currdim, self.l_dilkernels, self.dilated_convolutions, kernel_increase = self.conv_increase, max_pooling = dilmaxpooling_size, mean_pooling=dilmeanpooling_size, residual_after = self.dilpooling_residual, activation_function = kernel_function, strides = strides, dilations = dilations, bias = True, dropout = dropout)
            currdim, currlen = self.convolution_layers.currdim, self.convolution_layers.currlen

            print('2nd convolutions', currdim, currlen)
            
        if self.embedding_convs > 0:
            # Reduces dimension of cnn output before provided to transfomer
            self.embedding_convolutions = nn.Conv1d(currdim, self.embedding_convs, kernel_size = 1, bias = False)
            currdim = self.embedding_convs  
            print('Convolution before attention', currdim, currlen)
        
        if self.n_transformer > 0:
            # if self.duplicate_head:
            self.layer_norm = nn.LayerNorm(currdim*self.n_distattention)
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=currdim*self.n_distattention, nhead=self.n_distattention, dim_feedforward = int(self.dim_distattention *currdim), batch_first=True)                               
            self.transformer = nn.TransformerEncoder(self.encoder_layer, self.n_transformer, norm=self.layer_norm)
            currdim = currdim*self.n_distattention
            print('Transformer', currdim, currlen)
            if self.sum_attention:
                currdim = int(currdim/self.n_distattention)
                print('Sum multi-head attention', currdim, currlen)
        elif self.n_attention > 0: # Number of attentionblocks

            distattention = OrderedDict()
            for na in range(self.n_attention):
                distattention['Mheadattention'+str(na)] = MyAttention_layer(currdim, int(self.dim_distattention *currdim), self.n_distattention, dim_values = self.dim_embattention, dropout = dropout, bias = False, residual = True, sum_out = self.sum_attention, combine_out = True, batchnorm = self.batch_norm)
                if self.dim_embattention is None:
                    currdim = int(self.dim_distattention *currdim)
                else:
                    currdim = self.dim_embattention
                if self.maxpool_attention > 0:
                    if int(np.floor(1. + (currlen - (maxpool_attention-1)-1)/maxpool_attention)) > 0:
                        distattention['Maxpoolattention'+str(na)]= pooling_layer(True, False, pooling_size= maxpool_attention, stride=maxpool_attention)
                        currlen = int(np.floor(1. + (currlen - (maxpool_attention-1)-1)/maxpool_attention))
            self.distattention = nn.Sequential(distattention)
            print('Attention', currdim, currlen)
            
        
        
        # convolutional layers and pooling layers to reduce the dimension after transformer detected kernel interactions
        if self.transformer_convolutions > 0:
            if self.trconv_dim is None:
                self.trconv_dim = currdim
            if self.trmax_pooling:
                trmaxpooling_size = self.trpooling_size
            else:
                trmaxpooling_size = 0
            if self.trmean_pooling:
                trmeanpooling_size = self.trpooling_size
            else:
                trmeanpooling_size = 0
            self.trconvolution_layers = Res_Conv1d(currdim, currlen, self.trconv_dim, self.l_trkernels, self.transformer_convolutions, kernel_increase = self.conv_increase, max_pooling = trmaxpooling_size, mean_pooling=trmeanpooling_size, residual_after = self.trpooling_residual, activation_function = kernel_function, strides = trstrides, dilations = trdilations, bias = True, dropout = dropout)
            currdim, currlen = self.trconvolution_layers.currdim, self.trconvolution_layers.currlen
            print('Convolution after attention', currdim, currlen)
        
        elif self.trmax_pooling or self.trmean_pooling and self.transformer_convolutions == 0:
            self.trconvolution_layers = pooling_layer(self.trmax_pooling, self.trmean_pooling, pooling_size=self.trpooling_size, stride=self.trpooling_steps)
            currlen = int(1. + (currlen - (self.trpooling_size-1)-1)/self.trpooling_steps)
            currdim = (int(self.trmax_pooling) + int(self.trmean_pooling)) * currdim
        
        # Initialize gapped convolutions
        if self.gapped_convs is not None:
            cdim = 0
            modellist = []
            for g, gap_c in enumerate(self.gapped_convs):
                modellist.append(gap_conv(currdim, gap_c[2], gap_c[0], gap_c[1], stride=gap_c[3]))
                cdim += gap_c[2] * int(1.+(currlen - (gap_c[0]*2+gap_c[1]))/gap_c[3])
            currdim = cdim
            self.gapped_convolutions = parallel_module(modellist)
        else:
            # If gapped convolutions is not used, the output is flattened
            currdim = currdim * currlen
        
        print('Before FCL', currdim)
        # Initialize batch norm before fully connected layers or before classification layer
        self.NNreg = False
        if self.batch_norm:
            self.NNred = True
            self.reglayer = nn.BatchNorm1d(currdim)
        
        # Initialize fully connected layers
        if self.nfc_layers > 0:
            self.nfcs = Res_FullyConnect(currdim, outdim = currdim, n_classes = None, n_layers = self.nfc_layers, layer_widening = layer_widening, batch_norm = self.batch_norm, dropout = self.dropout, activation_function = self.fc_function, residual_after = self.nfc_residuals, bias = True)
            
        # Interaction layer multiplies every features with each other and accounts for non-linearities explicitly, often dimension gets to big to use. Parameters are of dimension d + d*(d-1)/2
        
        classifier = OrderedDict()
        if self.interaction_layer:
            classifier['Interaction_layer'] = interaction_module(currdim, n_classes, classes = self.outclass)
        elif self.neuralnetout > 0:
            classifier['Neuralnetout'] = Res_FullyConnect(currdim, outdim = 1, n_classes = n_classes, n_layers = self.neuralnetout, layer_widening = 1.1, batch_norm = self.batch_norm, dropout = self.dropout, activation_function = self.fc_function, residual_after = 1)
        else:
            classifier['Linear'] = nn.Linear(currdim, n_classes)
        
        if self.outclass == 'Class':
            classifier['Sigmoid'] = nn.Sigmoid()
        elif self.outclass == 'Multi_class':
            classifier['Softmax'] = nn.Softmax()
        elif self.outclass == 'Complex':
            classifier['Complex'] = Complex(n_classes)
        
        self.classifier = nn.Sequential(classifier)
        
   
    # The prediction after training are performed on the cpu
    def predict(self, X, pwm_out = None, mask = None, device = None):
        if self.fixed_kernels is not None:
            if pwm_out is None:
                pwm_out = pwm_scan(X, self.fixed_kernels, targetlen = self.l_kernels, motif_cutoff = self.motif_cutoff)
            pwm_out = torch.Tensor(pwm_out)
            
        predout = batched_predict(self, X, pwm_out =pwm_out, mask = mask, device = device, batchsize = self.batchsize)
        return predout
    
    def forward(self, x, xadd = None, mask = None, location = 'None'):
        # Forward pass through all the initialized layers
        if self.num_kernels > 0:
            pred = self.convolutions(x)
            if xadd is not None:
                # add pre_computed features from pwms to pred
                pred = torch.cat((pred, xadd), dim = -2)
        else:
            pred = xadd
        
        if mask is not None:
            if self.kernel_bias:
                if mask < self.convolutions.bias.size(dim=0):
                    pred[:,mask,:] = self.convolutions.bias[mask]
                else:
                    pred[:,mask,:] = 0
            else:
                pred[:,mask,:] = 0
        if location == '0':    
            return torch.flatten(pred, start_dim = 1)
        
        pred = self.modelstart(pred)
        
        if location == '1':
            return torch.flatten(pred, start_dim = 1)
        
        if self.dilated_convolutions > 0:
            pred = self.convolution_layers(pred)
        
        if location == '2':
            return torch.flatten(pred, start_dim = 1)
        
        if self.embedding_convs > 0:
            pred = self.embedding_convolutions(pred)
        
        if self.n_transformer >0:
            pred = torch.transpose(pred, -1, -2)
            pred = torch.flatten(pred.unsqueeze(2).expand(-1,-1,self.n_distattention,-1),start_dim = -2)
            pred = self.transformer(pred)
            pred = torch.transpose(pred, -1, -2)
            if self.sum_attention:
                pred = torch.sum(pred.view(pred.size(dim = 0), self.n_distattention, -1,pred.size(dim = -1)),dim = 1)    
        
        elif self.n_distattention > 0:
            pred = self.distattention(pred)
        
        if location == '3':
            return torch.flatten(pred, start_dim = 1)
        
        if self.transformer_convolutions > 0 or self.trmax_pooling or self.trmean_pooling:
            pred = self.trconvolution_layers(pred)
        
        if location == '4':
            return torch.flatten(pred, start_dim = 1)
        
        if self.gapped_convs is not None:
            pred = self.gapped_convolutions(pred)
        else:
            pred = torch.flatten(pred, start_dim = 1, end_dim = -1)
        
        if location == '5':
            return torch.flatten(pred, start_dim = 1)
        
        if self.NNreg:
            pred = self.reglayer(pred)
        if self.nfc_layers > 0:
            pred = self.nfcs(pred)
        
        if location == '-1' or location == '6':
            return torch.flatten(pred, start_dim = 1)
        
        pred = self.classifier(pred)
        return pred
    
    
    def fit(self, X, Y, XYval = None, sample_weights = None):
        self.saveloss = fit_model(self, X, Y, XYval = XYval, sample_weights = sample_weights, loss_function = self.loss_function, validation_loss = self.validation_loss, batchsize = self.batchsize, device = self.device, optimizer = self.optimizer, optim_params = self.optim_params, verbose = self.verbose, lr = self.lr, kernel_lr = self.kernel_lr, hot_start = self.hot_start, warm_start = self.warm_start, outname = self.outname, adjust_lr = self.adjust_lr, patience = self.patience, init_adjust = self.init_adjust, keepmodel = self.keepmodel, load_previous = self.load_previous, write_steps = self.write_steps, checkval = self.checkval, writeloss = self.writeloss, init_epochs = self.init_epochs, epochs = self.epochs, l1reg_last = self.l1reg_last, l2_reg_last = self.l2reg_last, l1_kernel = self.l1_kernel, reverse_sign = self.reverse_sign, shift_back = self.shift_sequence, **self.kwargs)
        





if __name__ == '__main__':
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    
    # Delimiter for values in output file
    delimiter = ','
    if '--delimiter' in sys.argv:
        delimiter = sys.argv[sys.argv.index('--delimiter')+1]
    
    # Whether to assign a different regional channel for each input file
    aregion = True
    if '--regionless' in sys.argv:
        aregion = False
    
    # Whether sequences in each region should be aligned to right side, for example if sequences have different lengths and there could be a positional signal at the end of sequences
    mirror = False
    if '--realign_input' in sys.argv:
        mirror = True
    
    # Whether to combine input files into a single input or keep them split as input for multiple networks
    combinput = True
    if '--combine_network' in sys.argv:
        combinput = False
    
    select_track = None
    if '--select_tracks' in sys.argv:
        select_track = sys.argv[sys.argv.index('--select_tracks')+1]
        if ',' in select_track:
            select_track = select_track.split(',')
        else:
            select_track = [select_track]
    
    X, Y, names, features, experiments = readin(inputfile, outputfile, delimiter = delimiter,return_header = True, assign_region = aregion, mirrorx = mirror, combinex = combinput, select_track = select_track)
    
    
    if ',' in inputfile:
        inputfiles = inputfile.split(',')
        inputfile = inputfiles[0]
        for inp in inputfiles[1:]:
            inputfile = create_outname(inp, inputfile, lword = 'and')
            
    outname = create_outname(inputfile, outputfile) 
    if '--regionless' in sys.argv:
        outname += '-rgls'
    print(outname)
    
    # Parameter dictionary for initializing the cnn
    params = {}
    
    if '--outdir' in sys.argv:
        outname = sys.argv[sys.argv.index('--outdir')+1] + os.path.split(outname)[1]
    
    if '--addname' in sys.argv:
        outname += '_'+sys.argv[sys.argv.index('--addname')+1]
    
    
    # reads in file that had locations and type of mutations, currently not able to handle mirror=True or multiple inputs, needs to be adjusted to common vcf file format
    weights = None
    if '--mutation_file' in sys.argv:
        mutfile = sys.argv[sys.argv.index('--mutation_file')+1]
        X,Y,names,experiments,weights = read_mutationfile(mutfile,X,Y,names,experiments)
        if sys.argv[sys.argv.index('--mutation_file')+2] != 'weighted':
            weights = None
        outname = create_outname(mutfile, outname+'.dat', lword = 'with')
        print(outname)

    if '--crossvalidation' in sys.argv:
        folds = check(sys.argv[sys.argv.index('--crossvalidation')+1])
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
    trainset, testset, valset = create_sets(len(X), folds, fold, Yclass = Yclass, genenames = names)
    print('Train', len(trainset))
    print('Test', len(testset), testset)
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
        if '+' in parameters:
            parameters = parameters.split('+')
        elif os.path.isfile(parameters):
            parameterfile = parameters.replace('model_params.dat', 'parameter.pth')
            obj = open(parameters,'r').readlines()
            parameters = []
            for l, line in enumerate(obj):
                if line[0] != '_' and line[:7] != 'outname':
                    parameters.append(line.strip().replace(' ', ''))
            train_model = False
        else:
            parameters = [parameters]
        
        params['device'] = get_device()
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
        params['n_features'], params['l_seqs'], params['n_classes']= np.shape(X)[-2], np.shape(X)[-1], np.shape(Y)[-1]
        model = cnn(**params)
        
    if weights is not None:
        weights = weights[trainset]
    
    translate_dict = None
    exclude_dict = []
    if '--load_parameters' in sys.argv:
        parameterfile = sys.argv[sys.argv.index('--load_parameters')+1]
        translate_params = sys.argv[sys.argv.index('--load_parameters')+2] # dictionary with names from current and alternatie models
        exclude_params = sys.argv[sys.argv.index('--load_parameters')+3] # list with names from loaded model that should be ignored when loading
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
        load_parameters(model, parameterfile, translate_dict = translate_dict, exclude = exclude_dict)
    
    # pre-loaded pwms are loaded into the model as convolutional weights
    if pwmusage == 'initialize':
        init_kernels = np.concatenate([pwmset(pwm, params['l_kernels']) for pwm in pwms], axis = 0)
        init_kernels = OrderedDict({ 'convolutions.weight': torch.Tensor(init_kernels[np.argsort(-np.sum(init_kernels, axis = (1,2)))])})
        load_parameters(model, init_kernels, allow_reduction = True)
        
    if train_model:
        model.fit(X[trainset], Y[trainset], XYval = [X[valset], Y[valset]], sample_weights = weights)
    
    Y_pred = model.predict(X[testset])
    
    if '--norm2output' in sys.argv:
        Y *= outnorm
        Y_pred *= outnorm[testset]
        
    elif '--norm2outputclass' in sys.argv:
        Y *= outnorm
        Y_pred *= outnorm
    
    if model.outname != outname:
        outname = model.outname
    
    if '--save_predictions' in sys.argv:
        print('SAVED', outname+'_pred.txt')
        np.savetxt(outname+'_pred.txt', np.append(names[testset][:, None], Y_pred, axis = 1), fmt = '%s')

        
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
        
        motifimpact, impact_direction = compute_importance(model, X[trainset], Y[trainset], activation_measure = activation_measure, pwm_in = pwm_in, normalize = False)
        
        np.savetxt(outname+'_kernel_importance.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), motifimpact], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(experiments))
        np.savetxt(outname+'_kernel_impact.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), impact_direction], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(experiments))
        
        # Due to early stopping the importance scores from training and testing should be the same
        # However training importance provide impacts of patterns from overfitting
        # Test set on the other hand could lack the statistical power
        if '--testset_importance' in sys.argv:
            motifimpact, impact_direction = compute_importance(model, X[testset], Y[testset], activation_measure = activation_measure, pwm_in = model.pwm_out, normalize = False)
            np.savetxt(outname+'_kernel_importance_test.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), motifimpact], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(experiments))
            np.savetxt(outname+'_kernel_impact_test.dat', np.concatenate([motifnames.reshape(-1,1), iupacmotifs.reshape(-1,1), impact_direction], axis = 1).astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(experiments))
            
            


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
         

    
    
    








