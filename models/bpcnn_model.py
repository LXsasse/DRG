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
from modules import parallel_module, gap_conv, interaction_module, pooling_layer, correlation_loss, correlation_both, cosine_loss, cosine_both, zero_loss, Complex, Expanding_linear, Res_FullyConnect, Residual_convolution, Res_Conv1d, MyAttention_layer, Kernel_linear, loss_dict, func_dict, Padded_Conv1d, final_convolution, RC_Conv1d
from train import pwmset, pwm_scan, batched_predict
from train import fit_model
from compare_expression_distribution import read_separated
from output import add_params_to_outname







# flexible Convolutional neural network architecture
class bpcnn(nn.Module):
    def __init__(self, loss_function = 'MSE', validation_loss = None, n_features = None, reverse_complement = False, n_classes = 1, l_seqs = None, l_out = None, num_kernels = 0, kernel_bias = True, fixed_kernels = None, motif_cutoff = None, l_kernels = 7, kernel_function = 'GELU', warm_start = False, hot_start = False, hot_alpha=0.01, kernel_thresholding = 0, max_pooling = False, mean_pooling = False, weighted_pooling = False, pooling_size = None, pooling_steps = None, net_function = 'GELU', dilated_convolutions = 0, strides = 1, conv_increase = 1., dilations = 1, l_dilkernels = None, dilmax_pooling = 0, dilmean_pooling = 0, dilweighted_pooling= 0, dilpooling_residual = 1, dilresidual_entire = False, dilresidual_concat = False, gapped_convs = None, gapconv_residual = True, gapconv_pooling = False, embedding_convs = 0, n_transformer = 0, n_attention = 0, n_interpolated_conv = 0, n_distattention = 0, dim_distattention=2.5, dim_embattention = None, attentionmax_pooling = 0, attentionweighted_pooling = 0, attentionconv_pooling = 1, attention_multiplier = 0.5, sum_attention = False, transformer_convolutions = 0, trdilations = 1, trstrides = 1, l_trkernels = None, trconv_dim = None, trmax_pooling = 0, trmean_pooling = 0, trweighted_pooling = 0, trpooling_residual = 1, trresidual_entire = False, final_kernel = 1, final_strides = 1, predict_from_dist = False, dropout = 0., batch_norm = False, l1_kernel = 0, l2reg_last = 0., l1reg_last = 0., shift_sequence = None, random_shift = False, reverse_sign = False, smooth_onehot = 0, epochs = 1000, lr = 1e-2, kernel_lr = None, adjust_lr = 'F', batchsize = None, patience = 25, outclass = 'Linear', outname = None, optimizer = 'Adam', optim_params = None, verbose = True, checkval = True, init_epochs = 3, writeloss = True, write_steps = 10, device = 'cpu', load_previous = True, init_adjust = True, seed = 101010, keepmodel = False, generate_paramfile = True, add_outname = True, restart = False, masks = None, nmasks = None, augment_representation = None, aug_kernel_size = None, aug_conv_layers = 1, aug_loss_masked = True, aug_loss = None, aug_loss_mix = None, **kwargs):
        super(bpcnn, self).__init__()
        
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
        self.reverse_complement = reverse_complement # whether to use reverse complement in first CNN
        self.l_seqs = l_seqs # length of padded sequences
        self.l_out = l_out # number of output regions per sequence
        if self.l_out is None:
            self.l_out = l_seqs
        self.n_classes = n_classes # output classes to predict
        
        self.shift_sequence = shift_sequence # During training sequences that are shifted by 'shift_sequence' positions will be added # either integer or array of shifts
        paddy = 0
        if shift_sequence is not None:
            if isinstance(shift_sequence,int):
                paddy = shift_sequence
            else:
                paddy = np.amax(shift_sequence)
        self.random_shift = random_shift # a random shift applies a random number in shift_sequence to the data in each step         
        
        self.reverse_sign = reverse_sign # During training, the sign of the input and the output will be shifted. This mirror image of the data can be helpful with training
        self.smooth_onehot = smooth_onehot # adds continuous values to the one hot encoding to smooth it between bases
        self.restart = restart # restart the training only with the learned kernels and reset all other parameters to random values
        
        
        if self.n_features is None or self.l_seqs is None:
            print('n_features and l_seqs need to be defined')
            sys.exit()
        
        self.adjust_lr = adjust_lr # don't ignore kernel_lr but also adjust other lrs according to the number of parameters in the layer, the location of the layer and the relationship between kernel_lr and lr
        self.kernel_lr = kernel_lr
        self.num_kernels = num_kernels  # number of learnable kernels
        self.l_kernels = l_kernels # length of learnable kernels
        self.kernel_bias = kernel_bias # Turn on/off bias of kernels
        self.kernel_function = kernel_function # Non-linear function applied to kernel outputs
        self.net_function = net_function # Non-linear function applied to other layers
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
        self.weighted_pooling = weighted_pooling
        self.pooling_size = pooling_size    # The size of the pooling window, Can span the entire sequence
        self.pooling_steps = pooling_steps # The step size of the pooling window, stepsizes smaller than the pooling window size create overlapping regions
        if self.max_pooling == False and self.mean_pooling == False and self.weighted_pooling == False:
            self.pooling_size = None
            self.pooling_steps = None
        elif self.pooling_size is None and self.pooling_steps is None:
            self.pooling_size = self.l_seqs + 2*paddy
            self.pooling_steps = self.l_seqs +2*paddy
        elif self.pooling_steps is None:
            self.pooling_steps = self.pooling_size
        elif self.pooling_size is None:
            self.pooling_size = self.pooling_steps
        
        self.dilated_convolutions = dilated_convolutions # Number of additional dilated convolutions
        self.strides = strides #Strides of additional convolutions
        self.dilations = dilations # Dilations of additional convolutions
        self.conv_increase = conv_increase # Factor by which number of additional convolutions increases in each layer
        self.dilpooling_residual = dilpooling_residual # Number of convolutions before residual is added
        self.dilresidual_entire = dilresidual_entire # if residual should be forwarded from beginning to end of dilated block
        self.dilresidual_concat = dilresidual_concat # if True the residual will be concatenated with the predictions instead of summed. 
        
        # Length of additional convolutional kernels
        if l_dilkernels is None:
            self.l_dilkernels = l_kernels
        else:
            self.l_dilkernels = l_dilkernels
        
        # Max pooling for additional convolutional layers
        self.dilmax_pooling = dilmax_pooling
        # Mean pooling for additional convolutional layers
        self.dilmean_pooling = dilmean_pooling
        if dilmean_pooling >0 or dilmax_pooling > 0:
            dilweighted_pooling = 0
        self.dilweighted_pooling = dilweighted_pooling
        
        # all to default if
        if self.dilated_convolutions == 0:
            self.strides, self.dilations = 1,1
            self.l_dilkernels, self.dilmax_pooling, self.dilmean_pooling, self.dilweighted_pooling = None, 0,0,0
        
        # If lists given, parallel gapped convolutions are initiated
        self.gapped_convs = gapped_convs # list of quadruples, first the size of the kernel left and right, second the gap, third the number of kernals, fourth the stride stride. Will generate several when parallel layers with different gapsizes if list is given.
        # Gapped convolutions are placed after maxpooling layer and then concatenated with output from previous maxpooling layer. 
        self.gapconv_residual = gapconv_residual
        self.gapconv_pooling = gapconv_pooling
        
        # reduce the dimensions of the output of the convolutional layer before giving it to the transformer layer
        self.embedding_convs = embedding_convs
        
        self.n_transformer = n_transformer # N_transformer defines how many TransformerEncoderLayer from torch will be used in TransformerEncoder (torch based module)
        self.n_attention = n_attention # this will set the number of custom attention/transformer layers that are modelled after the original
        self.n_interpolated_conv = n_interpolated_conv # uses
        
        self.n_distattention = n_distattention # defines the number of heads
        self.dim_distattention = dim_distattention # multiplicative value by which the input dimension will be increased/decreased in the key and query embedding
        ## make son that it can also be the dimension itself and not a multiplicative factor
        self.dim_embattention = dim_embattention # dimension of values
        self.sum_attention = sum_attention # if inputs are duplicated n_heads times then they will summed afterwards across all heads, more important if torch transformer is used because it increases the dimension by the multitude of the number of heads
        self.attentionmax_pooling = attentionmax_pooling # generate maxpool layer after attention layer to reduce length of input
        self.attentionweighted_pooling = attentionweighted_pooling # generate maxpool layer after attention layer to reduce length of input
        
        self.attentionconv_pooling = attentionconv_pooling # this is the stride if interpolated_conv
        self.attention_multiplier = attention_multiplier # this is the multiplier ins interpolated_conv
        
        
        self.transformer_convolutions = transformer_convolutions # Number of additional convolutions afer transformer layer
        self.trpooling_residual = trpooling_residual # Number of layers that are scipped by residual layer
        self.trresidual_entire = trresidual_entire # if residual from start of block should be added to after last convolution
        self.trstrides = trstrides #Strides of additional convolutions
        self.trdilations = trdilations # Dilations of additional convolutions
        self.trconv_dim = trconv_dim # dimensions of convolutions after transformer
        
        # Length of additional convolutional kernels
        if l_trkernels is None:
            self.l_trkernels = l_kernels
        else:
            self.l_trkernels = l_trkernels
        
        # Pooling sizes and step size for additional max pooling layers
        self.trmean_pooling = trmean_pooling
        self.trmax_pooling = trmax_pooling
        self.trweighted_pooling = trweighted_pooling
        
        # all to default if
        if self.transformer_convolutions == 0:
            self.trstrides, self.trdilations, self.trconv_dim = None, None, None
            self.l_trkernels, self.trmax_pooling, self.trmean_pooling, self.trweighted_pooling = None, 0,0,0
        if self.dilated_convolutions == 0 and self.transformer_convolutions == 0:
             self.conv_increase = 1   
        
        self.final_kernel = final_kernel
        self.final_strides = final_strides
        self.predict_from_dist = predict_from_dist
        
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
        
        self.masks = masks # location (start and end) of masks sequences
        self.nmasks = nmasks # number of masks that are randomly inserted in every training, val, and testing round per sequence
        self.augment_representation = augment_representation # location of representation which will be prvided to final deconvolution for sequence learning task
        self.aug_kernel_size = aug_kernel_size # kernel size for deconvolution of augmented sequence learning task
        self.aug_conv_layers = aug_conv_layers # number of convolutions in the final layer, if larger than one will be dilated by 2**n
        self.aug_loss_masked = aug_loss_masked # if True, the loss is only computed over the entries that were masked not if the other sequence elements were correctly reproduced. 
        self.aug_loss = aug_loss # loss function for augmentation sequence learning
        self.aug_loss_mix = aug_loss_mix # mixture value between main and augmenting task


        self.outclass = outclass # Class: sigmoid, Multi_class: Softmax, Complex: for non-linear scaling

        self.outname = outname
        if add_outname:
            ### Generate file name from all settings
            if outname is None:
                self.outname = 'CNNmodel'  # add all the other parameters
            else:
                self.outname = outname
            
            self.outname = add_params_to_outname(self.outname, self.__dict__)
            if self.verbose:
                print('ALL file names', self.outname)
    
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

        if trdilations is None:
            trdilations = np.ones(self.transformer_convolutions,dtype = int)
        elif isinstance(trdilations, int):
            trdilations = np.ones(self.transformer_convolutions,dtype = int) * trdilations
        else:
            trdilations = np.array(trdilations)

        if trstrides is None:
            trstrides = np.ones(self.transformer_convolutions,dtype = int)
        elif isinstance(trstrides, int):
            trstrides = np.ones(self.transformer_convolutions,dtype = int) * trstrides
        else:
            trstrides = np.array(trstrides)

        if generate_paramfile:
            obj = open(self.outname+'_model_params.dat', 'w')
            for key in self.__dict__:
                if str(key) == 'fixed_kernels' and self.__dict__[key] is not None:
                    obj.write(key+' : '+str(len(self.__dict__[key]))+'\n')
                else:
                    obj.write(key+' : '+str(self.__dict__[key])+'\n')
            obj.close()
        self.generate_paramfile = generate_paramfile
        # set learning_rate reduce or increase learning rate for kernels by hand
        if self.kernel_lr is None:
            self.kernel_lr = lr
        
        
        currdim = self.n_features
        currlen = self.l_seqs + 2*paddy
        if self.verbose:
            print('In features', currdim, currlen)
        # initialize convolutional layer and compute new feature dimension and length of sequence
        if self.num_kernels > 0:
            #self.convolutions = nn.Conv1d(self.n_features, self.num_kernels, kernel_size = self.l_kernels, bias = self.kernel_bias, padding = int(self.l_kernels/2))
            self.convolutions = Padded_Conv1d(self.n_features, self.num_kernels, kernel_size = self.l_kernels, bias = self.kernel_bias, padding = [int(self.l_kernels/2)-int(self.l_kernels%2==0), int(self.l_kernels/2)], reverse_complement = reverse_complement)
            currdim = np.copy(self.num_kernels)
        
        if self.fixed_kernels is not None:
            currdim += len(self.fixed_kernels)

        # Length of sequence is also changed if fixed_kernels are provided to the model
        # Model either needs to have convolutions in first layer or use the fixed kernels
        #padding resolves this currlen = int((self.l_seqs - (self.l_kernels -1))/1.)
        if self.verbose:
            print('Convolutions', currdim, currlen)

        ## a function that multiplies every kernel and fixed kernel with its own value and subtracts a bias, necessary for fixed kernels which do not come with a bias. Bias needs to be learned for sparsity    
        modellist = OrderedDict()
        if self.kernel_thresholding > 0:
            modellist['Kernelthresh'] = Kernel_linear(currdim, self.kernel_thresholding)
        
        # Non-linear conversion of kernel output
        modellist[kernel_function+'0'] = func_dict[kernel_function]
        
        # Max and mean pooling layers
        if self.max_pooling or self.mean_pooling:
            self.player = pooling_layer(max_pooling, mean_pooling, weighted_pooling, pooling_size=self.pooling_size, stride=self.pooling_steps, padding = int(np.ceil((self.pooling_size-currlen%self.pooling_steps)/2))*int(currlen%self.pooling_steps>0))
            modellist['Pooling'] = self.player
            currlen = int(np.ceil(currlen/self.pooling_steps))
            currdim = max(1,int(self.max_pooling) + int(self.mean_pooling)) * currdim
            if self.verbose:
                print('Pooling', currdim, currlen)
        
        # If dropout given, also introduce dropout after every layer
        if self.dropout > 0:
            modellist['Dropout_kernel'] = nn.Dropout(p=self.dropout)
        self.modelstart = nn.Sequential(modellist)
        
        # Initialize additional convolutional layers
        if self.dilated_convolutions > 0:
            self.convolution_layers = Res_Conv1d(currdim, currlen, currdim, self.l_dilkernels, self.dilated_convolutions, kernel_increase = self.conv_increase, max_pooling = dilmax_pooling, mean_pooling=dilmean_pooling, weighted_pooling=dilweighted_pooling, residual_after = self.dilpooling_residual, activation_function = net_function, strides = strides, dilations = dilations, bias = True, dropout = dropout, residual_entire = self.dilresidual_entire, concatenate_residual = dilresidual_concat)
            currdim, currlen = self.convolution_layers.currdim, self.convolution_layers.currlen
            if self.verbose:
                print('2nd convolutions', currdim, currlen)
            
        if self.embedding_convs > 0:
            # Reduces dimension of cnn output before provided to transfomer
            self.embedding_convolutions = nn.Conv1d(currdim, self.embedding_convs, kernel_size = 1, bias = False)
            currdim = self.embedding_convs  
            if self.verbose:
                print('Convolution before attention', currdim, currlen)
        
        if self.n_transformer > 0:
            # if self.duplicate_head:
            self.layer_norm = nn.LayerNorm(currdim*self.n_distattention)
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=currdim*self.n_distattention, nhead=self.n_distattention, dim_feedforward = int(self.dim_distattention *currdim), batch_first=True)                               
            self.transformer = nn.TransformerEncoder(self.encoder_layer, self.n_transformer, norm=self.layer_norm)
            currdim = currdim*self.n_distattention
            if self.verbose:
                print('Transformer', currdim, currlen)
            if self.sum_attention:
                currdim = int(currdim/self.n_distattention)
                if self.verbose:
                    print('Sum multi-head attention', currdim, currlen)
        
        # Long-range interpolated convolution to capture distal interactions
        elif self.n_interpolated_conv > 0:
            if self.dim_embattention is None:
                self.dim_embattention = currdim
            if self.n_distattention == 0:
                self.n_distattention = 16
            
            self.distattention = Res_Conv1d(currdim, currlen, self.dim_embattention, self.n_distattention, self.n_interpolated_conv, kernel_increase = self.dim_distattention, max_pooling = attentionmax_pooling, mean_pooling=0, weighted_pooling=attentionweighted_pooling, residual_after = 1, residual_same_len = False, activation_function = net_function, strides = attentionconv_pooling, dilations = 2, bias = True, dropout = self.dropout, batch_norm = self.batch_norm, act_func_before = False, residual_entire = False, concatenate_residual = self.sum_attention, linear_layer = self.sum_attention, long_conv = True, interpolation = 'linear', weight_function = 'linear', multiplier=attention_multiplier)

            currlen = self.distattention.currlen
            currdim = self.distattention.currdim
            if self.verbose:
                print('interpolated convolutions', currdim, currlen)
        
        elif self.n_attention > 0: # Number of attentionblocks
            distattention = OrderedDict()
            for na in range(self.n_attention):
                distattention['Mheadattention'+str(na)] = MyAttention_layer(currdim, int(self.dim_distattention *currdim), self.n_distattention, dim_values = self.dim_embattention, in_len= currlen, dropout = dropout, bias = False, residual = True, sum_out = self.sum_attention, batchnorm = self.batch_norm)
                if self.dim_embattention is None:
                    currdim = int(self.dim_distattention *currdim)
                else:
                    currdim = self.dim_embattention
                if self.attentionmax_pooling > 0:
                    if int(np.floor(1. + (currlen - (attentionmax_pooling-1)-1)/attentionmax_pooling)) > 0:
                        distattention['Maxpoolattention'+str(na)]= pooling_layer(True, False, False, pooling_size= attentionmax_pooling, stride=attentionmax_pooling,padding = int(np.ceil((attentionmax_pooling-currlen%attentionmax_pooling)/2))*int(currlen%attentionmax_pooling>0))
                        currlen = int(np.floor(1. + (currlen - (attentionmax_pooling-1)-1)/attentionmax_pooling))
                        
                if self.attentionweighted_pooling > 0:
                    if int(np.floor(1. + (currlen - (attentionweighted_pooling-1)-1)/attentionweighted_pooling)) > 0:
                        distattention['weightedpoolattention'+str(na)]= pooling_layer(False, False, True, pooling_size= attentionweighted_pooling, stride=attentionweighted_pooling, padding = int(np.ceil((attentionweighted_pooling-currlen%attentionweighted_pooling)/2))*int(currlen%attentionweighted_pooling>0))
                        currlen = int(np.floor(1. + (currlen - (attentionweighted_pooling-1)-1)/attentionweighted_pooling))
            self.distattention = nn.Sequential(distattention)
            if self.verbose:
                print('Attention', currdim, currlen)
            
        
        
        # convolutional layers and pooling layers to reduce the dimension after transformer detected kernel interactions
        if self.transformer_convolutions > 0:
            if self.trconv_dim is None:
                self.trconv_dim = currdim
            
            self.trconvolution_layers = Res_Conv1d(currdim, currlen, self.trconv_dim, self.l_trkernels, self.transformer_convolutions, kernel_increase = self.conv_increase, max_pooling = trmaxpooling_size, mean_pooling=trmeanpooling_size, residual_after = self.trpooling_residual, activation_function = net_function, strides = trstrides, dilations = trdilations, bias = True, dropout = dropout, residual_entire = self.trresidual_entire)
            currdim, currlen = self.trconvolution_layers.currdim, self.trconvolution_layers.currlen
            if self.verbose:
                print('Convolution after attention', currdim, currlen)
        
        elif (self.trmax_pooling >0  or self.trmean_pooling >0 or self.trweighted_pooling > 0) and self.transformer_convolutions == 0:
            trpooling_size = max(max(self.trmax_pooling,self.trmean_pooling),self.trweighted_pooling)
            self.trconvolution_layers = pooling_layer(self.trmax_pooling>0, self.trmean_pooling>0, self.trweighted_pooling > 0, pooling_size=trpooling_size, stride=trpooling_size, padding = np.ceil((trpooling_size-currlen%trpooling_size)/2)*int(currlen%trpooling_size>0))
            currlen = int(np.ceil(currlen/self.trpooling_size))
            currdim = (int(self.trmax_pooling>0) + int(self.trmean_pooling>0)) * currdim
        
        # Initialize gapped convolutions
        if self.gapped_convs is not None:
            cdim = 0
            modellist = []
            for g, gap_c in enumerate(self.gapped_convs):
                modellist.append(gap_conv(currdim, currlen, gap_c[2], gap_c[0], gap_c[1], stride=gap_c[3], batch_norm = self.batch_norm, dropout = self.dropout, residual = self.gapconv_residual, pooling= self.gapconv_pooling, edge_effect = True, activation_function = net_function))
                cdim += gap_c[2]
                currlen = modellist[-1].out_len
            currdim = cdim
            self.gapped_convolutions = parallel_module(modellist, flatten = False)
        
        if self.verbose:
            print(currdim, currlen)
            print('outclasses', n_classes, l_out)
        
        if l_out > currlen:
            print(currlen, 'less than l_out\nMake sure that currlen is larger than l_out')
            sys.exit()
        cutedges = None
        if currlen > l_out:
            cutedges = [int(np.floor((currlen-l_out)/2)), int(np.ceil((currlen-l_out)/2))]
            print(currlen, 'larger than l_out\nMake sure the outputs are correctly aligned to the input regions if edges are cut', cutedges)

        classifier = OrderedDict()
        self.linear_classifier = final_convolution(currdim, n_classes, self.final_kernel, cut_sites = cutedges, strides = self.final_strides, batch_norm = self.batch_norm, predict_from_dist = self.predict_from_dist)
        
        if self.outclass == 'Class':
            classifier['Sigmoid'] = nn.Sigmoid()
        elif self.outclass == 'Softmax':
            classifier['Softmax'] = nn.Softmax(dim = -1)
        elif self.outclass == 'Multi_class':
            classifier['Mclass'] = nn.Softmax(dim = -2)
        elif self.outclass != 'Linear':
            classifier[self.outclass] = func_dict[self.outclass]
        self.classifier = nn.Sequential(classifier)
        
   
    # The prediction after training are performed on the cpu
    def predict(self, X, pwm_out = None, mask = None, device = None):
        if device is None:
            device = self.device
        if self.fixed_kernels is not None:
            if pwm_out is None:
                pwm_out = pwm_scan(X, self.fixed_kernels, targetlen = self.l_kernels, motif_cutoff = self.motif_cutoff)
            pwm_out = torch.Tensor(pwm_out)
            
        predout = batched_predict(self, X, pwm_out =pwm_out, mask = mask, device = device, batchsize = self.batchsize, shift_sequence = self.shift_sequence, random_shift = self.random_shift)
        return predout
    
    def forward(self, x, xadd = None, mask = None, mask_value = 0, location = 'None'):
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
                pred[:,mask,:] = mask_value
        
        if location == '0':    
            # Don't flatten representatino anymore. 
            # User can define for themselves how to use the representation
            return pred
        
        pred = self.modelstart(pred)
        
        if location == '1':
            return pred
        
        if self.dilated_convolutions > 0:
            pred = self.convolution_layers(pred)
        
        if location == '2':
            return pred
        
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
            return pred
        
        if self.transformer_convolutions > 0 or self.trmax_pooling or self.trmean_pooling:
            pred = self.trconvolution_layers(pred)
        
        if location == '4':
            return pred
        
        if self.gapped_convs is not None:
            pred = self.gapped_convolutions(pred)
        
        if location == '5':
            return pred
        
        pred = self.linear_classifier(pred)
        
        if location == '-1' or location == '6':
            return pred
        
        if self.outclass != "Linear":
            pred = self.classifier(pred)
        
        return pred
    
    
    def fit(self, X, Y, XYval = None, sample_weights = None):
        self.saveloss = fit_model(self, X, Y, XYval = XYval, sample_weights = sample_weights, loss_function = self.loss_function, validation_loss = self.validation_loss, batchsize = self.batchsize, device = self.device, optimizer = self.optimizer, optim_params = self.optim_params, verbose = self.verbose, lr = self.lr, kernel_lr = self.kernel_lr, hot_start = self.hot_start, warm_start = self.warm_start, outname = self.outname, adjust_lr = self.adjust_lr, patience = self.patience, init_adjust = self.init_adjust, keepmodel = self.keepmodel, load_previous = self.load_previous, write_steps = self.write_steps, checkval = self.checkval, writeloss = self.writeloss, init_epochs = self.init_epochs, epochs = self.epochs, l1reg_last = self.l1reg_last, l2_reg_last = self.l2reg_last, l1_kernel = self.l1_kernel, reverse_sign = self.reverse_sign, shift_back = self.shift_sequence, random_shift=self.random_shift, smooth_onehot = self.smooth_onehot, restart = self.restart, masks = self.masks, nmasks = self.nmasks, augment_representation = self.augment_representation, aug_kernel_size = self.aug_kernel_size, aug_conv_layers = self.aug_conv_layers, aug_loss_masked = self.aug_loss_masked, aug_loss = self.aug_loss, aug_loss_mix = self.aug_loss_mix, **self.kwargs)
        





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
            
    

    
    
    








