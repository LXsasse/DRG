import sys, os 
import numpy as np
import torch.nn as nn
import torch
from collections import OrderedDict


from .modules import parallel_module, gap_conv, interaction_module, pooling_layer, correlation_loss, correlation_both, cosine_loss, cosine_both, zero_loss, Complex, Expanding_linear, Res_FullyConnect, Residual_convolution, Res_Conv1d, MyAttention_layer, Kernel_linear, loss_dict, func_dict, func_dict_single, Padded_Conv1d, RC_Conv1d, PredictionHead, Hyena_Conv, final_convolution
from .model_training import pwm_scan, batched_predict, fit_model
from .model_output import add_params_to_outname







# flexible Convolutional neural network architecture
class bpcnn(nn.Module):
    def __init__(self, loss_function = 'MSE', validation_loss = None, n_features = None, reverse_complement = False, n_classes = 1, l_seqs = None, l_out = None, num_kernels = 0, kernel_bias = True, fixed_kernels = None, motif_cutoff = None, l_kernels = 7, kernel_function = 'GELU', warm_start = False, hot_start = False, hot_alpha=0.01, kernel_thresholding = 0, max_pooling = False, mean_pooling = False, weighted_pooling = False, pooling_size = None, pooling_steps = None, net_function = 'GELU', dilated_convolutions = 0, strides = 1, conv_increase = 1., dilations = 1, l_dilkernels = None, dilmax_pooling = 0, dilmean_pooling = 0, dilweighted_pooling= 0, dilpooling_residual = 1, dilresidual_entire = False, dilresidual_concat = False, gapped_convs = None, gapconv_residual = True, gapconv_pooling = False, embedding_convs = 0, n_transformer = 0, n_attention = 0, n_interpolated_conv = 0, n_distattention = 0, dim_distattention=2.5, dim_embattention = None, attentionmax_pooling = 0, attentionweighted_pooling = 0, attentionconv_pooling = 1, attention_multiplier = 0.5, sum_attention = False, n_hyenaconv = 0, transformer_convolutions = 0, trdilations = 1, trstrides = 1, l_trkernels = None, trconv_dim = None, trmax_pooling = 0, trmean_pooling = 0, trweighted_pooling = 0, trpooling_residual = 1, trresidual_entire = False, final_kernel = 1, final_strides = 1, predict_from_dist = False, dropout = 0., batch_norm = False, l1_kernel = 0, l2reg_last = 0., l1reg_last = 0., shift_sequence = None, random_shift = False, reverse_sign = False, smooth_onehot = 0, epochs = 1000, lr = 1e-2, kernel_lr = None, adjust_lr = 'F', batchsize = None, patience = 25, outclass = 'Linear', outname = None, optimizer = 'Adam', optim_params = None, verbose = True, checkval = True, init_epochs = 3, writeloss = True, write_steps = 10, device = 'cpu', load_previous = True, init_adjust = True, seed = 101010, keepmodel = False, generate_paramfile = True, add_outname = True, restart = False, masks = None, nmasks = None, augment_representation = None, aug_kernel_size = None, aug_conv_layers = 1, aug_loss_masked = True, aug_loss = None, aug_loss_mix = None, **kwargs):
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
        
        self.n_hyenaconv = n_hyenaconv
        
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
        modellist[kernel_function+'0'] = func_dict_single[kernel_function]
        
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
            classifier[self.outclass] = func_dict_single[self.outclass]
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
        





    

    
    
    








