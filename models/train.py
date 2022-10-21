import torch
import numpy as np
import sys, os
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
from collections import OrderedDict
from torch import Tensor
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from init import MyDataset, get_device
from modules import loss_dict, func_dict, final_convolution
from torch_regression import torch_Regression


#### Wondering if there is something similar to the alpha fold techniques that we can do for this here: They use predicted information in a latter layer and pass it to an earlier layer 3 times. I sounds like a new type of recurrent block, a specialized recurrent block. Maybe there is something that we can do simiarly here.
# Generally, the issue is that we cannot align sequences because genes have different lengths and enhancers are placed in different locations. However, relaitve distances may play a role. 
# to capture these relative distances between motifs we need long-range equidistant modules
# We basically need an aligmnment that focuses on aligning regulatory sequences properly to each other




# Generates a list of pwms with targetlen from a pwm with different or same length
# for example, creates two pwms of length 7 from pwm of length 8
def pwmset(pwm, targetlen, shift_long = True):
    if np.shape(pwm)[-1] == targetlen:
        return pwm[None, :]
    elif np.shape(pwm)[-1] > targetlen:
        if shift_long:
            npwms = []
            for l in range(np.shape(pwm)[-1]-targetlen+1):
                npwm = np.zeros((np.shape(pwm)[0], targetlen))
                npwm[:] = pwm[:, l:l+targetlen]
                npwms.append(npwm)
        else:
            npwm = np.zeros((np.shape(pwm)[0], targetlen))
            l = int((np.shape(pwm)[-1]-targetlen)/2)
            npwm[:] = pwm[:, l:l+targetlen]
            npwms = [npwm]
        return np.array(npwms)
    elif np.shape(pwm)[-1] < targetlen:
        if shift_long:
            npwms = []
            for l in range(targetlen - np.shape(pwm)[-1]+1):
                npwm = np.zeros((np.shape(pwm)[0], targetlen))
                npwm[:,l:l+np.shape(pwm)[-1]] = pwm[:]
                npwms.append(npwm)
        else:
            npwm = np.zeros((np.shape(pwm)[0], targetlen))
            l = int((targetlen-np.shape(pwm)[-1])/2)
            npwm[:,l:l+np.shape(pwm)[-1]] = pwm[:]
            npwms = [npwm]
        return np.array(npwms)
        
# Scans onehot encoded numpy array for pwms
def pwm_scan(sequences, pwms, targetlen = None, activation = 'max', motif_cutoff = None, set_to = 0., verbose = False):
    # if pwms are longer than the targeted kernel size then its unclear if we should the convolution with the right side of the pwm or the left side. Each would create  a different positional pattern. Therefore we take the mean over both options all options of pwms with the target length.
    # If Pwms are smaller than the target len we use all the options of padded pwms to create scanning pattern
    if targetlen is None:
        targetlen = np.amax([len(pqm.T) for pqm in pwms])
    outscan = np.zeros((np.shape(sequences)[0], len(pwms), np.shape(sequences)[-1]-targetlen+1))
    if verbose:
        print('Scanning', len(sequences), 'sequences with', len(pwms), 'PWMs')
    for p, pwm in enumerate(pwms):
        if p%10 == 0 and verbose:
            print(p)
        setps = pwmset(pwm, targetlen = targetlen)
        for l in range(np.shape(sequences)[-1]-targetlen+1):
            setscans = np.sum(sequences[:,None,:,l:l+targetlen] * setps[None, :, :, :], axis = (-1, -2))
            # max activation across all shorter subpwms
            if activation == 'max':
                outscan[:,p, l] = np.amax(setscans, axis = -1)
            elif activation == 'mean':
                outscan[:,p, l] = np.mean(setscans, axis = -1)
    # pwms also assign values to partial fits of the sequence, to remove these partial fits one can 
    if motif_cutoff is not None:
        outscan[outscan < motif_cutoff] = set_to
    return outscan


def l1_loss(w):
    return torch.abs(w).mean()
    
def L2_loss(w):
    return torch.square(w).mean()
  
def save_model(model, PATH):
    torch.save(model.state_dict(), PATH)
    
def load_model(model, PATH, device):
    # avoids memory error on gpu: https://discuss.pytorch.org/t/cuda-error-out-of-memory-when-load-models/38011/3
    state_dict = torch.load(PATH, map_location = 'cpu')
    model.load_state_dict(state_dict)
    model.to(device)

class hook_grads():
    def __init__(self):
        self.grads = {}
    def save_grad(self,name):
        def hook(grad):
            self.grads[name] = grad
        return hook

def fit_model(model, X, Y, XYval = None, sample_weights = None, loss_function = 'MSE', validation_loss = None, batchsize = None, device = 'cpu', optimizer = 'Adam', optim_params = None,  verbose = True, lr = 0.001, kernel_lr = None, hot_start = False, hot_alpha = 0.01, warm_start = False, outname = 'Fitmodel', adjust_lr = 'F', patience = 25, init_adjust = True, reduce_lr_after = 1, keepmodel = False, load_previous = True, write_steps = 10, checkval = True, writeloss = True, init_epochs = 250, epochs = 1000, l1reg_last = 0, l2reg_last = 0, l1_kernel= 0, reverse_sign = False, shift_back = None, random_shift = False, smooth_onehot = 0, multiple_input = False, restart = False, masks = None, nmasks = None, augment_representation = None, aug_kernel_size = None, aug_conv_layers = 1, aug_loss_masked = True, aug_loss= None, aug_loss_mix = None, aug_lr = None, **kwargs):
    
    
    # Default parameters for each optimizer
    if optim_params is None:
        if optimizer == 'SGD':
            optim_params = 0
        elif optimizer == 'NAG':
            optim_params = 0.7 # momentum
        elif optimizer == 'Adagrad':
            optim_params = 0. # lr_decay rate
        elif optimizer == 'Adadelta':
            optim_params = 0.9 # coefficient for running average
        elif optimizer in ['Adam','NAdam','AdamW', 'Amsgrad']:
            optim_params = (0.9,0.999)
        else:
            optim_params = None
    
    if model.outname is not None:
        outname = model.outname
        
    if shift_back is not None:
        if isinstance(shift_back, int):
            if shift_back > 0:
                shift_back = np.arange(0,shift_back+1, dtype = int)
            else:
                shift_back = None
        else:
            shift_back = np.unique(np.append([0],shift_back))
    
    # Assignment of loss function
    if loss_function is None:
        loss_func = nn.MSELoss()
    elif isinstance(loss_function, str):
        loss_func = loss_dict[loss_function]
    elif type(loss_function) == nn.Module:
        loss_func = loss_function
    
    # Loss function for validation set to determine stop criterion
    if validation_loss is None:
        validation_loss = loss_function
        val_loss = loss_func
    elif isinstance(validation_loss, str):
        val_loss = loss_dict[validation_loss]
    elif type(validation_loss) == nn.Module:
        val_loss = validation_loss
    
    aug_convolution = aug_kernel_size
    seqmaskcut = None
    if augment_representation is not None:
        # find out size of output at "augment_representation" to determine padding for final_convolution()
        testrep = model.forward(torch.Tensor(X[:1]), location = augment_representation)
        rep_size = testrep.size(dim = -1)
        rep_dim = testrep.size(dim = -2)
        seq_size = np.shape(X)[-1]
        aug_convolution = nn.Sequential(final_convolution(rep_dim, 4, aug_kernel_size, strides = 1, n_convolutions = aug_conv_layers), nn.Softmax(dim = -2))
        aug_convolution.to(device)
        max_windowsize = min(rep_size, seq_size) # For some representation one may not be able to reconstruct masked windows along the entire sequence
        seqmaskcut = int(np.ceil((seq_size - max_windowsize)/2))
        # sequences values will only be masked within these values
        seqmaskcut = [seqmaskcut, seqmaskcut + max_windowsize]
        # remove masks that mask values outside the window that can be covered
        maskcover = np.where(masks[:,0])
        # find masks that have values outside seqmaskcut
        delmask = np.unique(maskcover[0][(maskcover[1]<seqmaskcut[0]) * (maskcover[1]>=seqmaskcut[1])])
        masks = masks[~np.isin(np.arange(len(masks)),delmask)]
        aug_loss = loss_dict[aug_loss]
        if aug_lr is None:
            aug_lr = lr
        
    
    # if the model uses fixed pwms that are concatenated with the input before pooled
    fixed_kernels = None
    if 'fixed_kernels' in model.__dict__:
        fixed_kernels = model.__dict__['fixed_kernels']
        l_kernels = model.__dict__['l_kernels']
        motif_cutoff = model.__dict__['motif_cutoff']
    
    last_layertensor = None
    if l1reg_last > 0 or l2reg_last > 0:
        # Determine the name of the parameter tensors in the last layer that need to be regularized
        last_layertensor = []
        for param_tensor in model.state_dict():
            if 'classifier' in str(param_tensor) and 'weight' in str(param_tensor):
                last_layertensor.append(str(param_tensor))

    
    kernel_layertensor = None
    if l1_kernel > 0:
        kernel_layertensor = []
        for param_tensor in model.state_dict():
            if 'convolutions.weight' in str(param_tensor):
                kernel_layertensor.append(str(param_tensor))
        
    # If pwms are provided, scan the sequences with them to avoid reoccuring scanning
    pwm_out = None
    if fixed_kernels is not None:
        # generate maxpooled feature list with given kernels
        pwm_out = pwm_scan(X, fixed_kernels, targetlen = l_kernels, verbose = verbose, motif_cutoff = motif_cutoff)
        pwm_out = torch.Tensor(pwm_out)
    
    
    # XYval represents a validation set on which the performance for the stop criterion is measured
    pwm_outval = None
    if XYval is None:
        # If XYval is None, the training data will be used
        Xval, Yval = X, Y
        if fixed_kernels is None:
            pwm_outval = pwm_out
    else:
        Xval, Yval = XYval[0], XYval[1]
        if fixed_kernels is not None:
            pwm_outval = pwm_scan(Xval, fixed_kernels, targetlen = l_kernels, verbose = verbose, motif_cutoff = motif_cutoff) 
            pwm_outval = torch.Tensor(pwm_outval)
    
    val_weights = None
    if sample_weights is not None:
        val_weights = torch.ones(np.shape(Xval)[0])
    
    if sample_weights is not None:
        # If weighted outputs then the loss needs to normalized by the sum of weights
        trainsize = np.sum(sample_weights)
    else:
        if multiple_input:
            trainsize = float(len(X[0]))
        else:
            trainsize = float(len(X))
    if multiple_input:
        valsize = float(len(Xval[0]))
    else:
        valsize = float(len(Xval))
    
    if multiple_input:
        X = [torch.Tensor(x) for x in X] # transform to torch tensor
        Xval = [torch.Tensor(xval) for xval in Xval] # transform to torch tensor
    else:
        X = torch.Tensor(X) # transform to torch tensor
        Xval = torch.Tensor(Xval) # transform to torch tensor
    trainlen, vallen = len(Y), len(Yval)
    Y = torch.Tensor(Y)
    Yval = torch.Tensor(Yval)
    
    
    my_dataset = MyDataset(X, Y, axis = int(multiple_input)) # create your datset
    if batchsize is None:
        batchsize = len(X)
    
    droplast = False
    mindata = 10 # minimum left data points for last batch to not be dropped
    if trainlen%batchsize < mindata:
        droplast = True
    dataloader = DataLoader(my_dataset, batch_size = batchsize, shuffle = True, drop_last = droplast)
    
    my_val_dataset = MyDataset(Xval, Yval, axis = int(multiple_input)) # create your datset
    val_batchsize = int(min(batchsize,vallen)) # largest batchsize for validation set is 250 to save memory on gpu
    
    vdroplast = False
    if vallen%val_batchsize < mindata:
        vdroplast = True
    val_dataloader = DataLoader(my_val_dataset, batch_size = val_batchsize, shuffle = True, drop_last = vdroplast)
    
    if batchsize < mindata and loss_function in ['Correlationclass', 'MSECorrelation']:
        print(loss_function, 'NOT RECOMMENDED WITH BATCHSIZE <', mindata)
    if val_batchsize < mindata and validation_loss in ['Correlationclass', 'MSECorrelation']:
        print(validation_loss, 'NOT RECOMMENDED WITH BATCHSIZE <', mindata)
    
    if warm_start:
        print('Warm start')
        tmodel = torch_Regression(alpha = 0., fit_intercept = False, loss_function = 'MSE', logistic = 'Linear', penalty = 'none', kernel_length = model.l_kernels, n_kernels = model.num_kernels, kernel_function = model.kernel_function, pooling = 'Max', pooling_length = None, alpha_kernels = 0., is_zero = 0, epochs = 100, lr = 0.1, optimizer = 'SGD', optim_params = None, batchsize = batchsize, device = device, seed = model.seed, verbose = verbose, patience = 5, adjust_lr = 0.3, outname = None, write_model_params = False, norm = False)
        tmodel.fit(X, Y,XYval = [Xval, Yval], sample_weights = sample_weights)
        Ypred = torch.Tensor(tmodel.predict(X))
        Yvalpred = torch.Tensor(tmodel.predict(Xval))
        print('Warmstart valitrain', val_loss(Ypred.to(device),Y).item()/trainsize)
        print('Warmstart valival', val_loss(Yvalpred.to(device), Yval).item()/valsize)
        hotpwms = tmodel.kernels_#*tmodel.coef_.T[...,None]
        
        #if self.kernel_bias:
        #    hotpwms += tmodel.intercept_ #nn.Parameter(torch.ones_like(self.convolutions.bias)*1e-16)
        model.convolutions.weight = nn.Parameter(torch.Tensor(hotpwms))
    # Hot start initializes the kernels with onehot encoded pwms from Lasso regression
    elif hot_start:
        ### REPLACE with 1-d convolutional neural network or introduce another variable to do that
        hotpwms = kernel_hotstart(model.num_kernels, model.l_kernels, X.numpy(), Y.numpy(), XYval = [Xval.numpy(), Yval.numpy()], alpha = hot_alpha)
        maxweights = np.sum(hotpwms, axis = (-1,-2))
        model.convolutions.weight = nn.Parameter(torch.Tensor(hotpwms))
        if model.kernel_bias:
            model.convolutions.bias = nn.Parameter(torch.ones_like(model.convolutions.bias)*-np.amin(maxweights)*0.5)
        if model.kernel_thresholding > 0:
            model.modelstart.Kernelthresh.bias = nn.Parameter(torch.ones_like(model.modelstart.Kernelthresh.bias)*-np.amin(maxweights)*0.5)
        
        
    
    # Send model parameter to GPU
    model.to(device)
    
    # Collect the sizes and names of all parameters of the model layers 
    layersizes = []
    layernames = []
    i = 0
    for param_tensor in model.state_dict():
        if verbose: 
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        layname = param_tensor.strip('.weight').strip('.bias')
        if layname in layernames:
            layersizes[layernames.index(layname)] += np.prod(model.state_dict()[param_tensor].size())
        else:
            layernames.append(layname)
            layersizes.append(np.prod(model.state_dict()[param_tensor].size()))
    
    # Increasing or decreasing learning rates from kernel_lr if learning_rate set to "gradual"
    # Parameter size adjusted lr with Size or Antisize
    if adjust_lr == 'LogGradual':
        a_lrs = 10**np.linspace(np.log10(kernel_lr), np.log10(lr), len(layernames))
    elif adjust_lr == 'Gradual':
        a_lrs = np.linspace(kernel_lr, lr, len(layernames))
    elif adjust_lr == 'Size':
        a_lrs = lr * np.array(layersizes)/np.amax(layersizes)
    elif adjust_lr == 'Antisize':
        a_lrs = lr/(np.array(layersizes)/np.amin(layersizes))
    else:
        a_lrs = lr*np.ones(len(layersizes))
        if kernel_lr is not None:
            for a, lan in enumerate(layernames):
                if 'convolutions' in lan:
                    a_lrs[a] = kernel_lr
    
    # Give all the lrs to a list of dictionaries
    a_dict = []
    for param_tensor, tensor in model.named_parameters():
        layname = param_tensor.strip('.weight').strip('.bias')
        if verbose:
            print(param_tensor, a_lrs[layernames.index(layname)])
        a_dict.append({'params':tensor, 'lr':a_lrs[layernames.index(layname)]})
   
    # initialize learning rate for aug_convolution
    if augment_representation is not None:
        for param_tensor, tensor in aug_convolution.named_parameters():
            if verbose:
                print(param_tensor, aug_lr)
            a_dict.append({'params':tensor, 'lr':aug_lr})
    
    # Give this dictionary to the optimizer
    if optimizer == 'SGD':
        optimizer = optim.SGD(a_dict, lr=lr, momentum=optim_params)
    elif optimizer == 'NAG':
        optimizer = optim.SGD(a_dict, lr=lr, momentum=optim_params, nesterov = True)
    elif optimizer == 'Adagrad':
        optimizer = optim.Adagrad(a_dict, lr=lr, lr_decay = optim_params)
    elif optimizer == 'Adadelta':
        optimizer = optim.Adadelta(a_dict, lr=lr, rho=optim_params)
    elif optimizer == 'Adam':
        optimizer = optim.Adam(a_dict, lr=lr, betas=optim_params)
    elif optimizer == 'NAdam':
        optimizer = optim.NAdam(a_dict, lr=lr, betas=optim_params, momentum_decay = 4e-3)
    elif optimizer == 'AdamW':
        optimizer = optim.AdamW(a_dict, lr=lr, betas=optim_params)
    elif optimizer == 'Amsgrad':
        optimizer = optim.AdamW(a_dict, lr=lr, betas=optim_params, amsgrad = True)
    else:
        print(optimizer, 'not allowed')
    
    
    # Compute losses at the beginning with randomly initialized model
    lossorigval, lossval = excute_epoch(model, val_dataloader, loss_func, pwm_outval, valsize, False, device, val_loss = val_loss, optimizer = None, l1reg_last = 0, l2reg_last = 0, l1_kernel = 0, last_layertensor = None, kernel_layertensor = None, sample_weights = None, val_all = Yval, reverse_sign = False, shift_back = shift_back, random_shift = random_shift,smooth_onehot = 0,multiple_input = multiple_input, masks = masks, nmasks = nmasks, seqmaskcut = seqmaskcut)
    
    beginning_loss, loss2 = excute_epoch(model, dataloader, loss_func, pwm_out, trainsize, False, device, val_loss = val_loss, optimizer = None, l1reg_last = l1reg_last, l2reg_last = l2reg_last, l1_kernel = l1_kernel, last_layertensor = last_layertensor, kernel_layertensor = kernel_layertensor, sample_weights = sample_weights, val_all = Y, reverse_sign = reverse_sign, shift_back = shift_back,random_shift = random_shift, smooth_onehot = smooth_onehot, multiple_input = multiple_input, masks = masks, nmasks = nmasks, seqmaskcut = seqmaskcut)
    
    saveloss = [lossval, loss2, lossorigval, beginning_loss, 0]
    save_model(model, outname+'_params0.pth')
    save_model(model, outname+'_parameter.pth')   
    
    roundto = 6
    
    if verbose:
        print('Train_loss(val), Val_loss(val), Train_loss(train), Val_loss(train)')
    if writeloss:
        writebeginning = str(round(lossorigval,roundto))+'\t'+str(round(lossval,roundto))+'\t'+str(round(beginning_loss,roundto))+'\t'+str(round(loss2,roundto))
        save_losses(outname+'_loss.txt', 0, writebeginning)
    if verbose:
        print(0, writebeginning)
    stopcriterion = stop_criterion(checkval, patience)
    early_stop, stopexp = stopcriterion(0, lossval)
    
    # Start epochs and updates
    restarted = 0
    e = 0
    been_larger = 1
    
    while True:
        trainloss, loss2 = excute_epoch(model, dataloader, loss_func, pwm_out, trainsize, True, device, val_loss = val_loss, optimizer = optimizer, l1reg_last = l1reg_last, l2reg_last = l2reg_last, l1_kernel = l1_kernel, last_layertensor = last_layertensor, kernel_layertensor = kernel_layertensor, sample_weights = sample_weights, val_all = Y, reverse_sign = reverse_sign, shift_back = shift_back, random_shift = random_shift, smooth_onehot = smooth_onehot, multiple_input = multiple_input, masks = masks, nmasks = nmasks, augment_representation = augment_representation, aug_layer = aug_convolution, aug_loss_masked = aug_loss_masked, aug_loss = aug_loss, aug_loss_mix = aug_loss_mix, seqmaskcut = seqmaskcut)
        
        model.eval() # Sets model to evaluation mode which is important for batch normalization over all training mean and for dropout to be zero
        e += 1
        
        lossorigval, lossval = excute_epoch(model, val_dataloader, loss_func, pwm_outval, valsize, False, device, val_loss = val_loss, optimizer = None, l1reg_last = 0, l2reg_last = 0, l1_kernel = 0, last_layertensor = None, kernel_layertensor = None, sample_weights = None, val_all = Yval, reverse_sign = False, shift_back = shift_back, random_shift = random_shift, smooth_onehot = 0, multiple_input = multiple_input, masks = masks, nmasks = nmasks, seqmaskcut = seqmaskcut)
        
        
        
        if e%write_steps == 0 or e < init_epochs:
            if writeloss:
                save_losses(outname+'_loss.txt', e, str(round(lossorigval,roundto))+'\t'+str(round(lossval,roundto))+'\t'+str(round(trainloss,roundto))+'\t'+str(round(loss2,roundto)))
            if verbose:
                print(e, str(round(lossorigval,roundto))+'\t'+str(round(lossval,roundto))+'\t'+str(round(trainloss,roundto))+'\t'+str(round(loss2,roundto)))
            
        model.train() # Set model modules back to training mode
        
        early_stop, stopexp = stopcriterion(e, lossval)
        
        #print(trainloss, beginning_loss, e, init_epochs, init_adjust)
        if init_adjust and e > init_epochs:
            if (np.isnan(lossval) or np.isnan(loss2) or np.isnan(lossorigval) or np.isnan(trainloss)) or ((been_larger >= reduce_lr_after) and (trainloss > beginning_loss)):
                # reduces learning rate if training loss goes up actually
                # need something learnable for each layer during training
                restarted += 1
                if restarted > 15:
                    break
                save_losses(outname+'_loss.txt', 0, writebeginning)
                load_model(model, outname+'_params0.pth',device)
                e = 0
                lossorigval, lossval = excute_epoch(model, val_dataloader, loss_func, pwm_outval, valsize, False, device, val_loss = val_loss, optimizer = None, l1reg_last = 0, l2reg_last = 0, l1_kernel = 0, last_layertensor = None, kernel_layertensor = None, sample_weights = None, val_all = Yval, reverse_sign = False, smooth_onehot = 0, shift_back = shift_back, random_shift = random_shift, multiple_input = multiple_input, masks = masks, nmasks = nmasks, seqmaskcut = seqmaskcut)
                early_stop, stopexp = stopcriterion(0, lossval)
                
                for a, adict in enumerate(a_dict):
                    a_dict[a]['lr'] = adict['lr'] *0.25
                print('Learning rate reduced', restarted,  lossorigval, lossval, a_dict[-1]['lr'])
                been_larger = 1
        
            elif trainloss > beginning_loss:
                been_larger += 1
            else:
                been_larger = 1
            
        if e == epochs or (early_stop and (e > init_epochs)):
            # stop after reaching maximum epoch number
            if restart:
                load_model(model, outname+'_parameter.pth',device)
                if verbose:
                    print("Loaded model from", outname+'_parameter.pth', e - saveloss[-1], 'steps ago with loss', saveloss[0])
                restarted =0
                for mname, layer in model.named_modules():
                    if hasattr(layer, 'reset_parameters') and 'convolutions' not in mname.split('.'):
                        layer.reset_parameters()
                        if verbose:
                            print('Reseted', mname)
                save_model(model, outname+'_params0.pth')
                beginning_loss, loss2 = excute_epoch(model, dataloader, loss_func, pwm_out, trainsize, False, device, val_loss = val_loss, optimizer = None, l1reg_last = l1reg_last, l2reg_last = l2reg_last, l1_kernel = l1_kernel, last_layertensor = last_layertensor, kernel_layertensor = kernel_layertensor, sample_weights = sample_weights, val_all = Y, reverse_sign = reverse_sign, shift_back = shift_back, random_shift = random_shift, smooth_onehot = smooth_onehot, multiple_input = multiple_input, masks = masks, nmasks = nmasks, seqmaskcut = seqmaskcut)
                lossorigval, lossval = excute_epoch(model, val_dataloader, loss_func, pwm_outval, valsize, False, device, val_loss = val_loss, optimizer = None, l1reg_last = 0, l2reg_last = 0, l1_kernel = 0, last_layertensor = None, kernel_layertensor = None, sample_weights = None, val_all = Yval, reverse_sign = False, smooth_onehot = 0, shift_back = shift_back, random_shift = random_shift, multiple_input = multiple_input, masks = masks, nmasks = nmasks, seqmaskcut = seqmaskcut)
                early_stop, stopexp = stopcriterion(0, lossval)
                restart = False
                epochs = epochs + e
                if verbose:
                    print('New start', str(round(lossorigval,roundto))+'\t'+str(round(lossval,roundto))+'\t'+str(round(trainloss,roundto))+'\t'+str(round(loss2,roundto)))
                
            else:
                if early_stop and verbose:
                    # Early stopping if validation loss is not improved for patience epochs
                    if verbose:
                        print(e, lossval, stopexp)
                
                if lossval > saveloss[0] and load_previous:
                    # Load better model if it was created in an earlier epoch
                    load_model(model, outname+'_parameter.pth',device)
                    if verbose:
                        print("Loaded model from", outname+'_parameter.pth', e - saveloss[-1], 'steps ago with loss', saveloss[0])
                else:
                    saveloss = [lossval, loss2, lossorigval, trainloss, e]
                os.remove(outname+'_params0.pth')
                break
        
        elif not early_stop:
            # if early stopping is False, save loss and parameters
            if (~np.isnan(lossval) and lossval < saveloss[0]) or (~np.isnan(lossval) and np.isnan(saveloss[0])):
                saveloss = [lossval, loss2, lossorigval, trainloss, e]
                save_model(model, outname+'_parameter.pth')
    
    if not keepmodel:
        os.remove(outname+'_parameter.pth')
    return saveloss


def save_losses(PATH, i, lo):
    if i == 0:
        obj = open(PATH, 'w')
    else:
        obj = open(PATH, 'a')
    obj.write(str(i)+'\t'+str(lo)+'\n')
    



class stop_criterion(nn.Module):
    def __init__(self, checkval, patience):
        super(stop_criterion, self).__init__()
        self.best_loss = None
        self.steps_since = 0
        self.checkval = checkval
        self.patience = patience
    
    def forward(self, i, lossi):
        # if loss gets nan it's not producing meaningful losses
        if np.isnan(lossi):
            return True, 'Loss nan'
        
        if self.checkval:
            if i == 0 or self.best_loss is None:
                self.best_loss = lossi
                self.steps_since = 0
            else:
                if lossi < self.best_loss:
                    self.best_loss = lossi
                    self.steps_since = 0
                else:
                    self.steps_since +=1
                if self.steps_since == self.patience:
                    return True, 'Loss has not decreased since '+str(self.patience)+' steps'
                
        return False, None



def reverse_inoutsign(x):
    x = torch.cat([x, -x], dim = 0)
    return x


# make version so that nothing gets lost and so that array with shifts can be given    
def shift_sequences(sample_x, shift_back, maxshift = None, just_pad = False, mode = 'constant', value = 0.25, random_shift = False): # 'constant', 'reflect', 'replicate' or 'circular'
    if maxshift is None:
        maxshift = np.amax(shift_back)
    sample_xs = [] #nn.functional.pad(sample_x, (maxshift, maxshift), mode = mode, value = value)]
    if random_shift and not just_pad:
        sbs = np.random.choice(shift_back, int(random_shift))
        for sb in sbs:
            sample_xs.append(nn.functional.pad(sample_x, (maxshift-sb, maxshift+sb), mode = mode, value = value))
            sample_xs.append(nn.functional.pad(sample_x, (maxshift+sb, maxshift-sb), mode = mode, value = value))
    elif not just_pad:
        for sb in shift_back:
            sample_xs.append(nn.functional.pad(sample_x, (maxshift-sb, maxshift+sb), mode = mode, value = value))
            sample_xs.append(nn.functional.pad(sample_x, (maxshift+sb, maxshift-sb), mode = mode, value = value))
    else:
        sample_xs = [nn.functional.pad(sample_x, (maxshift, maxshift), mode = mode, value = value)]
    sample_x = torch.cat(sample_xs, dim = 0)
    return sample_x

def smooth_onehotfunc(sample_x, ns = 2):
    sample_x = torch.cat([sample_x, torch.sign(torch.sum(sample_x, dim =1).repeat((ns,1))).unsqueeze(1)*(sample_x.absolute().repeat((ns,1,1)) - torch.rand(ns*sample_x.size(dim=0),ns*sample_x.size(dim=1),ns*sample_x.size(dim=2))*0.5).absolute()], dim = 0)
    return sample_x

def mask_sequence(sample_x, masks, nmasks, mask_val = 0.25):
    nsamp = sample_x.size(dim=0)
    cmasks = masks[np.random.choice(len(masks), nmasks * nsamp)]
    sample_x = sample_x.repeat(nmasks, 1, 1)
    sample_x[cmasks] = mask_val
    return sample_x, cmasks


# execute one epoch with training or validation set. Training set takes gradient but validation set computes loss without gradient
def excute_epoch(model, dataloader, loss_func, pwm_out, normsize, take_grad, device, val_loss = None, optimizer = None, l1reg_last = 0, l2reg_last = 0, l1_kernel = 0, last_layertensor = None, kernel_layertensor = None, sample_weights = None, val_all = None, reverse_sign = False, shift_back = None, random_shift = False, smooth_onehot = 0, multiple_input = False, masks = None, nmasks = None, augment_representation = None, aug_layer = None, aug_loss_masked = True, aug_loss = None, aug_loss_mix = None, seqmaskcut = None):
    
    if val_loss is None:
        val_loss = loss_func
    
    trainloss = 0.
    validatloss = 0.
    # if size of validation set different from batchsize, then do val_all
    if val_all is not None:
        Ypred_all = torch.empty(val_all.size())
    
    yclasses = model.n_classes
    if 'l_out' in model.__dict__:
        if model.l_out is not None:
            yclasses *= model.l_out
    
    tsize = 1
    augsize = None
    if nmasks is not None and augment_representation is not None:
        augsize = tsize * nmasks
    elif nmasks is not None and seqmaskcut is None:
        tsize = tsize * nmasks
    
    if shift_back is not None:
        if random_shift:
            tsize = tsize*int(random_shift)*2
        else:
            tsize = tsize*2*len(shift_back)
    if reverse_sign:
    # all operations are also performed in the negative space, i.e. 1 => -1 in one hot encoding and y = 4 => y = -4
    # shoud only be used if no relu is used after first convolution, otherwise it's confusing why this should help. Idea is basically to learn a symmetric function.
        tsize = tsize*2
    if smooth_onehot > 0:
        tsize = tsize *(smooth_onehot +1)
    
    
    for sample_x, sample_y, index in dataloader:
        if nmasks is not None:
            if augment_representation is not None and take_grad:
                sample_xaug, sample_mask = mask_sequence(sample_x,masks, nmasks)
                sample_xaug = sample_xaug[:,:,seqmaskcut[0]:seqmaskcut[1]]
                sample_mask = sample_mask[:,:,seqmaskcut[0]:seqmaskcut[1]]
                sample_yaug = torch.cat(nmasks*[sample_x[:,:,seqmaskcut[0]:seqmaskcut[1]]])
                sample_xaug = sample_xaug.to(device)
                sample_yaug = sample_yaug.to(device)
                sample_mask = sample_mask.to(device)
            elif seqmaskcut is None: 
                sample_x, sample_mask = mask_sequence(sample_x,masks, nmasks)
                sample_y = torch.cat(nmasks*[sample_y])
        
        if pwm_out is None:
            saddx = None
        else:
            saddx = pwm_out[index]
        # Add samples that are shifted by 1 to 'shift_back' positions
        if shift_back is not None:
            if multiple_input:
                sample_x = [shift_sequences(sam_x, shift_back, random_shift = random_shift) for sam_x in sample_x]
            else:
                sample_x = shift_sequences(sample_x, shift_back, random_shift = random_shift)
            if random_shift:
                sample_y = torch.cat((2*int(random_shift))*[sample_y])
            else:
                sample_y = torch.cat((len(shift_back)*2)*[sample_y])
            if saddx is not None:
                if random_shift:
                    saddx = saddx.repeat(3, 1, 1)
                else:
                    saddx = saddx.repeat(len(shift_back)*2, 1, 1)
        
        # Add samples with reverse sign in X and Y
        if reverse_sign:
            if multiple_input:
                sample_x = [reverse_inoutsign(sam_x) for sam_x in sample_x]
            else:
                sample_x = reverse_inoutsign(sample_x)
            sample_y = reverse_inoutsign(sample_y)
            if saddx is not None:
                saddx = reverse_inoutsign(saddx)
        
        if smooth_onehot > 0:
            if multiple_input:
                sample_x = [smooth_onehotfunc(sam_x, ns = smooth_onehot) for sam_x in sample_x]
            else:
                sample_x = smooth_onehotfunc(sample_x, ns = smooth_onehot)
            sample_y = torch.cat((smooth_onehot+1)* [sample_y])
            if saddx is not None:
                saddx = saddx.repeat(smooth_onehot +1, 1, 1)
            
        if saddx is not None:
            if multiple_input:
                saddx = [sax.to(device) for sax in saddx]
            else:
                saddx = saddx.to(device)
        if multiple_input:
            sample_x = [sam_x.to(device) for sam_x in sample_x]
        else:
            sample_x= sample_x.to(device)
        sample_y = sample_y.to(device)
        
        if take_grad:
            optimizer.zero_grad()
            Ypred = model.forward(sample_x, xadd = saddx)
            loss = loss_func(Ypred, sample_y)
            if sample_weights is not None:
                loss = loss*sample_weights[index][:,None]
            
            loss = torch.sum(loss)
            trainloss += float(loss.item())
            
            if l1reg_last > 0 and last_layertensor is not None:
                # NEED to use named_parameters instead of model.state_dict because state_dict does not come with gradient
                for param_name, tensor in model.named_parameters():
                    if param_name in last_layertensor:
                        loss += l1reg_last * l1_loss(tensor)
            
            if l2reg_last > 0 and last_layertensor is not None:
                for param_name, tensor in model.named_parameters():
                    if param_name in last_layertensor:
                        loss += l2reg_last * l2_loss(tensor)
                    
            if l1_kernel > 0 and kernel_layertensor is not None:
                for param_name, tensor in model.named_parameters():
                    if param_name in kernel_layertensor:
                        loss += l1_kernel * l1_loss(tensor)
            if augment_representation is not None:
                aug_rep = model.forward(sample_xaug, location = augment_representation)
                aug_rep = aug_layer(aug_rep)
                if aug_loss_masked:
                    aug_rep = aug_rep[sample_mask].reshape(1,4,-1)
                    sample_yaug = sample_yaug[sample_mask].reshape(1,4,-1)
                loss += aug_loss_mix * torch.sum(aug_loss(aug_rep, sample_yaug))
            
            loss.backward()
            optimizer.step()
            
        else:
            with torch.no_grad():
                Ypred = model.forward(sample_x, xadd = saddx)
                loss = loss_func(Ypred, sample_y)
                if sample_weights is not None:
                    loss = loss*sample_weights[index][:,None]
                    
                loss = torch.sum(loss)
                trainloss += float(loss.item())
                
        if val_all is not None:
            Ypred_all[index] = Ypred[:len(index)].detach().cpu()
        else:
            with torch.no_grad():
                validatloss += float(torch.sum(val_loss(Ypred, sample_y)).item())
    if val_all is not None:
        with torch.no_grad():
            validatloss = float(torch.sum(val_loss(Ypred_all, val_all)).item())
            
    trainloss /= (normsize*tsize)*yclasses
    validatloss /= normsize*yclasses
    return trainloss, validatloss


# The prediction after training are performed on the cpu
def batched_predict(model, X, pwm_out = None, mask = None, mask_value = 0, device = 'cpu', batchsize = None, shift_sequence = None, random_shift = True):
    if shift_sequence is not None:
        if isinstance(shift_sequence, int):
            if shift_sequence > 0:
                if not random_shift:
                    shift_sequence = np.arange(1,shift_sequence+1, dtype = int)
            else:
                shift_sequence = None
    model.eval()
    if device is None:
        device = model.device
    model.to(device)
    # Use no_grad to avoid computation of gradient and graph
    with torch.no_grad():
        islist = False
        if isinstance(X, list):
            islist = True
            X = [torch.Tensor(x) for x in X]
            dsize = X[0].size(dim = 0)
        else:
            X = torch.Tensor(X)
            dsize = X.size(dim = 0)
        if batchsize is not None:
            predout = []
            for i in range(0, int(dsize/batchsize)+int(dsize%batchsize != 0)):
                if pwm_out is not None:
                    pwm_outin = pwm_out[i*batchsize:(i+1)*batchsize]
                    if shift_sequence is not None:
                        pwm_outin = shift_sequences(pwm_outin, shift_sequence, just_pad = random_shift)
                    pwm_outin = pwm_outin.to(device)
                else:
                    pwm_outin = None
                if islist:
                    xin = [x[i*batchsize:(i+1)*batchsize] for x in X]
                    xsize = xin[0].size(dim = 0)
                    if shift_sequence is not None:
                        xin = [shift_sequences(x, shift_sequence, just_pad = random_shift) for x in xin]
                    xin = [x.to(device) for x in xin]
                
                else:
                    xin = X[i*batchsize:(i+1)*batchsize]
                    xsize = xin.size(dim = 0)
                    if shift_sequence is not None:
                        xin = shift_sequences(xin, shift_sequence, just_pad = random_shift)
                    xin = xin.to(device)
                fpred = model.forward(xin, xadd = pwm_outin, mask = mask,mask_value = mask_value).detach().cpu().numpy()
                if shift_sequence is not None:
                    fpred = fpred.reshape(-1,xsize,np.shape(fpred)[-1]).mean(axis =0)
                predout.append(fpred)
            predout = np.concatenate(predout, axis = 0)
        else:
            if islist:
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            predout = model.forward(X, xadd = pwm_out, mask = mask, mask_value = mask_value)
            predout = predout.detach().cpu().numpy()
            if shift_sequence is not None:
                # combine predictions for each gene across different shifts
                predout = predout.reshape(-1,dsize,fpred.size(dim=-1)).mean(dim =0)
    return predout

