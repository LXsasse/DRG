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
from modules import loss_dict, func_dict
from torch_regression import torch_Regression
from predict import pwmset, pwm_scan

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

def fit_model(model, X, Y, XYval = None, sample_weights = None, loss_function = 'MSE', validation_loss = None, batchsize = None, device = 'cpu', optimizer = 'Adam', optim_params = None,  verbose = True, lr = 0.001, kernel_lr = None, hot_start = False, hot_alpha = 0.01, warm_start = False, outname = 'Fitmodel', adjust_lr = 'F', patience = 25, init_adjust = True, keepmodel = False, load_previous = True, write_steps = 10, checkval = True, writeloss = True, init_epochs = 250, epochs = 1000, l1reg_last = 0, l2reg_last = 0, l1_kernel= 0, reverse_sign = False, shift_back = False, **kwargs):
    
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
        trainsize = float(len(X))
    valsize = float(len(Xval))
    
    X = torch.Tensor(X) # transform to torch tensor
    Y = torch.Tensor(Y)
    Xval = torch.Tensor(Xval) # transform to torch tensor
    Yval = torch.Tensor(Yval)
    
    
    my_dataset = MyDataset(X, Y) # create your datset
    if batchsize is None:
        batchsize = len(X)
    dataloader = DataLoader(my_dataset, batch_size = batchsize, shuffle = True)
    
    my_val_dataset = MyDataset(Xval, Yval) # create your datset
    val_batchsize = int(min(batchsize,valsize)) # largest batchsize for validation set is 250 to save memory on gpu
    val_dataloader = DataLoader(my_val_dataset, batch_size = val_batchsize, shuffle = True)
    
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
    if verbose:
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    
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
            a_lrs[0] = kernel_lr
    
    # Give all the lrs to a list of dictionaries
    a_dict = []
    for param_tensor, tensor in model.named_parameters():
        layname = param_tensor.strip('.weight').strip('.bias')
        if verbose:
            print(param_tensor, a_lrs[layernames.index(layname)])
        a_dict.append({'params':tensor, 'lr':a_lrs[layernames.index(layname)]})
    
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
    
    # Compute losses at the beginning with random model
    lossorigval, lossval = excute_epoch(model, val_dataloader, loss_func, pwm_outval, valsize, False, device, val_loss = val_loss, optimizer = None, l1reg_last = 0, l2reg_last = 0, l1_kernel = 0, last_layertensor = None, kernel_layertensor = None, sample_weights = None, val_all = Yval, reverse_sign = False, shift_back = 0)
    
    beginning_loss, loss2 = excute_epoch(model, dataloader, loss_func, pwm_out, trainsize, False, device, val_loss = val_loss, optimizer = None, l1reg_last = l1reg_last, l2reg_last = l2reg_last, l1_kernel = l1_kernel, last_layertensor = last_layertensor, kernel_layertensor = kernel_layertensor, sample_weights = sample_weights, val_all = Y, reverse_sign = reverse_sign, shift_back = shift_back)
    
    saveloss = [lossval, loss2, lossorigval, beginning_loss, 0]
    save_model(model, outname+'_params0.pth')
    save_model(model, outname+'_parameter.pth')   
    
    if writeloss:
        writebeginning = str(round(lossorigval,4))+'\t'+str(round(lossval,4))+'\t'+str(round(beginning_loss,4))+'\t'+str(round(loss2,4))
        save_losses(outname+'_loss.txt', 0, writebeginning)
    if verbose:
        print(0, writebeginning)
    stopcriterion = stop_criterion(checkval, patience)
    early_stop, stopexp = stopcriterion(0, lossval)
    
    # Start epochs and updates
    restarted = 0
    e = 0
    
    
    while True:
        
        trainloss, loss2 = excute_epoch(model, dataloader, loss_func, pwm_out, trainsize, True, device, val_loss = val_loss, optimizer = optimizer, l1reg_last = l1reg_last, l2reg_last = l2reg_last, l1_kernel = l1_kernel, last_layertensor = last_layertensor, kernel_layertensor = kernel_layertensor, sample_weights = sample_weights, val_all = Y, reverse_sign = reverse_sign, shift_back = shift_back)
        
        model.eval() # Sets model to evaluation mode which is important for batch normalization over all training mean and for dropout to be zero
        e += 1
        
        lossorigval, lossval = excute_epoch(model, val_dataloader, loss_func, pwm_outval, valsize, False, device, val_loss = val_loss, optimizer = None, l1reg_last = 0, l2reg_last = 0, l1_kernel = 0, last_layertensor = None, kernel_layertensor = None, sample_weights = None, val_all = Yval, reverse_sign = False, shift_back = 0)
        
        
        
        if e%write_steps == 0 or e < init_epochs:
            if writeloss:
                save_losses(outname+'_loss.txt', e, str(round(lossorigval,4))+'\t'+str(round(lossval,4))+'\t'+str(round(trainloss,4))+'\t'+str(round(loss2,4)))
            if verbose:
                print(e, str(round(lossorigval,4))+'\t'+str(round(lossval,4))+'\t'+str(round(trainloss,4))+'\t'+str(round(loss2,4)))
            
        model.train() # Set model modules back to training mode
        
        early_stop, stopexp = stopcriterion(e, lossval)
        
        if (np.isnan(lossval) or np.isnan(loss2) or (trainloss > beginning_loss)) and (e > 0) and (e < init_epochs) and init_adjust:
            # reduces learning rate if training loss goes up actually
            # need something learnable for each layer during training
            restarted += 1
            if restarted > 15:
                break
            save_losses(outname+'_loss.txt', 0, writebeginning)
            load_model(model, outname+'_params0.pth',device)
            e = 0
            
            for a, adict in enumerate(a_dict):
                a_dict[a]['lr'] = adict['lr'] *0.25
            
            if verbose:
                print(optimizer)
            

        elif e == epochs or early_stop:
            # stop after reaching maximum epoch number
            if early_stop and verbose:
                # Early stopping if validation loss is not improved for patience epochs
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
            # if no early stopping is False, save loss and parameters for the first few epochs and afterward every patience epoch
            if ((e%write_steps == 0) or (e<patience)) and ~np.isnan(lossval) and lossval < saveloss[0]:
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





# execute one epoch with training or validation set. Training set takes gradient but validation set computes loss without gradient
def excute_epoch(model, dataloader, loss_func, pwm_out, normsize, take_grad, device, val_loss = None, optimizer = None, l1reg_last = 0, l2reg_last = 0, l1_kernel = 0, last_layertensor = None, kernel_layertensor = None, sample_weights = None, val_all = None, reverse_sign = False, shift_back = 0):
    if val_loss is None:
        val_loss = loss_func
    
    trainloss = 0.
    validatloss = 0.
    # if size of validation set different from batchsize, then do val_all
    if val_all is not None:
        Ypred_all = torch.empty(val_all.size())
    tsize = 1
    if reverse_sign:
        tsize = tsize*2
    if shift_back > 0:
        tsize = tsize*2*shift_back
    
    for sample_x, sample_y, index in dataloader:
        
        if pwm_out is None:
            saddx = None
        else:
            saddx = pwm_out[index]
        
        # Add samples with reverse sign in X and Y
        if reverse_sign:
            sample_x2 = -sample_x
            sample_y2 = -sample_y
            sample_x = torch.cat([sample_x, sample_x2], dim = 0)
            sample_y = torch.cat([sample_y, sample_y2], dim = 0)
            if saddx is not None:
                saddx = torch.cat([saddx, -saddx], dim = 0)
            
        # Add samples that are shifted by 1 to 'shift_back' positions
        if shift_back > 0:
            sample_xs = [sample_x]
            sample_ys = [sample_y]
            saddxs = [saddx]
            for sb in range(1, shift_back+1):
                sample_x2 = torch.zeros_like(sample_x)
                sample_x2[..., :sb] = sample_x[..., -sb:]
                sample_x2[..., sb:] = sample_x[..., :-sb]
                sample_xs.append(sample_x2)
                sample_x2 = torch.zeros_like(sample_x)
                sample_x2[..., -sb:] = sample_x[..., :sb]
                sample_x2[..., :-sb] = sample_x[..., sb:]
                sample_xs.append(sample_x2)
                if saddx is not None:
                    saddx2 = torch.zeros_like(saddx)
                    saddx2[..., :sb] = saddx[..., -sb:]
                    saddx2[..., sb:] = saddx[..., :-sb]
                    saddxs.append(saddx2)
                    saddx2 = torch.zeros_like(saddx)
                    saddx2[..., -sb:] = saddx[..., :sb]
                    saddx2[..., :-sb] = saddx[..., sb:]
                    saddxs.append(saddx2)
                sample_ys.append(torch.cat([sample_y, sample_y],dim = 0))
            sample_x = torch.cat(sample_xs, dim = 0)
            sample_y = torch.cat(sample_ys, dim = 0)
            if saddx is not None:
                saddx = torch.cat(saddxs, dim = 0)
            
        if saddx is not None:
            saddx = saddx.to(device)
        sample_x, sample_y = sample_x.to(device), sample_y.to(device)
        
        if take_grad:
            optimizer.zero_grad()
            Ypred = model.forward(sample_x, xadd = saddx)
            loss = loss_func(Ypred, sample_y)
            if sample_weights is not None:
                loss = loss*sample_weights[index][:,None]
            loss = torch.sum(loss)
            trainloss += float(loss.item())
            
            if l1reg_last > 0 and last_layertensor is not None:
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
            validatloss += float(torch.sum(val_loss(Ypred_all, val_all)).item())
    trainloss /= (normsize*tsize)
    validatloss /= normsize
    return trainloss, validatloss




