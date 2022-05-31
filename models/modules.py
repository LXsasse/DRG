import sys, os 
import numpy as np
import torch.nn as nn
import torch
from collections import OrderedDict
from torch import Tensor
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F



func_dict = {'ReLU': nn.ReLU(), 
             'GELU': nn.GELU(),
             'Sigmoid': nn.Sigmoid(),
             'Tanh': nn.Tanh(),
             'Id0': nn.Identity(),
             'Softmax' : nn.Softmax()}

class fcc_convolution(nn.Module):
    def __init__(self, n_in, n_out, l_kernel, n_layers, connect_function = 'GELU'):
        self.weight = nn.Parameter(torch.rand(n_in, n_out, l_kernel))
        
    def forward(self, x):
        lx = x.size(dim = -1)
        # maybe work with 2d convolutions
        # Include customized Convolutions: 
    # CNN network for each convolution, can be interpreted as one complex motif, should not sum over all positions but instead put them into a fully connected network and only sum at the end. So that this network creates outputs for each position 


# Second layer of gapped convolutions for interactions (e.g. 10,20,30.. gap, 5 conv each side, 6 convolutions for each)
# Gapped convolutions have gap between two convolutional filters: 
class gap_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_gap, stride=1):
        super(gap_conv, self).__init__()
        # kernel_size defines the size of two kernels on each side of the gap
        self.kernel_gap = kernel_gap
        self.kernel_size = kernel_size
        self.leftcov = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias = True)
        self.rightcov = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias = True)
        
    def forward(self, x):
        """
        Forward propagation of a batch.
        """
        outleft = self.leftcov(x[:,:,:-self.kernel_gap-self.kernel_size]) # returns tensor of shape nseq, lseq-lker+1, nkernel
        outright = self.rightcov(x[:,:,self.kernel_gap+self.kernel_size:])
        #print(outleft.size(), outright.size())
        out = outleft + outright
        return out

# parallel module execute a list of modules with the same input and concatenate their output after flattening the last dimensions
class parallel_module(nn.Module):
    def __init__(self, modellist):
        super(parallel_module, self).__init__()
        self.modellist = nn.ModuleList(modellist)
        
    def forward(self, x):
        out = []
        for m in self.modellist:
            outadd = m(x)
            out.append(torch.flatten(m(x), start_dim = 1, end_dim = -1))
            
        out = torch.cat(out, dim = 1)
        return out


# Interaction module creates non-linear interactions between all features by multiplying them with each other and then multiplies a weight matrix to them
class interaction_module(nn.Module):
    def __init__(self, indim, outdim):
        super(interaction_module, self).__init__()
        self.outdim = outdim # if outdim is 1 then use softmax output
        # else use RelU
        self.indim = indim
        self.lineara = nn.Linear(indim, outdim, bias = False)
        self.linearb = nn.Linear((indim*(indim-1))/2, outdim, bias = False)
        self.classes = classes
    def forward(self, barray):
        insquare = torch.bmm(torch.unsqueeze(barray,-1), torch.unsqueeze(barray,-2))
        loc = torch.triu_indices(self.indim, self.indim, 1)
        insquare = insquare[:, loc[0], loc[1]]
        # could also concatenate and then use linear, then bias would be possible but all parameters would get same regularization treatment
        outflat = self.lineara(barray)
        outsquare = self.linearb(insquare.flatten(-2,-1)) # flatten
        out = outflat + outsquare
        return out

# Options:
# maxpooling
# mean_pooling
# conv_pooling
# Conv pooling with softmax over positions can get you attention pooling 

# Custom pooling layer that can max and mean pool 
class pooling_layer(nn.Module):
    def __init__(self, max_pooling, mean_pooling = False, conv_pooling = False, pooling_size = None, stride = None):
        super(pooling_layer, self).__init__()
        self.mean_pooling = mean_pooling
        self.max_pooling = max_pooling
        self.conv_pooling = conv_pooling

        if stride is None:
            stride = pooling_size
        
        if mean_pooling and max_pooling:
            self.poola = nn.AvgPool1d(pooling_size, stride=stride)
            self.poolb = nn.MaxPool1d(pooling_size, stride=stride)
            
        elif max_pooling and not mean_pooling:
            self.pool = nn.MaxPool1d(pooling_size, stride=stride)
        
        elif mean_pooling and not max_pooling:
            self.pool = nn.AvgPool1d(pooling_size, stride=stride)

        elif conv_pooling:
            self.pool = nn.Conv1d(insize, insize, kernel_size = pooling_size, stride = stride, bias = False)
        
    def forward(self, barray):
        if self.mean_pooling and self.max_pooling:
            return torch.cat((self.poola(barray), self.poolb(barray)), dim = -2)
        else:
            return self.pool(barray)

# Custom class that performs convolutions within the pooling size, then pools it to avoid a large intermediate matrix. 
class pooled_conv():
    def __init__(self):
        return NotImplementedError

# Module that computes the correlation as a loss function 
class correlation_loss(nn.Module):
    def __init__(self, dim = 0, reduction = 'mean'):
        super(correlation_loss, self).__init__()
        self.dim = dim
        self.reduction = reduction
    
    def forward(self, outy, tary):
        if self.dim == 0 :
            outy = outy - torch.mean(outy, dim = self.dim)[None, :]
            tary = tary - torch.mean(tary, dim = self.dim)[None, :]
        else:
            outy = outy - torch.mean(outy, dim = self.dim)[:, None]
            tary = tary - torch.mean(tary, dim = self.dim)[:,None]
        out = 1. - torch.sum(outy*tary, dim = self.dim)/torch.sqrt(torch.sum(outy**2, dim = self.dim) * torch.sum(tary**2, dim = self.dim))
        if self.reduction != 'none':
            if self.reduction == 'mean':
                out = multi * torch.mean(out)
            if self.reduction == 'sum':
                out = torch.sum(out)
        return out

# Loss with correlation along x and y axis, cannot use reduce
class correlation_both(nn.Module):
    def __init__(self, reduction = 'mean'):
        super(correlation_both, self).__init__()
        self.reduction = reduction
    def forward(self, outy, tary):
        outymean0, tarymean0 = outy - torch.mean(outy, dim = 0), tary - torch.mean(tary, dim = 0)
        outymean1, tarymean1 = outy - torch.mean(outy, dim = 1)[:,None], tary - torch.mean(tary, dim = 1)[:,None]
        if self.reduction == 'sum':
            multi = float(outy.size(dim=0))
        else:
            multi = 1.
        return multi * torch.mean(1. - torch.sum(outymean1*tarymean1, dim = 1)/torch.sqrt(torch.sum(outymean1**2, dim = 1) * torch.sum(tarymean1**2, dim = 1))) + torch.mean(1. - torch.sum(outymean0*tarymean0, dim = 0)/torch.sqrt(torch.sum(outymean0**2, dim = 0) * torch.sum(tarymean0**2, dim = 0)))
        
# Cosine loss along defined dimension
class cosine_loss(nn.Module):
    def __init__(self, dim = 0, reduction = 'mean'):
        super(cosine_loss, self).__init__()
        self.dim = dim
        self.reduction = reduction
    def forward(self, outy, tary):
        out = 1. - torch.sum(outy*tary, dim = self.dim)/torch.sqrt(torch.sum(outy**2, dim = self.dim) * torch.sum(tary**2, dim = self.dim))
        if self.reduction != 'none':
            if self.reduction == 'mean':
                out = multi * torch.mean(out)
            if self.reduction == 'sum':
                out = torch.sum(out)
        return out

# Cosine loss module for rows and columns
class cosine_both(nn.Module):
    def __init__(self, reduction = 'mean'):
        super(cosine_both, self).__init__()
        self.reduction = reduction
    def forward(self, outy, tary):
        if self.reduction == 'sum':
            multi = float(outy.size(dim=0))
        else:
            multi = 1.
        return multi * torch.mean(1. - torch.sum(outy*tary, dim = 0)/torch.sqrt(torch.sum(outy**2, dim = 0) * torch.sum(tary**2, dim = 0))) + torch.mean(1. - torch.sum(outy*tary, dim = 1)/torch.sqrt(torch.sum(outy**2, dim = 1) * torch.sum(tary**2, dim = 1)))

# Dummy loss for unpenalized regression
class zero_loss(nn.Module):
    def __init__(self, **kwargs):
        super(zero_loss, self).__init__()
        
    def forward(self, outy, tary):
        return 0.

# non-linearly converts the output by b*exp(ax) + c*x^d
# does not work yet, need probably a significantly smaller learning rate

class Complex(nn.Module):
    def __init__(self, outclasses):
        super(Complex, self).__init__()
        self.variables = Parameter(torch.ones(1, outclasses, 2))
        self.exponents = Parameter(torch.zeros(1, outclasses, 2))
        
    def forward(self, pred):
        x1 = torch.exp(pred)
        pred2 = torch.cat([pred.unsqueeze(-1), x1.unsqueeze(-1)], dim = -1)
        pred2 = pred2**torch.log2(2+self.exponents)
        pred2 = pred2 * self.variables
        pred = torch.sum(torch.cat([pred.unsqueeze(-1), pred2], dim =-1), dim = -1)
        return pred


# Creates several layers for each class seperately:
# So if you have 100 outclasses it generates linear layers for each of the 100 classes
# So if 500 dim come in and 550 should go out, the weights will be 500,100,550
# IF it is already expanded, so 100,500 come in, the weights will be 500,100,550 as well but each of the 100 classes has their own weight
class Expanding_linear(nn.Module):
    def __init__(self, indim, extradim, bias = True):
        super(Expanding_linear, self).__init__()
        self.param_size = [1]
        self.param_size.append(int(indim))
        self.scale = np.copy(indim)
        if isinstance(extradim, list):
            self.param_size += extradim
            self.scale += extradim[-1]
        else:
            self.param_size.append(extradim)
            self.scale += extradim
        self.scale = 1./np.sqrt(self.scale/2.)
        self.weight = Parameter((torch.rand(self.param_size)-0.5) * self.scale)
        if bias:
            self.bias = Parameter((torch.rand(self.param_size[2:])-0.5) * self.scale)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        x = x[(...,)+(None,)*(len(self.weight.size())-len(x.size()))]
        if x.size(dim = 1) != self.weight.size(dim =1):
            x = x.transpose(1,2)
        pred = torch.sum(x*self.weight, dim = 1)
        if self.bias is not None:
            pred += self.bias
        return pred.squeeze(-1)

    


class Res_FullyConnect(nn.Module):
    def __init__(self, indim, outdim = None, n_classes = None, n_layers = 1, layer_widening = 1., batch_norm = False, dropout = 0., activation_function = 'ReLU', residual_after = 1, bias = True):
        super(Res_FullyConnect, self).__init__()
        # Initialize fully connected layers
        self.nfcs = nn.ModuleDict()
        self.act_function = func_dict[activation_function]
        self.n_layers = n_layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.residual_after = residual_after
        if outdim is None:
            outdim = indim
        #self.layer_widening = 1.2 # Factor by which number of parameters are increased for each layer
        # Fully connected layers are getting wider and then back smaller to the original size
        # For-loops for getting wider
        currdim = np.copy(indim)
        resdim = np.copy(indim)
        for n in range(int(self.n_layers/2.)):
            currdim2 = int(layer_widening*currdim)
            if n_classes is None:
                self.nfcs['Fullyconnected'+str(n)] = nn.Linear(currdim, currdim2, bias = bias)
            else:
                self.nfcs['MultiFullyconnected'+str(n)] = Expanding_linear(currdim, [n_classes, currdim2], bias = bias)
            if self.batch_norm:
                self.nfcs['Bnorm_fullyconnected'+str(n)] = nn.BatchNorm1d(currdim2)
            if self.dropout > 0:
                self.nfcs['Dropout_fullyconnected'+str(n)] = nn.Dropout(p=self.dropout)
            self.nfcs['Actfunc'+str(n)] = self.act_function
            if residual_after > 0 and (n+1)%residual_after == 0:
                if n_classes is None:
                    self.nfcs['Residuallayer'+str(n)] = nn.Linear(resdim, currdim2, bias = False)
                else:
                    self.nfcs['MultiResiduallayer'+str(n)] = Expanding_linear(resdim, [n_classes, currdim2], bias = False)
                resdim = currdim2
            currdim = currdim2

        # for-loops for center layer, if odd number of layers
        for n in range(int(self.n_layers/2.), int(self.n_layers/2.)+int(self.n_layers%2.==1)):
            if n_classes is None:
                self.nfcs['Fullyconnected'+str(n)] = nn.Linear(currdim, currdim, bias = bias)
            else:
                self.nfcs['MultiFullyconnected'+str(n)] = Expanding_linear(currdim, [n_classes, currdim], bias = bias)
            
            if self.batch_norm:
                self.nfcs['Bnorm_fullyconnected'+str(n)] = nn.BatchNorm1d(currdim)
            if self.dropout > 0:
                convlayers['Dropout_fullyconnected'+str(n)] = nn.Dropout(p=self.dropout)
            self.nfcs['Actfunc'+str(n)] = self.act_function
            if residual_after > 0 and (n+1)%residual_after == 0:
                if n_classes is None:
                    self.nfcs['Residuallayer'+str(n)] = nn.Linear(resdim, currdim, bias = False)
                else:
                    self.nfcs['MultiResiduallayer'+str(n)] = Expanding_linear(resdim, [n_classes, currdim], bias = False)
                resdim = currdim
            
        # for loops with decreasing number of features
        for n in range(int(self.n_layers/2.)+int(self.n_layers%2.==1), int(self.n_layers)):
            if n == int(self.n_layers) -1:
                currdim2 = outdim
            else:
                currdim2 = int(currdim/layer_widening)
            if n_classes is None:
                self.nfcs['Fullyconnected'+str(n)] = nn.Linear(currdim, currdim2, bias = bias)
            else:
                self.nfcs['MultiFullyconnected'+str(n)] = Expanding_linear(currdim, [n_classes, currdim2], bias = bias)
            if self.batch_norm:
                self.nfcs['Bnorm_fullyconnected'+str(n)] = nn.BatchNorm1d(currdim2)
            if self.dropout> 0:
                convlayers['Dropout_fullyconnected'+str(n)] = nn.Dropout(p=self.dropout)
            self.nfcs['Actfunc'+str(n)] = self.act_function
            if residual_after > 0 and (n+1)%residual_after == 0:
                if n_classes is None:
                    self.nfcs['Residuallayer'+str(n)] = nn.Linear(resdim, currdim2, bias = False)
                else:
                    self.nfcs['MultiResiduallayer'+str(n)] = Expanding_linear(resdim, [n_classes, currdim2], bias = False)
                resdim = currdim2
            currdim = currdim2
    
    def forward(self, x):
        if self.residual_after > 0:
            res = x
        pred = x
        for key, item in self.nfcs.items():
            if "Residuallayer" in key:
                pred = pred + item(res)
                res = pred
            else:
                pred = item(pred)
        return pred
    
    
class Residual_convolution(nn.Module):
    def __init__(self, resdim, currdim, pool_lengths):
        super(Residual_convolution, self).__init__()
        self.rconv = OrderedDict()
        self.compute_residual = False
        if resdim != currdim:
            self.rconv['conv'] = nn.Conv1d(resdim, currdim, kernel_size = 1, bias = False, stride = 1)
            self.compute_residual = True
        if len(pool_lengths) > 0:
            plen, slen = pool_lengths
            for p, pl in enumerate(plen):
                self.rconv['pool'+str(p)] = nn.AvgPool1d(pl, stride = slen[p]) 
            self.compute_residual = True
        self.rconv = nn.Sequential(self.rconv)
        
    def forward(self, x):
        if self.compute_residual:
            return self.rconv(x)
        return x

class Res_Conv1d(nn.Module):
    def __init__(self, indim, inlen, n_kernels, l_kernels, n_layers, kernel_increase = 1., max_pooling = 0, mean_pooling=0, residual_after = 1, activation_function = 'ReLU', strides = 1, dilations = 1, bias = True, dropout = 0., batch_norm = False):
        super(Res_Conv1d, self).__init__()
        self.residual_after = residual_after
        self.convlayers = nn.ModuleDict()
        self.kernel_function = func_dict[activation_function]
        
        if isinstance(l_kernels, list) or isinstance(l_kernels, np.ndarray):
            l_kernels = np.array(l_kernels)
        else:
            l_kernels = np.ones(n_layers, dtype = int)*l_kernels
        
        if isinstance(strides, list) or isinstance(strides, np.ndarray):
            strides = np.array(strides)
        else:
            strides = np.ones(n_layers, dtype = int)*strides
        
        if isinstance(dilations, list) or isinstance(dilations, np.ndarray):
            dilations = np.array(dilations)
        else:
            dilations = np.ones(n_layers, dtype = int)*dilations
        
        currdim, currlen = np.copy(indim), np.copy(inlen)
        if residual_after > 0:
            resdim = np.copy(currdim)
        reswindows = [[],[]]
        for n in range(n_layers):
            
            if int(np.floor((currlen - dilations[n]*(l_kernels[n]-1)-1)/strides[n]+1)) <= 0:
                break
            if batch_norm:
                self.convlayers['Bnorm'+str(n)] = self.nn.BatchNorm1d(currdim)
                
            self.convlayers['Conv'+str(n)] = nn.Conv1d(currdim, int(currdim*kernel_increase), kernel_size = l_kernels[n], bias = bias, stride = strides[n], dilation = dilations[n])
            self.convlayers['Conv_func'+str(n)] = self.kernel_function
            reswindows[0].append(l_kernels[n])
            reswindows[1].append(strides[n])
            currdim = int(currdim*kernel_increase)
            currlen = int((currlen - dilations[n]*(l_kernels[n]-1)-1)/strides[n])+1
            
            if max_pooling > 0 or mean_pooling > 0:
                if max_pooling > currlen or mean_pooling > currlen:
                    if residual_after > 0 and (n+1)%residual_after == 0:
                        self.convlayers['ResiduallayerConv'+str(n)] = Residual_convolution(resdim, currdim, reswindows)
                        reswindows = [[],[]]
                        resdim = np.copy(currdim)
                    if dropout > 0.:
                        self.convlayers['Dropout_Convs'+str(n)] = nn.Dropout(p=dropout)
                    break
                self.convlayers['Poolingconvs'+str(n)] = pooling_layer(max_pooling > 0, mean_pooling > 0, pooling_size = max(max_pooling, mean_pooling), stride=max(max_pooling, mean_pooling))
                reswindows[0].append(max(max_pooling, mean_pooling))
                reswindows[1].append(max(max_pooling, mean_pooling))
                currlen = int(1. + (currlen - (max(max_pooling, mean_pooling)-1)-1)/max(max_pooling, mean_pooling))
                currdim = (int(max_pooling > 0) + int(mean_pooling>0)) * currdim
            if residual_after > 0 and (n+1)%residual_after == 0:
                self.convlayers['ResiduallayerConv'+str(n)] = Residual_convolution(resdim, currdim, reswindows)
                reswindows = [[],[]]
                resdim = np.copy(currdim)
                
            if dropout > 0.:
                self.convlayers['Dropout_Convs'+str(n)] = nn.Dropout(p=dropout)
        self.currdim, self.currlen = currdim, currlen
    
    def forward(self,x):
        if self.residual_after > 0:
            res = x
        pred = x
        for key, item in self.convlayers.items():
            if "Residuallayer" in key:
                residual = item(res)
                pred = pred + residual
                res = pred
            else:
                pred = item(pred)
        return pred

# if attention should not spread along the entire sequence, use mask to set entries beyond a certain distance to zero
class receptive_matmul(nn.Module):
    def __init__(self, l_seq, receptive, multi_head = True):
        super(receptive_attention, self).__init__()
        self.l_seq = l_seq
        self.mask = torch.ones(l_seq, l_seq)
        self.mask = torch.tril(torch.triu(self.mask,-receptive),receptive)
        if multi_head:
            self.mask = self.mask.view(1,1,l_seq,l_seq,1)
        else:
            self.mask = self.mask.view(1,l_seq,l_seq,1)
        self.multi_head = multi_head
    def forward(self, queries, keys):
        if self.multi_head:
            atmat = torch.sum(queries.unsqueeze(-2).expand(queries.size(dim = 0), queries.size(dim = 1), self.l_seq, self.l_seq, -1)* self.mask* keys.unsqueeze(0).expand(keys.size(dim = 0), keys.size(dim = 1),self.l_seq, self.l_seq, -1), -1)
        else:
            atmat = torch.sum(queries.unsqueeze(-2).expand(queries.size(dim = 0), self.l_seq, self.l_seq, -1)* self.mask* keys.unsqueeze(0).expand(keys.size(dim = 0), self.l_seq, self.l_seq, -1), -1)
        return atmat
        
# Include Feed forward with RELUs and residual around them
# Include batchnorm after residuals
class MyAttention_layer(nn.Module):
    def __init__(self, indim, dim_embedding, n_heads, dim_values = None, dropout = 0., bias = False, residual = False, sum_out = False, positional_embedding = True, posdim = None, batchnorm = False, layernorm = True, Linear_layer = True, Activation = 'GELU', receptive_field = None):
        super(MyAttention_layer, self).__init__()
        
        
        if dim_values is None:
            self.dim_values = dim_embedding
        else:
            self.dim_values = dim_values
        
        if posdim is None:
            posdim = indim
        self.posdim = posdim
        self.n_heads = n_heads
        self.dim_embedding = dim_embedding
        self.sum_out = sum_out
        self.residual = residual
        self.positional_embedding = positional_embedding
        self.pos_queries = None
        self.receptive_field = receptive_field
        self.receptive_matmul = None
        
        # Generate embedding for each head
        dim_embedding = self.n_heads * dim_embedding
        dim_values = self.n_heads *self.dim_values
        
        if self.positional_embedding:
            self.embedd_queries = nn.Conv1d(indim+posdim, dim_embedding, 1, bias = False)
            self.embedd_keys = nn.Conv1d(indim+posdim, dim_embedding, 1, bias = False)
        else:    
            self.embedd_queries = nn.Conv1d(indim, dim_embedding, 1, bias = False)
            self.embedd_keys = nn.Conv1d(indim, dim_embedding, 1, bias = False)
        
        self.embedd_values = nn.Conv1d(indim, dim_values, 1, bias = False)
        
        
        if self.sum_out:
            self.combine_layer = nn.Conv1d(self.dim_values, self.dim_values, 1, bias = False)
        else:
            self.combine_layer = nn.Conv1d(self.dim_values*self.n_heads, self.dim_values, 1, bias = False)
        
        
        if self.residual:
            if self.dim_values == indim:
                self.reslayer = nn.Identity()
            else:
                self.reslayer = nn.Conv1d(indim, self.dim_values, 1, bias = False)
        
        self.dropout = dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bnorm = nn.BatchNorm1d(indim)
        
        self.layernorm = layernorm
        if self.layernorm:
            self.layer_norm = nn.LayerNorm(self.dim_values)
        
        self.Linear_layer = Linear_layer
        if self.Linear_layer:
            self.feedforward = nn.Sequential(nn.Conv1d(self.dim_values, self.dim_values*3, 1, bias = False), func_dict[Activation], nn.Conv1d(self.dim_values*3, self.dim_values, 1, bias = False))
            if self.layernorm:
                self.feedforward_layer_norm = nn.LayerNorm(self.dim_values)
            
        
    def pos_embedding_function(self, dim, length):
        dsin = int(dim/2)
        dcos = dim - dsin
        wavelengths = (length/2)**((torch.arange(dsin)+10)/dsin).unsqueeze(-1)
        xpos = torch.arange(length).unsqueeze(0)
        sines = torch.sin(2*np.pi*xpos/wavelengths)
        cosines = torch.cos(2*np.pi*xpos/wavelengths)
        posrep = torch.cat([sines,cosines], dim = 0)
        return posrep
        
    def forward(self,x):
        if self.batchnorm:
            x = self.bnorm(x)
        if self.positional_embedding:
            if self.pos_queries is None:
                self.pos_queries = self.pos_embedding_function(self.posdim, x.size(dim = -1))
                if self.embedd_queries.weight.is_cuda:
                    devicetobe = self.embedd_queries.weight.get_device()
                    self.pos_queries = self.pos_queries.to('cuda:'+str(devicetobe))
            bsize = x.size(dim = 0)
            qpred = self.embedd_queries(torch.cat((x, self.pos_queries.expand(bsize,-1,-1)),dim = 1))
            kpred = self.embedd_keys(torch.cat((x, self.pos_queries.expand(bsize,-1,-1)),dim = 1))
        else:
            qpred = self.embedd_queries(x)
            kpred = self.embedd_keys(x)
        vpred = self.embedd_values(x)
        
        # split into n_heads
        qpred = qpred.view(qpred.size(dim = 0), self.n_heads, -1, qpred.size(dim = -1))
        kpred = kpred.view(kpred.size(dim = 0), self.n_heads, -1, kpred.size(dim = -1))
        vpred = vpred.view(vpred.size(dim = 0), self.n_heads, -1, vpred.size(dim = -1))
        # compute attention matrix
        
        qpred = qpred.transpose(-1,-2)
        if receptive_field is not None:
            # Only has none-zero elements within receptive field but needs for loop unfortunately
            if self.receptive_matmul is None:
                self.receptive_matmul = receptive_matmul(qpred.size(-1), self.receptive_field)
                devicetobe = self.qpred.get_device()
                self.receptive_matmul.to('cuda:'+str(devicetobe))
            attmatix = self.receptive_matmul(qpred, kpred)
        else:
            attmatrix = torch.matmul(qpred, kpred)
        attmatrix /= np.sqrt(self.dim_embedding)
        # compute softmax
        soft = nn.Softmax(dim = -1)
        attmatrix = soft(attmatrix)
        # compute mixture of values from attention
        attmatrix = torch.matmul(attmatrix, vpred.transpose(-1,2)).transpose(-2,-1)
        
        if self.sum_out:
            pred = torch.sum(attmatrix, dim = 1)
        else:
            pred = torch.flatten(attmatrix, start_dim = 1, end_dim = 2)
        
        if self.dropout >0:
            pred = self.dropout_layer(pred)
        
        
        pred = self.combine_layer(pred)
        
        if self.residual:
            pred = pred + self.reslayer(x)
        
        if self.layernorm:
            pred = self.layer_norm(pred.transpose(-1,-2))
            pred = pred.transpose(-2,-1)
        
        if self.Linear_layer:
            if self.residual:
                res = pred
            if self.dropout:
                pred = self.dropout_layer(pred)
            
            pred = self.feedforward(pred)
            
            if self.residual:
                pred = pred + res
            
            if self.layernorm:
                pred = self.feedforward_layer_norm(pred.transpose(-1,-2))
                pred = pred.transpose(-1,-2)
        
        return pred
    
 



# Returns a stretching and adds bias for each kernel dimension after convolution
# Also good example how write own module with any tensor multiplication and initialized parameters
class Kernel_linear(nn.Module):
    def __init__(self, n_kernels: int) -> None:
        super(Kernel_linear, self).__init__()
        self.n_kernels = n_kernels
        self.weight = Parameter(torch.empty((1, n_kernels, 1), **factory_kwargs))
        self.bias = Parameter(torch.empty(n_kernels, **factory_kwargs))
        self.init_parameters()
        
    def init_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1. / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inp: Tensor) -> Tensor:
        onp = inp*self.weight + self.bias[:, None]
        return onp

# class to access attributes after using parallel module to train on multiple gpu at the same time
class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)

# dictionary with loss functions
loss_dict = {'MSE':nn.MSELoss(reduction = 'none'), 
             'L1loss':nn.L1Loss(reduction = 'none'),
             'CrossEntropy': nn.CrossEntropyLoss(reduction = 'none'),
             'BCE': nn.BCELoss(reduction = 'none'),
             'KL': nn.KLDivLoss(reduction = 'none'),
             'PoissonNNL':nn.PoissonNLLLoss(reduction = 'none'),
             'GNLL': nn.GaussianNLLLoss(reduction = 'none'),
             'NLL': nn.NLLLoss(reduction = 'none'),
             'Cosinedata': cosine_loss(dim = 1, reduction = 'none'),
             'Cosineclass': cosine_loss(dim = 0, reduction = 'none'),
             'Cosineboth': cosine_both(reduction = 'none'),
             'Correlationdata': correlation_loss(dim = 1, reduction = 'none'),
             'Correlationclass': correlation_loss(dim = 0, reduction = 'none'),
             'Correlationboth': correlation_both(reduction = 'none')}



