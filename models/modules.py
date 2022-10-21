import sys, os 
import numpy as np
import torch.nn as nn
import torch
from collections import OrderedDict
from torch import Tensor
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F



class EXPmax(nn.Module):
    def __init__(self, crop_max = 128):
        super(EXPmax, self).__init__()
        self.cmax = np.log(crop_max)
        self.thresh = nn.Threshold(-self.cmax, -self.cmax)
    def forward(self, x):
        x = -self.thresh(-x)
        return torch.exp(x)

class POWmax(nn.Module):
    def __init__(self, crop_max = 128, exponent = 2):
        super(POWmax, self).__init__()
        self.cmax = crop_max**(1./exponent)
        self.thresh = nn.Threshold(-self.cmax, -self.cmax)
        self.exponent = exponent
    def forward(self, x):
        x = -self.thresh(-x)
        return torch.pow(x, self.exponent)

    

func_dict = {'ReLU': nn.ReLU(), 
             'GELU': nn.GELU(),
             'Sigmoid': nn.Sigmoid(),
             'Tanh': nn.Tanh(),
             'Id0': nn.Identity(),
             'Softmax' : nn.Softmax(),
             'EXPmax' : EXPmax(crop_max = 128),
             'EXP': EXPmax(crop_max = 2**32),
             'SQUARED': POWmax(exponent = 2),
             'THIRD': POWmax(exponent = 3)}

class fcc_convolution(nn.Module):
    def __init__(self, n_in, n_out, l_kernel, n_layers, connect_function = 'GELU'):
        self.weight = nn.Parameter(torch.rand(n_in, n_out, l_kernel))
        
    def forward(self, x):
        lx = x.size(dim = -1)
        # maybe work with 2d convolutions
        # Include customized Convolutions: 
    # CNN network for each convolution, can be interpreted as one complex motif, should not sum over all positions but instead put them into a fully connected network and only sum at the end. So that this network creates outputs for each position 


# Module that computes the correlation as a loss function 
class correlation_loss(nn.Module):
    def __init__(self, dim = 0, reduction = 'mean', eps = 1e-8):
        super(correlation_loss, self).__init__()
        self.dim = dim
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, outy, tary):
        outy = outy - torch.mean(outy, dim = self.dim).unsqueeze(self.dim)
        tary = tary - torch.mean(tary, dim = self.dim).unsqueeze(self.dim)
            
        out = 1. - (torch.sum(outy*tary, dim = self.dim)+self.eps)/(self.eps+torch.sqrt(torch.sum(outy**2, dim = self.dim) * torch.sum(tary**2, dim = self.dim)))
        if self.reduction != 'none':
            if self.reduction == 'mean':
                out = torch.mean(out)
            if self.reduction == 'sum':
                out = torch.sum(out)
        else:
            out = out.unsqueeze(self.dim).expand(outy.size())
        return out

# Loss with correlation along x and y axis, cannot use reduce
class correlation_both(nn.Module):
    def __init__(self, reduction = 'mean', ratio = 0.5):
        super(correlation_both, self).__init__()
        self.reduction = reduction
        self.ratio = ratio
        self.correlation0 = correlation_loss()
        self.correlation1 = correlation_loss(dim = 1)
    
    def forward(self, outy, tary):
        if self.reduction == 'none' or self.reduction == 'sum':
            multi = float(np.prod(outy.size()))
        else:
            multi = 1.
        return multi *(self.ratio* self.correlation1(outy, tary) + (1.-self.ratio)*self.correlation0(outy,tary))


class correlation_mse(nn.Module):
    def __init__(self, reduction = 'mean', ratio = 0.95, dimcorr = 1):
        super(correlation_mse, self).__init__()
        self.reduction = reduction
        self.ratio = ratio
        self.correlation0 = nn.MSELoss(reduction = 'mean')
        self.correlation1 = correlation_loss(dim = dimcorr)
    
    def forward(self, outy, tary):
        if self.reduction == 'none' or self.reduction == 'sum':
            multi = float(np.prod(outy.size()))
        else:
            multi = 1.
        return multi *(self.ratio* self.correlation1(outy, tary) + (1.-self.ratio)*self.correlation0(outy,tary))

        
# Cosine loss along defined dimension
class cosine_loss(nn.Module):
    def __init__(self, dim = 0, reduction = 'mean'):
        super(cosine_loss, self).__init__()
        self.dim = dim
        self.reduction = reduction
    def forward(self, outy, tary):
        
        # Does not work if one variance of one is zero, sets all the parameters to nan after backpropagation then
        out = 1. - torch.sum(outy*tary, dim = self.dim)/torch.sqrt(torch.sum(outy**2, dim = self.dim) * torch.sum(tary**2, dim = self.dim))
        if self.reduction != 'none':
            if self.reduction == 'mean':
                out = torch.mean(out)
            if self.reduction == 'sum':
                out = torch.sum(out)
        else:
            out = out.unsqueeze(self.dim).expand(outy.size())
        return out

# Cosine loss module for rows and columns
class cosine_both(nn.Module):
    def __init__(self, reduction = 'mean', ratio = 0.5):
        super(cosine_both, self).__init__()
        self.reduction = reduction
        self.ratio = ratio
        self.cosine0 = cosine_loss()
        self.cosine1 = cosine_loss(dim = 1)
    def forward(self, outy, tary):
        if self.reduction == 'none' or self.reduction == 'sum':
            multi = float(np.prod(outy.size()))
        else:
            multi = 1.
        return multi * (self.ratio *self.cosine1(outy, tary) + (1.-self.ratio)*self.cosine0(outy, tary))

# Dummy loss for unpenalized regression
class zero_loss(nn.Module):
    def __init__(self, **kwargs):
        super(zero_loss, self).__init__()
        
    def forward(self, outy, tary):
        return 0.

# include possibility to use mse in window size Z, e.g 25 bp windows
class JSD(nn.Module):
    def __init__(self, sum_axis = -1, norm_last = True, reduction = 'none', eps = 1e-8, include_mse = True, mse_ratio = 10., mean_size = 25):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)
        self.mse = None
        if include_mse:
            self.mse = nn.MSELoss(reduction = reduction)
            self.mean_size = mean_size
            self.meanpool = None    
        self.mse_ratio = mse_ratio
        self.sum_axis = sum_axis
        self.norm_last = norm_last
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, p: torch.tensor, q: torch.tensor):
        if self.mse is not None:
            if self.mean_size is None:
                self.mean_size = p.size(dim = -1)
            if self.meanpool is None:
                self.meanpool = nn.AvgPool1d(self.mean_size, padding = int((p.size(dim = -1)%self.mean_size)/2), count_include_pad = False)
                self.l_out = int(np.ceil(p.size(dim = -1)/self.mean_size))
            pn = self.meanpool(p).repeat(1,1,self.mean_size)
            qn = self.meanpool(q).repeat(1,1,self.mean_size)
        
        # ONLY USE JSD if values are GREATER than ZERO. Make sure by using RELU, Sigmoid, or Softmax function
        #p = p-torch.min(p,dim =-1)[0].unsqueeze(dim=-1)
        #q = q-torch.min(q,dim =-1)[0].unsqueeze(dim=-1)
        p, q = p + self.eps, q + self.eps
        if self.norm_last:
            normp, normq = p.sum(dim = -1)[...,None], q.sum(dim = -1)[...,None]
            #normp[normp == 0] = 1.
            #normq[normq == 0] = 1.
            p = p/normp
            q = q/normq
        m = (0.5 * (p + q)).log()
        p = p.log()
        q = q.log()
        kl = 0.5 * (self.kl(m, p) + self.kl(m, q)) 
        if self.sum_axis is not None:
            klsize = kl.size()
            if self.sum_axis == -1:
                self.sum_axis = len(klsize) -1
            kl = kl.sum(dim = self.sum_axis)
            if self.reduction == 'none':
                kl = kl.unsqueeze(self.sum_axis).expand(klsize) #self.expand)
        if self.mse is not None:
            kl = kl + self.mse_ratio* self.mse(pn, qn)
        return kl


class BCEMSE(nn.Module):
    def __init__(self, reduction = 'none', log_counts = True, eps = 1, mse_ratio = 10., mean_size = None):
        super(BCEMSE, self).__init__()
        self.bce = nn.BCELoss(reduction=reduction)
        self.mse = nn.MSELoss(reduction = reduction)
        self.mean_size = mean_size
        self.meanpool = None    
        self.mse_ratio = mse_ratio
        self.log_counts = log_counts
        self.eps = eps
        
    def forward(self, p: torch.tensor, q: torch.tensor):
        if self.mean_size is None:
            self.mean_size = p.size(dim = -1)
        if self.meanpool is None:
            self.meanpool = nn.AvgPool1d(self.mean_size, padding = int((p.size(dim = -1)%self.mean_size)/2), count_include_pad = False)
        pn = self.meanpool(p).repeat(1,1,self.mean_size)
        qn = self.meanpool(q).repeat(1,1,self.mean_size)
        if self.log_counts:
            pn = (pn+self.eps).log()
            qn = (qn+self.eps).log()
        p = p-torch.min(p,dim =-1)[0].unsqueeze(dim=-1)
        q = q-torch.min(q,dim =-1)[0].unsqueeze(dim=-1)
        normp, normq = p.sum(dim = -1)[...,None], q.sum(dim = -1)[...,None]
        normp[normp == 0] = 1.
        normq[normq == 0] = 1.
        p = p/normp
        q = q/normq
        loss = self.bce(p,q) + self.mse_ratio* self.mse(pn, qn)
        return loss

class LogMSELoss(nn.Module):
    def __init__(self, reduction = 'none', eps = 1., log_prediction = False):
        super(LogMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction = reduction)
        self.eps = eps
    def forward(self, p, q):
        minq = torch.min(q,dim =-1)[0]
        q = q-minq.unsqueeze(-1)
        q =torch.log(q+self.eps)
        if self.log_prediction:
            p = p-minp.unsqueeze(-1)
            minp = torch.min(p,dim =-1)[0]
            p =torch.log(p+self.eps)
        return self.mse(p,q)
    
class LogL1Loss(nn.Module):
    def __init__(self, reduction = 'none', eps = 1.):
        super(LogL1Loss, self).__init__()
        self.mse = nn.L1Loss(reduction = reduction)
        self.eps = eps
    def forward(self, p, q):
        minp = torch.min(p,dim =-1)[0]
        minq = torch.min(q,dim =-1)[0]
        p = p-minp.unsqueeze(-1)
        q = q-minq.unsqueeze(-1)
        p =torch.log(p+self.eps)
        q =torch.log(q+self.eps)
        return self.mse(p,q)   
        
class LogCountDistLoss(nn.Module):
    def __init__(self, reduction = 'none', eps = 1., log_counts = False, sum_counts = True, ratio = 10.):
        super(LogCountDistLoss, self).__init__()
        self.mse = nn.MSELoss(reduction = reduction)
        self.eps = eps
        self.log_counts = log_counts
        self.sum_counts = sum_counts
        self.ratio = ratio
    def forward(self, p, q):
        minp = torch.min(p,dim =-1)[0]
        minq = torch.min(q,dim =-1)[0]
        pn = p-minp.unsqueeze(-1)
        qn = q-minq.unsqueeze(-1)
        normp, normq = pn.sum(dim = -1).unsqueeze(-1), qn.sum(dim = -1).unsqueeze(-1)
        normp[normp == 0] = 1.
        normq[normq == 0] = 1.
        pn = pn/normp
        qn = qn/normq
        if self.log_counts:
            p =torch.log(p+self.eps)
            q =torch.log(q+self.eps)
        if self.sum_counts:
            psize = p.size()
            p = p.mean(dim = -1).unsqueeze(-1).expand(psize)
            q = q.mean(dim = -1).unsqueeze(-1).expand(psize)
        return self.mse(p,q) + self.ratio * self.mse(pn,qn)

# dictionary with loss functions
basic_loss_dict = {'MSE':nn.MSELoss(reduction = 'none'), 
             'L1Loss':nn.L1Loss(reduction = 'none'),
             'CrossEntropy': nn.CrossEntropyLoss(reduction = 'none'),
             'BCELoss': nn.BCELoss(reduction = 'none'),
             'BCEMSE': BCEMSE(reduction = 'none'),
             'KL': nn.KLDivLoss(reduction = 'none'),
             'PoissonNNL':nn.PoissonNLLLoss(reduction = 'none'),
             'GNLL': nn.GaussianNLLLoss(reduction = 'none'),
             'NLL': nn.NLLLoss(reduction = 'none'),
             'Cosinedata': cosine_loss(dim = 1, reduction = 'none'),
             'Cosineclass': cosine_loss(dim = 0, reduction = 'none'),
             'Cosineboth': cosine_both(reduction = 'none'),
             'Correlationdata': correlation_loss(dim = -1, reduction = 'none'),
             'Correlationclass': correlation_loss(dim = 0, reduction = 'none'),
             'Correlationmse': correlation_mse(reduction = 'none', dimcorr = -1),
             'MSECorrelation': correlation_mse(reduction = 'none', dimcorr = 0),
             'Correlationboth': correlation_both(reduction = 'none')}

class ContrastiveLoss(nn.Module):
    def __init__(self, mainloss = 'MSE', contrastive_loss = 'MSE', contrastive_metric = 'Dif'):
        super(ContrastiveLoss, self).__init__()
        self.main = basic_loss_dict[mainloss]
        self.cont = basic_loss_dict[contrastive_loss]
        self.contrastive_metric = contrastive_metric
    def forward(self, p,q):
        l = self.main(p,q)
        # loss across all differences between tracks from multi-task learning
        if self.contrastive_metric == 'Dif':
            pdif, qdif = p.unsqueeze(-2) - p.unsqueeze(-1), q.unsqueeze(-2) - q.unsqueeze(-1)
        elif self.contrastive_metric == 'Frac' or self.contrastive_metric == 'Logfrac':
            # need to subtract minimum first
            pqmin = min(torch.min(p), torch.min(q))
            pdif, qdif = p-pqmin, q-pqmin
            pdif, qdif = pdif + 0.1, qdif + 0.1
            pdif, qdif = pdif.unsqueeze(-2) / pdif.unsqueeze(-1), qdif.unsqueeze(-2) / qdif.unsqueeze(-1)
            if self.contrastive_metric == 'Logfrac':
                pdif, qdif = torch.log(pdif + 1.), torch.log(qdif + 1.)
        
        lc = self.cont(pdif, qdif).mean(-1)
        return l + lc
    
# try an error that measures the error of counts and the error of probabilities within one sequence    

# non-linearly converts the output by b*exp(ax) + c*x^d
# does not work yet, need probably a significantly smaller learning rate

class Complex(nn.Module):
    def __init__(self, outclasses):
        super(Complex, self).__init__()
        self.variables = Parameter(torch.ones(1, outclasses, 2))
        self.exponents = Parameter(torch.zeros(1, outclasses, 2))
        self.exp = EXPmax()
    def forward(self, pred):
        x1 = self.exp(pred)
        pred2 = torch.cat([pred.unsqueeze(-1), x1.unsqueeze(-1)], dim = -1)
        pred2 = pred2**torch.log2(2+self.exponents)
        pred2 = pred2 * self.variables
        pred = torch.sum(torch.cat([pred.unsqueeze(-1), pred2], dim =-1), dim = -1)
        return pred



class simple_multi_output(nn.Module):
    def __init__(self, out_relation = 'DIFFERENCE', pre_function = None, log = None, logoffset = None, offset = 1e-12):
        super(simple_multi_output, self).__init__()
        if logoffset is None and log is not None:
            self.logoffset = log
            self.log = log
        else:
            self.logoffset = logoffset
            self.log = log
        if self.log is not None:
            self.log = torch.log(torch.tensor(log))
        self.out_relation = out_relation # difference, expdifference, FRACTION, LOGXPLUSFRACTION, LOGXPLUSDIFFERENCE
        self.pre_function = pre_function
        if self.pre_function is not None:
            self.pre_function = func_dict[pre_function]
        self.offset = offset
        if 'EXPDIFFERENCE' in self.out_relation.upper():
            self.exp = EXPmax(crop_max = 2048)


    def forward(self, x):
        xa, xb = x[0], x[1]
        if self.pre_function is not None:
            xa, xb = self.pre_function(xa), self.pre_function(xb)
        
        if 'DIFFERENCE' == self.out_relation.upper():
            x = xb - xa
        elif 'DIFFERENCE' in self.out_relation.upper():
            x = xa - xb
        elif 'FRACTION' in self.out_relation.upper():
            x = (xa+self.offset)/(xb+self.offset)
        
        if 'EXPDIFFERENCE' in self.out_relation.upper():
            x = self.exp(-x)
        
        if 'LOGXPLUS' in self.out_relation.upper():
            x = torch.log(self.logoffset + x)
            if self.log is not None:
                x = x/self.log
        return x


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

    


# add: if size does not change between layers then don't perform extra linear layer for residual
# special case of residuals between layers that have the same number of features (U-net? )
class Res_FullyConnect(nn.Module):
    def __init__(self, indim, outdim = None, n_classes = None, n_layers = 1, layer_widening = 1., batch_norm = False, dropout = 0., activation_function = 'GELU', residual_after = 1, bias = True):
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
            if self.batch_norm:
                self.nfcs['Bnorm_fullyconnected'+str(n)] = nn.BatchNorm1d(currdim)
            if n_classes is None:
                self.nfcs['Fullyconnected'+str(n)] = nn.Linear(currdim, currdim2, bias = bias)
            else:
                self.nfcs['MultiFullyconnected'+str(n)] = Expanding_linear(currdim, [n_classes, currdim2], bias = bias)
            
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
            if self.batch_norm:
                self.nfcs['Bnorm_fullyconnected'+str(n)] = nn.BatchNorm1d(currdim)
            
            if n_classes is None:
                self.nfcs['Fullyconnected'+str(n)] = nn.Linear(currdim, currdim, bias = bias)
            else:
                self.nfcs['MultiFullyconnected'+str(n)] = Expanding_linear(currdim, [n_classes, currdim], bias = bias)
            
            if self.dropout > 0:
                self.nfcs['Dropout_fullyconnected'+str(n)] = nn.Dropout(p=self.dropout)
            self.nfcs['Actfunc'+str(n)] = self.act_function
            if residual_after > 0 and (n+1)%residual_after == 0:
                if n_classes is None:
                    self.nfcs['Residuallayer'+str(n)] = nn.Linear(resdim, currdim, bias = False)
                else:
                    self.nfcs['MultiResiduallayer'+str(n)] = Expanding_linear(resdim, [n_classes, currdim], bias = False)
                resdim = currdim
            
        # for loops with decreasing number of features
        for n in range(int(self.n_layers/2.)+int(self.n_layers%2.==1), int(self.n_layers)):
            if self.batch_norm:
                self.nfcs['Bnorm_fullyconnected'+str(n)] = nn.BatchNorm1d(currdim)
            if n == int(self.n_layers) -1:
                currdim2 = outdim
            else:
                currdim2 = int(currdim/layer_widening)
            if n_classes is None:
                self.nfcs['Fullyconnected'+str(n)] = nn.Linear(currdim, currdim2, bias = bias)
            else:
                self.nfcs['MultiFullyconnected'+str(n)] = Expanding_linear(currdim, [n_classes, currdim2], bias = bias)
            if self.dropout> 0:
                self.nfcs['Dropout_fullyconnected'+str(n)] = nn.Dropout(p=self.dropout)
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
    
# This average pooling can use dilation and also include padding that is larger than half of the kernel_size to cover kernel_size*dilation/2
class Padded_AvgPool1d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation = 1, count_include_pad=True):
        super(Padded_AvgPool1d, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        #print('stride', stride)
        self.dilation = dilation
        self.padding = padding
        self.gopad = False
        if isinstance(self.padding, int):
            if self.padding > 0:
                self.padding = (padding, padding)
                self.gopad = True
        elif isinstance(self.padding, list):
            self.gopad = True
        self.count_include_pad = count_include_pad
        self.register_buffer('weight', torch.ones(1,1,1,kernel_size)/kernel_size)
        self.register_buffer('norm',None)
    def forward(self, x):
        xs = x.size()
        #print(xs)
        if self.norm is None:
            #print(self.norm, dict(self.named_buffers()))
            self.norm = torch.ones(xs[1:]).unsqueeze(0).unsqueeze(0)
            #print(self.norm, dict(self.named_buffers()))
            #print('normsize', self.norm.size())
            if self.gopad:
                if self.count_include_pad:
                    val = 1
                else:
                    val = 0
                self.norm = F.pad(self.norm, self.padding, value = val)
            if self.weight.is_cuda and not self.norm.is_cuda:
                devicetobe = self.weight.get_device()
                self.norm = self.norm.to('cuda:'+str(devicetobe))
            self.norm = F.conv2d(self.norm, self.weight, stride = (1,self.stride), dilation = (1,self.dilation))
            #print('fnormsize', self.norm.size(), self.norm[0,0,0])
        if self.gopad:
            x = F.pad(x, self.padding)
        #print(x.size())
        x = F.conv2d(x.unsqueeze(1), self.weight, stride = (1,self.stride), dilation = self.dilation)
        #print(x.size(), self.norm.size())
        x = x/self.norm
        x = x.squeeze(1)
        #print('fxsize', x.size())
        return x
            
        
    
class Residual_convolution(nn.Module):
    def __init__(self, resdim, currdim, pool_lengths):
        super(Residual_convolution, self).__init__()
        rconv = OrderedDict()
        # if empty list is given as pool_lengths and resdim == currdim then no module is initiated and self.compute_residual is set to False, which then returns the input itself
        self.compute_residual = False
        if resdim != currdim:
            rconv['conv'] = nn.Conv1d(resdim, currdim, kernel_size = 1, bias = False, stride = 1)
            self.compute_residual = True
        if len(pool_lengths) > 0:
            for p, plen in enumerate(pool_lengths):
                pl, slen, pool_type, pad, dilation = plen
                if pool_type == 'Avg' or pool_type == 'Mean' or pool_type == 'mean':
                    rconv['pool'+str(p)] = Padded_AvgPool1d(pl, stride = slen, padding = pad, dilation = dilation, count_include_pad = False) 
                elif pool_type == 'Max' or pool_type == 'MAX' or pool_type[p] == 'max':
                    gopad = False
                    if isinstance(pad, int):
                        if pad > 0:
                            pad = (pad, pad, 0, 0)
                            gopad = True
                    elif isinstance(pad,list):
                        pad = pad + (0,0)
                        gopad = True
                    if gopad:
                        rconv['Padding'] = nn.ZeroPad2d(pad)
                    rconv['pool'+str(p)] = nn.MaxPool1d(pl, stride = slen, dilation = dilation)
                
            self.compute_residual = True
        self.rconv = nn.Sequential(rconv)
        
    def forward(self, x):
        if self.compute_residual:
            return self.rconv(x)
        return x

class Padded_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, bias=True, padding_mode='zeros', value = None, reverse_complement = False):
        super(Padded_Conv1d, self).__init__()
        self.in_channels = in_channels, 
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding 
        if isinstance(self.padding, int):
            self.padding = [padding, padding]
        self.dilation = dilation
        self.bias = bias
        self.padding_mode = padding_mode
        self.value = value
        if padding_mode == 'zeros':
            self.padding_mode = 'constant'
            self.value = 0
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, bias=bias)
        self.reverse_complement = reverse_complement
        
    def forward(self, x):
        if self.padding is not None:
            x = F.pad(x, self.padding, mode = self.padding_mode, value = self.value)
        if self.reverse_complement:
            xr = torch.flip(self.conv1d(torch.flip(x, dims = [-2,-1])), dims = [-2])
            xf = self.conv1d(x)
            x = torch.amax(torch.cat([xf.unsqueeze(0), xr.unsqueeze(0)], dim = 0), dim = 0)
        else:
            x = self.conv1d(x)
        return x
 
# Reverse complement convolution that uses both + and minus strand and combines them with maxpooling
class RC_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, stride_rows = None, padding=None, dilation=1, bias=True, padding_mode='zeros', value = None, reverse_complement = True):
        ###Uses 2D convolution for 8 row representation of on reverse complement
        super(RC_Conv1d, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride 
        if stride_rows is None:
            self.stride_rows = in_channels
        else:
            self.stride_rows = stride_rows
       
        self.reverse_complement = reverse_complement
        self.padding = padding 
        if isinstance(self.padding, int):
            self.padding = [padding, padding]
        self.dilation = dilation
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.bias = None
            
        self.padding_mode = padding_mode
        self.value = value
        if padding_mode == 'zeros':
            self.padding_mode = 'constant'
            self.value = 0
        # use functional conv2d to perform dilation in only one direction
        self.weight = Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.init_parameters()
        
    def init_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1. / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        if self.padding is not None:
            x = F.pad(x, self.padding, mode = self.padding_mode, value = self.value)
        # reverse complement row is read in the opposite direction by the tf, therefore, we would need to reverse the weights of the kernels in the last dimension and then read it from left to right.
        # Instead we can flip the direction of the input and the flip the output back so that the positions are properly aligned again.
        if self.reverse_complement:
            x = torch.cat([x, torch.flip(x,dims = 1)])
            xf = F.conv2d(x[:,:int(x.size(dim =1)/2)].unsqueeze(1), self.weight.unsqueeze(1), bias = self.bias, stride = (self.stride_rows, self.stride), dilation = (1, self.dilation))
            xb = torch.flip(F.conv2d(torch.flip(x[:,-int(x.size(dim =1)/2):].unsqueeze(1), dims = [-1]), self.weight.unsqueeze(1), bias = self.bias, stride = (self.stride_rows, self.stride), dilation = (1, self.dilation)) , dims = [-1])
            x = torch.cat([xf, xb], dim = 2)
            x = torch.amax(x,dim = 2)
        else:   
            x = F.conv2d(x.unsqueeze(1), self.weight.unsqueeze(1), bias = self.bias, stride = (self.stride_rows, self.stride), dilation = (1, self.dilation))
            x = torch.flatten(x,start_dim = 1, end_dim = 2)
        return x
 
 
# Tower of residual dilated convolution blocks
class Res_Conv1d(nn.Module):
    def __init__(self, indim, inlen, n_kernels, l_kernels, n_layers, kernel_increase = 1., max_pooling = 0, mean_pooling=0, residual_after = 1, residual_same_len = False, activation_function = 'GELU', strides = 1, dilations = 1, bias = True, dropout = 0., batch_norm = False, act_func_before = True, residual_entire = False):
        '''
        indim: number of channels of input at axis 1
        inlen: len of sequence of input at axis 2
        n_kernels: number of kernels that should be used in every layer
        l_kernels: length of kernels that should be used in every layer
        kernel_increase: multiplier that increases the number of kernels
        max_pooling: if > 0 then maxpooling with stride and size
        mean_pooling: if > 0 then meanpooling with stride and size
        residual_after: after how many conv layers a residual should be added
        residual_same_len: If True, even if the convolutional layer does not change the dimension of the output to the input, a mean pooling will be performed that meanpools in the size and stride as the convolutional layer
        '''
        super(Res_Conv1d, self).__init__()
        self.residual_after = residual_after
        self.convlayers = nn.ModuleDict()
        self.kernel_function = func_dict[activation_function]
        self.residual_entire = residual_entire
        
        kernel_increase = np.ones(n_layers)*kernel_increase
        if n_kernels != indim:
            kernel_increase[0] = n_kernels/indim
            
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
        resdim = np.copy(currdim)
        reslen = np.copy(currlen)
        reswindows = []
        
        if residual_entire:
            resentire = []
            resedim = np.copy(currdim)
            
        for n in range(n_layers):
            # early stopping criteria if one chose too many layers and the sequence got shorter than the kernel length
            if currlen < l_kernels[n]:
                break
            # batch norm before providing the sequence to the convolution
            if batch_norm:
                self.convlayers['Bnorm'+str(n)] = self.nn.BatchNorm1d(currdim)
            # decide if activation function should be applied before or after convolutional layer
            if act_func_before:
                self.convlayers['Conv_func'+str(n)] = self.kernel_function
            
            # (non-symmetric) padding to have same padding left and right of sequence and get same sequence length
            convpad = [int(np.floor((dilations[n]*(l_kernels[n]-1)+1)/2))-int((dilations[n]*(l_kernels[n]-1)+1)%2==0), int(np.floor((dilations[n]*(l_kernels[n]-1)+1)/2))]
            # padded convolutional layer
            self.convlayers['Conv'+str(n)] = Padded_Conv1d(currdim, int(currdim*kernel_increase[n]), kernel_size = l_kernels[n], bias = bias, stride = strides[n], dilation = dilations[n], padding = convpad)
            # see above
            if not act_func_before:
                self.convlayers['Conv_func'+str(n)] = self.kernel_function
            # compute the new dimension and length of after the convolution
            currdim = int(currdim*kernel_increase[n])
            currlen = int(np.floor((currlen +convpad[0]+convpad[1]- dilations[n]*(l_kernels[n]-1)-1)/strides[n]+1))
            # if the reslen is different from the currlen, due to strides > 1, then perform mean pooling to adujust residual
            if residual_same_len or reslen != currlen:
                reswindows.append([l_kernels[n], strides[n], 'Mean', convpad, dilations[n]])
            # if a residual from before conv block should be added to output after the last convolution, from start to end basically then collect all dimensionality changes.
            if residual_entire and (residual_same_len or reslen != currlen):
                resentire.append(reswindows[-1])
            
            
            if (residual_after > 0 and (n+1)%residual_after == 0):
                print('Residuals introduced', reswindows, resdim, currdim)
                self.convlayers['ResiduallayerConv'+str(n)] = Residual_convolution(resdim, currdim, reswindows)
                reswindows = []
                resdim = np.copy(currdim)
                reslen = np.copy(currlen)
                
            if dropout > 0.:
                self.convlayers['Dropout_Convs'+str(n)] = nn.Dropout(p=dropout)
            
            if max_pooling > 0 or mean_pooling > 0:
                if max_pooling > currlen or mean_pooling > currlen:
                    break
                maxpad = int(np.ceil((max(max_pooling, mean_pooling) - currlen%max(max_pooling, mean_pooling))/2))*int(currlen%max(max_pooling, mean_pooling)>0)
                self.convlayers['Poolingconvs'+str(n)] = pooling_layer(max_pooling > 0, mean_pooling > 0, pooling_size = max(max_pooling, mean_pooling), stride=max(max_pooling, mean_pooling), padding = maxpad)
                
                currdim = (int(max_pooling > 0) + int(mean_pooling>0)) * currdim
                currlen = int(np.ceil(currlen/max(max_pooling, mean_pooling)))
                
                if residual_after > 0:
                    if max_pooling > 0:
                        reswindows.append([max(max_pooling, mean_pooling),max(max_pooling, mean_pooling),'Max', maxpad, 1])
                    else:
                        reswindows.append([max(max_pooling, mean_pooling),max(max_pooling, mean_pooling),'Mean',maxpad,1])
                reslen = currlen
                if residual_entire:
                    resentire.append(reswindows[-1])
                
                
                
        if residual_entire:
            self.residual_entire = Residual_convolution(resedim, currdim, resentire)
        else:
            self.residual_entire = None
        self.currdim, self.currlen = currdim, currlen
        
    def forward(self,x):
        if self.residual_entire is not None:
            res0 = x
        if self.residual_after > 0:
            res = x
        pred = x
        for key, item in self.convlayers.items():
            if "Residuallayer" in key:
                residual = item(res)
                res = pred
                pred = pred + residual
            else:
                pred = item(pred)
            
        if self.residual_entire is not None:
            pred = pred + self.residual_entire(res0)
        return pred






# Second layer of gapped convolutions for interactions (e.g. 10,20,30.. gap, 5 conv each side, 6 convolutions for each)
# Gapped convolutions have gap between two convolutional filters: 
class gap_conv(nn.Module):
    def __init__(self, in_channels, in_len, out_channels, kernel_size, kernel_gap, stride=1, pooling = False, residual = False, batch_norm = False, dropout= 0., edge_effect = 'maintain', activation_function = 'GELU'):
        super(gap_conv, self).__init__()
        # kernel_size defines the size of two kernels on each side of the gap
        self.kernel_gap = kernel_gap
        self.kernel_size = kernel_size
        self.edge_effect = edge_effect # reduce, maintain, expand
        
        if batch_norm:
            #batchnorm before giving input to 
            self.batch_norm = nn.BarchNorm1d(in_channels)
        
        left_padding = [int(np.floor(kernel_size/2))-int(kernel_size%2==0), int(np.floor(kernel_size/2))]
        right_padding = [int(np.floor(kernel_size/2)), int(np.floor(kernel_size/2))-int(kernel_size%2==0)]
        padding = 2*int(np.floor(kernel_size/2))-int(kernel_size%2==0)
        
        # Use costum moduel because torch.nn.Conv1d cannot deal with different pooling sizes left and right of the sequence
        self.leftcov = Padded_Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias = True, padding = left_padding)
        self.rightcov = Padded_Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias = True, padding = right_padding)
        
        self.out_len = int(np.floor((in_len + padding - kernel_size)/stride +1))
        # max pooling before layers are flattened to reduce dimension of output given to fully connected layer
        self.pooling = None
        if pooling == True:
            poolstride = int(kernel_size/2)
        elif pooling > 1:
            poolstride = pooling
            pooling = True
        
        if pooling: # pooling max
            # pooling_padding makes sure that some positions are not ignored in pooling, simlar to ceil_mode but with symmetric effect
            pooling_padding = int(np.ceil((kernel_size - self.out_len%int(kernel_size/2))*int(self.out_len%int(kernel_size/2)>0)/2))
            self.pooling = nn.MaxPool1d(kernel_size, stride = poolstride, padding = pooling_padding)
            # when dimensions are pooled, the kernel_size and kernel_gap is reduced by the pooling size
            self.kernel_gap = int(self.kernel_gap/poolstride)
            self.kernel_size = int(self.kernel_size/poolstride)
            # pool with kernel_size but stride kernel_size/2
            # gap becomes gap*2/kernelsize 
            self.out_len = int((self.out_len+2*pooling_padding-kernel_size)/poolstride)+1
        
        # residuals can be generated for the gapped conv operation
        self.residual = None
        if residual:
            self.residual = OrderedDict()
            if in_channels != out_channels:
                self.residual['ResConv'] = nn.Conv1d(in_channels, out_channels, kernel_size = 1, bias = False)
            if not pooling:
                self.residual['AvgPool'] = nn.Identity()
            else:
                self.residual['AvgPool'] = nn.AvgPool1d(kernel_size, stride = poolstride, padding = pooling_padding, count_include_pad=False)
            self.residual = nn.Sequential(self.residual)
            #add residual to every to every gapped kernel 
        
        if self.edge_effect == 'reduce':
            self.out_len = self.out_len - self.kernel_gap - self.kernel_size
        elif self.edge_effect == 'expand':
            self.out_len = self.out_len + self.kernel_gap + self.kernel_size
        
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
            
        self.act_func = func_dict[activation_function]
        self.kernelgap_size = self.kernel_size + self.kernel_gap
        
    def forward(self, x):
        """
        Forward propagation of a batch.
        """
        outleft = self.leftcov(x)
        outright = self.rightcov(x)
        if self.pooling is not None:
            outleft = self.pooling(outleft)
            outright = self.pooling(outright)
        if self.edge_effect == 'expand':
            out = F.pad(outleft, (self.kernelgap_size,0)) + F.pad(outright, (0, self.kernelgap_size))
        elif self.edge_effect == 'reduce':
            out = outleft[:,:,:-self.kernel_gap-self.kernel_size] + outright[:,:,self.kernelgap_size:]
        else:
            out = F.pad(outleft, (int(self.kernelgap_size/2),0))[:,:,:-int(self.kernelgap_size/2)] + F.pad(outright, (0,int(self.kernelgap_size/2)))[:,:,int(self.kernelgap_size/2):]
            
        out = self.act_func(out)
        if self.residual is not None:
            res = self.residual(x)
            #if self.pooling is not None:
                #res = self.pooling(res)
            if self.edge_effect == 'expand':
                # instead of using this, we can just pad them on different sides
                res = F.pad(res, (self.kernelgap_size, 0)) + F.pad(res, (0, self.kernelgap_size))
            elif self.edge_effect == 'reduce':
                res = res[:,:,:-self.kernel_gap-self.kernel_size] + res[:,:,self.kernelgap_size:]
            else:
                res = F.pad(res, (int(self.kernelgap_size/2),0))[:,:,:-int(self.kernelgap_size/2)] + F.pad(res, (0,int(self.kernelgap_size/2)))[:,:,int(self.kernelgap_size/2):]
            out = out + res
        
        return out

# parallel module execute a list of modules with the same input and concatenate their output after flattening the last dimensions
class parallel_module(nn.Module):
    def __init__(self, modellist, flatten = True):
        super(parallel_module, self).__init__()
        self.modellist = nn.ModuleList(modellist)
        self.flatten = flatten
        if not self.flatten:
            outlens = len(np.unique([m.out_len for m in self.modellist])) == 1
            if not outlens:
                raise Warning("Module outputs cannot be concatenated along dim = -1")
        
    def forward(self, x):
        out = []
        for m in self.modellist:
            outadd = m(x)
            if self.flatten:
                out.append(torch.flatten(m(x), start_dim = 1, end_dim = -1))
            else:
                out.append(m(x))
        out = torch.cat(out, dim = 1)
        return out



class final_convolution(nn.Module):
    def __init__(self, indim, out_classes, l_kernels, cut_sites = None, strides = 1, bias = True, n_convolutions = 1, in_len = None, batch_norm = False, padding = 'same', predict_from_dist = False):
        super(final_convolution, self).__init__()
        
        self.batch_norm = batch_norm
        self.cut_sites = cut_sites
        if self.cut_sites is None:
            self.cut_sites = [0,0]
        elif isinstance(self.cut_sites,int):
            self.cut_sites = [cut_sites, cut_sites]
        else:
            self.cut_sites = cut_sites
        if batch_norm:
            self.Bnorm = self.nn.BatchNorm1d(currdim)

        self.n_convolutions = n_convolutions
        if n_convolutions > 1:
            if in_len is None:
                in_len = 10000 # dummy parameter that is not needed here because we're not max pooling
            dilations = (2**np.arange(n_convolutions -1)).astype(int)
            self.dilconvs = Res_Conv1d(indim, in_len, indim, l_kernels, n_convolutions-1, kernel_increase = 1., max_pooling = 0, mean_pooling=0, residual_after = 1, residual_same_len = False, activation_function = 'GELU', strides = 1, dilations = dilations, bias = True, dropout = 0., batch_norm = False, act_func_before = True, residual_entire = False)

        self.predict_from_dist = predict_from_dist
        if self.predict_from_dist:
            self.cpred = nn.Linear(indim, out_classes)
            self.spred = nn.Softmax(dim = -1)
        
        self.padding = padding 
        if isinstance(self.padding, int):
            self.padding = [padding, padding]
        if self.padding == 'same':
            self.padding = [int(np.floor(l_kernels/2))-int(l_kernels%2==0), int(np.floor(l_kernels/2))]
        
        self.fconvlayer = nn.Conv1d(indim, out_classes, kernel_size = l_kernels, bias = bias, stride = strides)
    
    def forward(self, x):
        if self.n_convolutions > 1:
            x = self.dilconvs(x)
        
        if self.predict_from_dist:
            mx = x.mean(dim = -1)
            mcounts = self.cpred(mx)
        if self.batch_norm:
            x = self.Bnorm(x)
        if self.padding is not None:
            x = F.pad(x, self.padding, mode = 'constant', value = 0)
        x = self.fconvlayer(x)
        x = x[..., self.cut_sites[0]:x.size(dim=-1)-self.cut_sites[1]]
        if self.predict_from_dist:
            x = self.spred(x)
            x = x * mcounts.unsqueeze(-1)
        return x


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
    def __init__(self, max_pooling, mean_pooling = False, conv_pooling = False, pooling_size = None, stride = None, padding = 0):
        super(pooling_layer, self).__init__()
        self.mean_pooling = mean_pooling
        self.max_pooling = max_pooling
        self.conv_pooling = conv_pooling

        if stride is None:
            stride = pooling_size
        
        if mean_pooling and max_pooling:
            self.poola = nn.AvgPool1d(pooling_size, stride=stride, padding = padding, count_include_pad = False)
            self.poolb = nn.MaxPool1d(pooling_size, stride=stride, padding = padding )
            
        elif max_pooling and not mean_pooling:
            self.pool = nn.MaxPool1d(pooling_size, stride=stride, padding = padding)
        
        elif mean_pooling and not max_pooling:
            self.pool = nn.AvgPool1d(pooling_size, stride=stride, padding = padding, count_include_pad = False)

        elif conv_pooling:
            self.pool = nn.Conv1d(insize, insize, kernel_size = pooling_size, stride = stride, bias = False, padding = padding)
        
    def forward(self, barray):
        if self.mean_pooling and self.max_pooling:
            return torch.cat((self.poola(barray), self.poolb(barray)), dim = -2)
        else:
            return self.pool(barray)

# Custom class that performs convolutions within the pooling size, then pools it to avoid a large intermediate matrix. 
class pooled_conv():
    def __init__(self):
        return NotImplementedError










# if attention should not spread along the entire sequence, use mask to set entries beyond a certain distance to zero
# Better would be if we could prevent computing the entries that are set to zero with the mask
class receptive_attention(nn.Module):
    def __init__(self, l_seq, receptive, multi_head = True):
        super(receptive_attention, self).__init__()
        self.l_seq = l_seq
        # mask is one within the receptive field
        mask = torch.ones(l_seq, l_seq)
        # and zero otherwise
        mask = torch.tril(torch.triu(mask,-receptive),receptive)
        # mask needs one more dimension if we have multiple heads
        # need to register the mask to sent it to the correct device with the model
        if multi_head:
            # dimensions are batch, n_heads, l_seq X l_seq, and 1 for the dimension of the embedding
            self.register_buffer('mask', mask.view(1,1,l_seq,l_seq,1))
        else:
            self.register_buffer('mask', mask.view(1,l_seq,l_seq,1))
        self.multi_head = multi_head
    
    def forward(self, queries, keys):
        # perform multiplication of keys and queries with diagonal elements outside the receptive field being set to zero
        # to do that, we create a l_seq X l_seq mat of the keys and queries by introducing a new dimension and expanding it by l_seq
        # then we set off diagonals to zero
        # hopefully this reduces time since off-diagonal elements don't need to be matrix multiplied
        # however, the extension of the dimension may result in high memory usage and maybe slow it down actually 
        # instead of matmul, we just sum over the embedding dimension
            #atmat = torch.sum(queries.unsqueeze(-2).expand(queries.size(dim = 0), queries.size(dim = 1), self.l_seq, self.l_seq, -1)* self.mask* keys.unsqueeze(-3).expand(keys.size(dim = 0), keys.size(dim = 1),self.l_seq, self.l_seq, -1), -1)
            # this might work better than expanding first
        atmat = torch.sum(queries.unsqueeze(-2)* self.mask* keys.unsqueeze(-3), -1)
        #else:
            #atmat = torch.sum(queries.unsqueeze(-2).expand(queries.size(dim = 0), self.l_seq, self.l_seq, -1)* self.mask* keys.unsqueeze(-3).expand(keys.size(dim = 0), self.l_seq, self.l_seq, -1), -1)
        return atmat
        
# Include Feed forward with RELUs and residual around them
# Include batchnorm after residuals
# Look at faster implementation of attention
class MyAttention_layer(nn.Module):
    def __init__(self, indim, dim_embedding, n_heads, dim_values = None, in_len = None, dropout = 0., bias = False, residual = True, sum_out = False, positional_embedding = True, posdim = None, batchnorm = False, layernorm = True, Linear_layer = True, Activation = 'GELU', receptive_field = None):
        super(MyAttention_layer, self).__init__()
        
        # keys and queries have naturaly a dimesion dim_embedding
        # while the values can have another dimension that is independent of the two embeddings
        if dim_values is None:
            self.dim_values = dim_embedding
        else:
            self.dim_values = dim_values
        
        # The dimension of the positional embedding 
        if posdim is None:
            posdim = indim
        self.posdim = posdim
        self.n_heads = n_heads # number of attention heads that are processed in parallel
        self.dim_embedding = dim_embedding # dimension of the the embedding keys and queries
        self.sum_out = sum_out # if True sums over the outputs representations from all heads instead of concatenating them
        self.residual = residual # if True residual will be added to output of attention layer
        self.positional_embedding = positional_embedding # if True then a positional embedding will be added to the input before the key and query embedding
        if in_len is not None:
            self.register_buffer('pos_queries', self.init_pos_embedding(self.posdim,in_len))
        else:   
            self.register_buffer('pos_queries', None) # pos_queries are the positional embedding that will be added
        
        self.receptive_field = receptive_field # if given, attention matrix will only be computed between positions that are within the receptive field distance ATTENTION: may not work yet.
        if in_len is not None and self.receptive_field is not None:
            self.receptive_matmul = receptive_attention(in_len, self.receptive_field)
        else:
            self.receptive_matmul = None # this is the receptive matmul function. However, we cannot inialize it here because we do not require the input len to be given. So it is initialized in the forward loop. ATTENTION: This procedure may cause problems when we change the device of the model later. 
        
        # Generate embedding for each head
        # if multiple heads are used, the embedding for each head is generated together. 
        dim_embedding = self.n_heads * dim_embedding
        dim_values = self.n_heads *self.dim_values
        
        # if positional embedding is used, the embedding convolution input needs to be extended by the dimension of the positional embedding
        if self.positional_embedding:
            self.embed_queries = nn.Conv1d(indim+posdim, dim_embedding, 1, bias = False)
            self.embed_keys = nn.Conv1d(indim+posdim, dim_embedding, 1, bias = False)
            self.embed_values = nn.Conv1d(indim+posdim, dim_values, 1, bias = False)
        else:    
            self.embed_queries = nn.Conv1d(indim, dim_embedding, 1, bias = False)
            self.embed_keys = nn.Conv1d(indim, dim_embedding, 1, bias = False)
            self.embed_values = nn.Conv1d(indim, dim_values, 1, bias = False)
        
        # combine layer is the linear layer that is used after concatenation of the heads in multi-head attention, here the output dimension is reduced to dim_values from n_heads*dim_values
        if self.sum_out:
            self.combine_layer = nn.Conv1d(self.dim_values, self.dim_values, 1, bias = False)
        else:
            self.combine_layer = nn.Conv1d(self.dim_values*self.n_heads, self.dim_values, 1, bias = False)
        
        # if residuals should be added then the input is directly added if the indim is equal to the output dimension, otherwise a convlution of width 1 is used to adjust dimensions
        if self.residual:
            if self.dim_values == indim:
                self.reslayer = nn.Identity()
            else:
                self.reslayer = nn.Conv1d(indim, self.dim_values, 1, bias = False)
        # dropout can be included at two different positions, before linear tranformations
        self.dropout = dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        
        # batch norm is performed to the input
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bnorm = nn.BatchNorm1d(indim)
        
        # layernorm is performed each time after residual was added
        self.layernorm = layernorm
        if self.layernorm:
            self.layer_norm = nn.LayerNorm(self.dim_values)
        
        # Fully connected feed forward network, see 3.3 in paper: Liner_layer applies two linear convolutions but expands the dimension by a factor 4 in the middle. 
        self.Linear_layer = Linear_layer
        if self.Linear_layer:
            self.feedforward = nn.Sequential(nn.Conv1d(self.dim_values, self.dim_values*4, 1, bias = False), func_dict[Activation], nn.Conv1d(self.dim_values*4, self.dim_values, 1, bias = False))
            if self.layernorm:
                self.feedforward_layer_norm = nn.LayerNorm(self.dim_values)
            
    
    # this function initializes the positional embedding that is constant and will not be updated
    def init_pos_embedding(self, dim, length):
        # number of dimensions represented by sines
        dsin = int(dim/2)
        # nubmer of dimensions represented by cosines
        dcos = dim - dsin
        # each dimension has a different wavelength with which the sine and cosine oszilate
        # from (length/2)**10/dsin to (length/2)**(10+dsin)/dsin
        # the orignal in the paper is 10000**2i/dsin, but this might be inadequate for the length of the sequences that we're using since this results in very long wavelengths and very little change
        sinwavelengths = (length/2)**((torch.arange(dsin)+10)/dsin).unsqueeze(-1)
        coswavelengths = (length/2)**((torch.arange(dcos)+10)/dcos).unsqueeze(-1)
        # for each position, there will be a different combination of sine and cosine values
        xpos = torch.arange(length).unsqueeze(0)
        sines = torch.sin(2*np.pi*xpos/sinwavelengths)
        cosines = torch.cos(2*np.pi*xpos/coswavelengths)
        posrep = torch.cat([sines,cosines], dim = 0)
        # posrep dimension: (dim, length)
        return posrep
        
    def forward(self,x):
        
        if self.batchnorm:
            x = self.bnorm(x)
        
        if self.residual:
            residual = self.reslayer(x)
        
        if self.positional_embedding:
            # if positional embedding dont exist yet, they are initialized in the first forward loop since we don't have the length of the sequence in the first place. 
            if self.pos_queries is None:
                self.pos_queries = self.init_pos_embedding(self.posdim, x.size(dim = -1))
                if self.embed_queries.weight.is_cuda:
                    devicetobe = self.embed_queries.weight.get_device()
                    self.pos_queries = self.pos_queries.to('cuda:'+str(devicetobe))
            bsize = x.size(dim = 0)
            # concatenate a dimensional embedding with every x to give to query and key
            x = torch.cat((x, self.pos_queries.unsqueeze(0).expand(bsize,-1,-1)),dim = 1)
        
        # embed x into keys queries and values
        qpred = self.embed_queries(x)
        kpred = self.embed_keys(x)
        vpred = self.embed_values(x)
        
        # split into n_heads
        qpred = qpred.view(qpred.size(dim = 0), self.n_heads, -1, qpred.size(dim = -1))
        kpred = kpred.view(kpred.size(dim = 0), self.n_heads, -1, kpred.size(dim = -1))
        vpred = vpred.view(vpred.size(dim = 0), self.n_heads, -1, vpred.size(dim = -1))
        # compute attention matrix
        
        
        if self.receptive_field is not None:
            # Only has none-zero elements within receptive field but may need too much memory 
            # the dimensions of the inputs may not be correct, need to be debugged.
            if self.receptive_matmul is None:
                self.receptive_matmul = receptive_attention(qpred.size(-1), self.receptive_field)
                if self.receptive_matmul.mask.is_cuda:
                    devicetobe = self.qpred.get_device()
                    self.receptive_matmul.to('cuda:'+str(devicetobe))
            attmatix = self.receptive_matmul(qpred, kpred)
        else:
            qpred = qpred.transpose(-1,-2)
            attmatrix = torch.matmul(qpred, kpred)
        # scale attention matrix
        attmatrix /= np.sqrt(self.dim_embedding)
        # compute softmax
        soft = nn.Softmax(dim = -1)
        attmatrix = soft(attmatrix)
        # compute mixture of values from attention
        attmatrix = torch.matmul(attmatrix, vpred.transpose(-1,2)).transpose(-2,-1)
        
        # sum over all heads
        if self.sum_out:
            pred = torch.sum(attmatrix, dim = 1)
        else:
            pred = torch.flatten(attmatrix, start_dim = 1, end_dim = 2)
        
        # apply dropout before linear layer
        if self.dropout >0:
            pred = self.dropout_layer(pred)
        
        # apply linear layer that combines all heads to dim_values
        pred = self.combine_layer(pred)
        
        # add residual
        if self.residual:
            pred = pred + residual 
        
        # perform layer norm
        if self.layernorm:
            pred = self.layer_norm(pred.transpose(-1,-2))
            pred = pred.transpose(-2,-1)
        
        # apply fully connected layer
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
loss_dict = basic_loss_dict | {'JSD': JSD(reduction = 'none', include_mse = False),
             'JensenShannon': JSD(reduction = 'none', mean_size = 25),
             'JensenShannonCount': JSD(reduction = 'none', mean_size = None),
             'DLogMSE': LogMSELoss(reduction = 'none', log_prediction = True),
             'LogMSE': LogMSELoss(reduction = 'none', log_prediction = False),
             'LogL1Loss': LogL1Loss(reduction = 'none'),
             'LogCountDistLoss': LogCountDistLoss(reduction = 'none', log_counts = True),
             'CountDistLoss': LogCountDistLoss(reduction = 'none'),
             'MSEcontrastMSED':ContrastiveLoss(mainloss = 'MSE', contrastive_loss = 'MSE', contrastive_metric = 'Dif'),
             'MSEcontrastMSEF':ContrastiveLoss(mainloss = 'MSE', contrastive_loss = 'MSE', contrastive_metric = 'Frac'),
             'L1contrastL1D':ContrastiveLoss(mainloss = 'L1Loss', contrastive_loss = 'L1Loss', contrastive_metric = 'Dif'),
             'L1contrastMSED':ContrastiveLoss(mainloss = 'L1Loss', contrastive_loss = 'MSE', contrastive_metric = 'Dif'),
             'MSEcontrastCORD':ContrastiveLoss(mainloss = 'MSE', contrastive_loss = 'Correlationdata', contrastive_metric = 'Dif'),
             'L1contrastCORD':ContrastiveLoss(mainloss = 'L1Loss', contrastive_loss = 'Correlationdata', contrastive_metric = 'Dif')}




