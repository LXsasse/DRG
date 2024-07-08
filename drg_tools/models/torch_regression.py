import sys, os 
import numpy as np
import scipy.stats as stats
from scipy.sparse.linalg import svds
from sklearn import linear_model, metrics
from sklearn.decomposition import SparsePCA
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
from collections import OrderedDict
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from init import MyDataset, get_device


# Missing: 
    # taking derivative for regularization parameter
        # understand torch.optim
        # either write own torch optimizer that uses backward to compute .grad
        # or have to use jax to avoid computation of grad multiple times
            # Problem: then we cannot use GPU and torch tensors.



# dictionary with loss functions
loss_dict = {'MSE':nn.MSELoss(reduction = 'none'), 
             'L1loss':nn.L1Loss(reduction = 'none'),
             'CrossEntropy': nn.CrossEntropyLoss(reduction = 'none'),
             'BCE': nn.BCELoss(reduction = 'none'),
             'KL': nn.KLDivLoss(reduction = 'none'),
             'PoissonNNL':nn.PoissonNLLLoss(reduction = 'none'),
             'GNLL': nn.GaussianNLLLoss(reduction = 'none'),
             'NLL': nn.NLLLoss(reduction = 'none')}

func_dict = {'ReLU': nn.ReLU(), 
             'GELU': nn.GELU(),
             'Sigmoid': nn.Sigmoid(),
             'Tanh': nn.Tanh(),
             'Id0': nn.Identity(),
             'Softmax' : nn.Softmax()}




# This is a simple linear model that uses SGD or Adam to solve multi-task LR
# But it can also train a one-layer CNN that first extracts kernels and then utilizes a liner model
# Interestingly, the model can be regularized and eventually, we want to also use gradient decent to learn the regularization
class torch_Regression(nn.Module):
    def __init__(self, alpha = 0.01, fit_intercept = True, loss_function = 'MSE', logistic = 'Linear', penalty = 'l1',  kernel_length = None, n_kernels = None, kernel_function = 'GELU', pooling = 'Max', pooling_length = None, alpha_kernels = 0.01, learn_alpha = False, is_zero = 0, epochs = 1000, lr = 0.001, optimizer = 'SGD', optim_params = None, batchsize = None, device = 'cpu', warm_start = False, coef_ = None, intercept_ = None, kernels_ = None, seed = 100101, verbose = True, patience = 10, adjust_lr = 0.5, outname = None, write_model_params = False, norm = False):
        super(torch_Regression, self).__init__()
        torch.manual_seed(seed) # set seed for the method
        
        self.device = device # cpu or cuda:n
        self.fit_intercept = fit_intercept # include bias or not
        self.loss_function = loss_function # loss that will be optimized, multi-class output can be weighted by classes and samples
        self.penalty = str(penalty) # paramter regularization
        self.logistic = logistic # output type (class or continuous)
        
        self.kernel_length = kernel_length # if length is given, the regression learns motifs in form of num_kernels
        self.n_kernels = n_kernels # Number of kernels
        self.pooling = pooling # if kernel is assinged a pooling needs to be determined (default = 'Max')
        self.pooling_length = pooling_length # if kernels are given we also need a pooling (default is None, which means across the entire sequence)
        self.kernel_function = kernel_function # Non-linear function applied to kernel afterwards
        
        self.norm = norm # If input data should be normalized, for CNN, normalization gets updated after each step
        
        self.alpha = alpha # penalty hyperparamter for linear layer at the end, can also be an array for each out_class
        self.learn_alpha = learn_alpha # Can introduce optimizer that optimizes alphas only on the validation set
        self.alpha_kernels = alpha_kernels # add an additional alpha to introduce sparsity in kernels, this should also be learnable
        self.is_zero = is_zero # Cutoff for parameters below which they are set to 0
        
        self.epochs = epochs # epochs to train on
        self.lr = lr # learning rate
        self.optimizer = optimizer # optimizer
        self.optim_params = optim_params # parameters for the optimizer
        self.batchsize = batchsize # batchsize used in each step
        
        self.patience = patience # number of epochs before updates are interupted and the best model is realoaded
        self.verbose = verbose 
        self.adjust_lr = adjust_lr # adjust learning rate if training loss goes up
        
        self.outname = outname
        if self.outname is None:
            self.outname = 'LR_'
        self.outname += self.loss_function + str(fit_intercept)[0]
        if logistic != "Linear":
            self.outname += '_'+logistic
        if self.penalty != 'None':
            self.outname += '_p'+penalty+str(alpha)
            if kernel_length is not None:
                self.outname += 'ka'+str(alpha_kernels)
        
        if kernel_length is not None:
            print(self.kernel_function)
            self.outname += 'cnn'+str(kernel_length)+'n'+str(n_kernels)+'-'+str(pooling)+ str(pooling_length)+self.kernel_function
        if self.learn_alpha: 
            self.outname += 'la'+str(learn_alpha)[0]
            
        if write_model_params:
            obj = open(self.outname+'_model_params.dat', 'w')
            for key in self.__dict__:
                obj.write(key+' : '+str(self.__dict__[key])+'\n')
            obj.close()
        
        self.loss_func = loss_dict[loss_function]
        
        self.warm_start = warm_start # if warm_start the parameters can be set from outside
        self.coef_ = coef_ # of the last layer output
        self.intercept_ = intercept_ # Intercept of the last layer
        self.kernels_ = kernels_ # parameters of the initial kernels
        self.best_loss = None
        
        if optim_params is None:
            if self.optimizer == 'SGD':
                self.optim_params = 0.
            elif self.optimizer in ['Adam','NAdam','AdamW', 'Amsgrad']:
                self.optim_params = (0.9,0.999)
            
    def get_params(self):
        return self.__dict__

    def set_params(self, **kwargs):
        for kwar in kwargs:
            if str(kwar) in self.__dict__.keys():
                setattr(self, kwar, kwargs[kwar])
        
    def forward(self, X, index = None):
        if self.kernel_length is not None:
            X = self.feature_extractor(X)
        if self.norm:
            recompute = False
            if self.kernel_length is not None and index is not None:
                self.Xrep[index,:] = X.detach().cpu()[:,:]
                recompute = True
            with torch.no_grad():
                X = self.norm_function(X, recompute = recompute)
        return self.predictor(X)
    
    def save_model(self, PATH):
        torch.save(self.state_dict(), PATH)
    
    def load_model(self, PATH):
        # avoids memory error on gpu: https://discuss.pytorch.org/t/cuda-error-out-of-memory-when-load-models/38011/3
        state_dict = torch.load(PATH, map_location = 'cpu')
        self.load_state_dict(state_dict)
        self.to(self.device)
    
    def stop_criterion(self, i, lossi):
        
        # if loss gets nan it's not producing meaningful losses
        if np.isnan(lossi):
            return True, 'Loss nan'
        
        if self.best_loss is None or i == 0:
            self.steps_since = 0
            self.best_loss = lossi
        else:
            if lossi < self.best_loss:
                self.best_loss = lossi
                self.steps_since = 0
            else:
                self.steps_since +=1
            if self.steps_since == self.patience:
                return True, 'Loss has not decreased since '+str(self.patience)+' steps'

        return False, None
    
    def norm_function(self, X, recompute = False):
        if recompute:
            self.mean = torch.mean(self.Xrep, dim = 0)
            self.var = torch.std(self.Xrep, dim = 0)
        return (X-self.mean)/self.var
    
    # Losses for weight regularization
    def l1_loss(self, w, summarize = 'mean'):
        if summarize == 'mean':
            return torch.abs(w).mean()
        return torch.abs(w)

    def l2_loss(self, w, summarize = 'mean'):
        if summarize == 'mean':
            return torch.square(w).mean()
        return torch.square(w)
    
    def set_zero(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.copy_(param*torch.absolute(param)>self.is_zero)
            # Should set parameters to true ZERO if they are smaller than a defined value.
            # doesnt work somehow
            ##state_dict = self.state_dict()
            #for param_tensor in state_dict:
                #state_dict[param_tensor] = state_dict[param_tensor]*torch.absolute(state_dict[param_tensor])>self.is_zero
            #self.load_state_dict(state_dict)
        #for param in self.parameters():
            #param = param * (torch.absolute(param)>self.is_zero)
        
        
    # execute one epoch with training or validation set. Training set takes gradient but validation set computes loss without gradient
    def excute_epoch(self, dataloader, normsize, take_grad, optimizer = None, sample_weight = None):
        trainloss = 0.
        for sample_x, sample_y, index in dataloader:
            sample_x, sample_y = sample_x.to(self.device), sample_y.to(self.device)
            if take_grad:
                optimizer.zero_grad()
                Ypred = self.forward(sample_x, index = index)
                loss = self.loss_func(Ypred, sample_y)
                if sample_weight is not None:
                    loss = torch.sum(loss*sample_weight[index][:,None])
                else:
                    loss = torch.sum(loss)
                trainloss += float(loss.item())
                if self.penalty == 'l1' or self.penalty == 'elastic':
                    loss += self.alpha*float(sample_x.size(dim = 0))/normsize * torch.norm(self.predictor.Linear.weight, p = 1)
                    if self.alpha_kernels is not None and self.kernel_length is not None:
                        loss += self.alpha_kernels*float(sample_x.size(dim = 0))/normsize * torch.norm(self.feature_extractor.Kernels.weight, p = 1)
                    
                if self.penalty == 'l2' or self.penalty == 'elastic':
                    loss += self.alpha*float(sample_x.size(dim = 0))/normsize * torch.norm(self.predictor.Linear.weight, p = 2)
                    if self.alpha_kernels is not None and self.kernel_length is not None:
                        loss += self.alpha_kernels*float(sample_x.size(dim = 0))/normsize * torch.norm(self.feature_extractor.Kernels.weight, p = 2)
                                         
                loss.backward()
                optimizer.step()
                if self.is_zero > 0:
                    with torch.no_grad():
                        self.set_zero()
            else:
                Ypred = self.forward(sample_x)
                loss = self.loss_func(Ypred, sample_y)
                if sample_weight is not None:
                    loss = torch.sum(loss*sample_weight[index][:,None])
                else:
                    loss = torch.sum(loss)
                trainloss += float(loss.item())
        trainloss /= normsize
        return trainloss
    
    def fit(self, X, Y, XYval = None, sample_weight = None, class_weight = None, norm = True, **kwargs):
        
        if len(np.shape(Y)) == 1:
            Y = Y.reshape(-1,1)
        self.Ydim = np.shape(Y)
        self.Xdim = np.shape(X)
        
        # Initialize model 
        modellist = OrderedDict()
        if self.kernel_length is not None and len(self.Xdim) >= 3:
            modellist['Kernels'] = nn.Conv1d(self.Xdim[1], self.n_kernels, self.kernel_length, bias = False)
            if self.kernels_ is not None:
                modellist['Kernels'].weight = nn.Parameter(torch.Tensor(self.kernels_))
            modellist['Kernel_func'] = func_dict[self.kernel_function]
            currlen = int((self.Xdim[-1] - (self.kernel_length -1))/1.)
            if self.pooling_length is None:
                self.pooling_length = currlen
            if self.pooling == 'Max':
                modellist['Pooling'] = nn.MaxPool1d(self.pooling_length)
            elif self.pooling == 'Mean':
                modellist['Pooling'] = nn.AvgPool1d(self.pooling_length)
            elif self.pooling == 'Conv':
                modellist['Pooling'] = nn.Conv1d(self.n_kernels, self.n_kernels, self.pooling_length, bias = False, stride = self.pooling_length)
            if self.pooling in ['Max', 'Mean', 'Conv']:
                currlen = int(np.floor(1. + (currlen - (self.pooling_length-1)-1)/self.pooling_length))
            modellist['Flatten'] = nn.Flatten()
            self.feature_extractor = nn.Sequential(modellist)
            inshape = self.n_kernels *currlen
        else:
            inshape = self.Xdim[1]
            self.kernel_length = None
        
            
        
        # Initialize model 
        modellist = OrderedDict()
        modellist['Linear'] = nn.Linear(inshape, np.shape(Y)[1], bias = self.fit_intercept)
        # Set parameters to given values if given to the model
        if self.coef_ is not None:
            modellist['Linear'].weight = nn.Parameter(torch.Tensor(self.coef_))
        if self.intercept_ is not None:
            modellist['Linear'].bias = nn.Parameter(torch.Tensor(self.intercept_))
        
        if self.logistic == 'Multiclass':
            modellist['Logistic'] = nn.Softmax()
        elif self.logistic == 'Multilabel':
            modellist['Logistic'] == nn.Sigmoid()
        
        self.predictor = nn.Sequential(modellist)
        
        if self.verbose:
            for param_tensor in self.state_dict():
                print(param_tensor, "\t", self.state_dict()[param_tensor].size())
        
        # transform input data
        X = torch.Tensor(X)
        Y = torch.Tensor(Y)
        my_dataset = MyDataset(X, Y)
        if self.batchsize is None:
            self.batchsize = len(X)
        dataloader = DataLoader(my_dataset, batch_size = self.batchsize, shuffle = True)        
        
        if self.norm: 
            # Function whose mean and variance is updated whenever necessary
            if self.kernel_length is not None:
                # If recompute is True, Xrep will be updated after each minibatch and the mean and std will be computed again
                self.Xrep = torch.empty(X.size(dim=0),inshape)
                with torch.no_grad():
                    for sample_x, sample_y, index in dataloader:
                        self.Xrep[index] = self.feature_extractor(sample_x).detach().cpu()
            else:
                self.Xrep = X
            self.mean = torch.mean(self.Xrep, dim = 0)
            self.var = torch.std(self.Xrep, dim = 0)
            
        
        # establish if validation set was provided for early stopping
        if XYval is not None:
            Xval, Yval = torch.Tensor(XYval[0]), torch.Tensor(XYval[1])
            my_val_dataset = MyDataset(Xval, Yval) # create your validation datset
            valsize = float(Xval.size(dim = 0))
            val_batchsize = int(min(self.batchsize,valsize)) # largest batchsize for validation set is 250 to save memory on gpu
            val_dataloader = DataLoader(my_val_dataset, batch_size = val_batchsize, shuffle = True)
        else:
            val_dataloader = dataloader
        
        val_weights = None    
        if sample_weight is not None:
            # If weighted outputs then the loss needs to normalized by the sum of weights
            trainsize = np.sum(sample_weight)
            if XYval is not None:
                val_weights = torch.ones(np.shape(Xval)[0])
        else:
            trainsize = float(X.size(dim=0))
        
        # send predictor parameters to device
        self.to(self.device)
            
        if self.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.optim_params)
        elif self.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=self.optim_params)
        
        self.coef_0 = self.predictor.Linear.weight.clone().detach().cpu().numpy()
        if self.fit_intercept:
            self.intercept_0 = self.predictor.Linear.bias.clone().detach().cpu().numpy()
        if self.kernel_length is not None:
            self.kernels_0 = self.feature_extractor.Kernels.weight.clone().detach().cpu().numpy()
        
        self.save_model(self.outname+'_params0.pth')
        self.save_model(self.outname+'_parameter.pth')
        # Compute losses with intial random parameters for training and validation set
        beginning_loss = self.excute_epoch(dataloader, trainsize, False, optimizer = None, sample_weight = sample_weight)
        if XYval is not None:
            lossval = self.excute_epoch(val_dataloader, valsize, False, optimizer = None, sample_weight = val_weights)
        else:
            lossval = beginning_loss
        self.saveloss = [lossval, beginning_loss, 0]
        if self.verbose:
            print(0, lossval, beginning_loss)
        early_stop, stopexp = self.stop_criterion(0, lossval)
        
        # Start epochs and updates
        restarted = 0
        e = 0
        while True:
            # update parameters 
            trainloss = self.excute_epoch(dataloader, trainsize, True, optimizer = optimizer, sample_weight = sample_weight)
            
            # Assess parameter update on validation set
            if XYval is not None:
                self.eval() # Sets model to evaluation mode which is important for batch normalization over all training mean and for dropout to be zero
                lossval = self.excute_epoch(val_dataloader, valsize, False, optimizer = None, sample_weight = val_weights)
                self.train()                
            else:
                lossval = trainloss
            
            e += 1
            if e >= 1 and ((trainloss > beginning_loss) or np.isnan(trainloss)):
                self.lr *= self.adjust_lr
                if self.verbose:
                    print(e, trainloss ,'>', beginning_loss)
                    print('LR', self.lr)
                e = 0
                early_stop, stopexp = self.stop_criterion(0, self.best_loss)
                #self.predictor.Linear.reset_parameters()
                self.load_model(self.outname+'_params0.pth')
                for g in optimizer.param_groups:
                    g['lr'] = self.lr
            else:        
                if self.verbose and ((e < 10) or (e %10 ==0)):
                    print(e, lossval, trainloss)
                early_stop, stopexp = self.stop_criterion(e, lossval)
                if e == self.epochs or early_stop:
                    # stop after reaching maximum epoch number
                    if early_stop:
                        # Early stopping if validation loss is not improved for patience epochs
                        self.load_model(self.outname+'_parameter.pth')
                        print("Loaded model from", self.outname+'_parameter.pth', self.saveloss[-1], 'at steps with loss', self.saveloss[0])
                        print(e, lossval, stopexp)
                    break
                elif self.saveloss[0] > lossval:
                    self.saveloss = [lossval, trainloss, e]
                    self.save_model(self.outname+'_parameter.pth')
        
        os.remove(self.outname+'_parameter.pth')          
        os.remove(self.outname+'_params0.pth')   
        
        if self.norm: 
            if self.kernel_length is not None:
                self.Xrep = torch.empty(X.size(dim=0),inshape)
                with torch.no_grad():
                    for sample_x, sample_y, index in dataloader:
                        self.Xrep[index] = self.feature_extractor(sample_x).detach().cpu()
            else:
                self.Xrep = X
            self.mean = torch.mean(self.Xrep, dim = 0)
            self.var = torch.std(self.Xrep, dim = 0)
        
        self.coef_ = self.predictor.state_dict()['Linear.weight'].clone().detach().cpu().numpy()
        if self.fit_intercept:
            self.intercept_ = self.predictor.state_dict()['Linear.bias'].clone().detach().cpu().numpy()
        if self.kernel_length is not None:
            self.kernels_ = self.feature_extractor.state_dict()['Kernels.weight'].clone().detach().cpu().numpy()
        
        
    def predict(self, X, mask = None, device = None):
        self.eval()
        if device is None:
            device = self.device
        if device != self.device:
            self.to(device)
        with torch.no_grad():
            X = torch.Tensor(X)
            if device == self.device:
                predout = []
                for i in range(0, int(X.size(dim=0)/self.batchsize)+int(X.size(dim=0)%self.batchsize != 0)):
                    xin = X[i*self.batchsize:(i+1)*self.batchsize]
                    xin = xin.to(device)
                    predout.append(self.forward(xin).detach().cpu().numpy())
                predout = np.concatenate(predout, axis = 0)
            else:
                X = X.to(device)
                predout = self.forward(X)
                predout.detach().cpu().numpy()
        return predout



from data_processing import readin

from k_mer_model import mse, create_sets,manipulate_input, correlation, print_averages, create_outname, save_performance, check,numbertype
    

if __name__ == '__main__':
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    outname = create_outname(inputfile, outputfile) 
    
    if '--outdir' in sys.argv:
        outname = sys.argv[sys.argv.index('--outdir')+1] + os.path.split(outname)[1]

    delimiter = ','
    if '--delimiter' in sys.argv:
        delimiter = str(sys.argv[sys.argv.index('--delimiter')+1])
        print(delimiter)
    X, Y, names, features, experiments = readin(inputfile, outputfile, delimiter = delimiter,return_header = True)
    
    X, features = manipulate_input(X, features, sys.argv)
    
    # use a random number of samples from the entire data set
    if '--sub_sample' in sys.argv:
        subsamp = float(sys.argv[sys.argv.index('--sub_sample')+1])
        if subsamp < 1:
            subsamp = int(len(X)*subsamp)
        subsamp = int(subsamp)
        outname +='ss'+str(subsamp)
        sub = np.random.permutation(len(X))[:subsamp]
        names, X, Y = names[sub], X[sub], Y[sub]
    
    if '--crossvalidation' in sys.argv:
        folds = int(sys.argv[sys.argv.index('--crossvalidation')+1])
        fold = int(sys.argv[sys.argv.index('--crossvalidation')+2])
        if '--significant_genes' in sys.argv:
            siggenes = np.genfromtxt(sys.argv[sys.argv.index('--significant_genes')+1], dtype = str)
            Yclass = np.isin(names, siggenes).astype(int)
        elif '--gene_classes' in sys.argv:
            Yclass = []
            for n, name in enumerate(names):
                Yclass.append(name.split(':')[0])
            Yclass = np.array(Yclass)
        else:    
            cutoff = float(sys.argv[sys.argv.index('--crossvalidation')+3])
            Yclass = (np.sum(np.absolute(Y)>=cutoff, axis = 1) > 0).astype(int)
    else:
        folds, fold, Yclass = 10, 0, None
        
    outname += '-cv'+str(folds)+'-'+str(fold)
    trainset, testset, valset = create_sets(len(X), folds, fold, Yclass = Yclass)
    
    
    if '--norm2output' in sys.argv:
        outname += '-n2out'
        norm =np.sqrt(np.sum(Y*Y, axis = 1))[:, None] 
        Y = Y/norm
        
    elif '--norm2outputclass' in sys.argv:
        outname += '-n2outclass'
        norm =np.sqrt(np.sum(Y*Y, axis = 0))
        Y = Y/norm

    weights = None
    if '--sample_weights' in sys.argv:
        weightfile = np.genfromtxt(sys.argv[sys.argv.index('--sample_weights')+1], dtype = str)
        outname += '_sweigh'
        weight_names, weights = weightfile[:,0], weightfile[:,1].astype(float)
        sortweight = np.argsort(weight_names)[np.isin(np.sort(weight_names), names)]
        weight_names, weights = weight_names[sortweight], weights[sortweight]
        if not np.array_equal(weight_names, names):
            print("Weights cannot be used")
            sys.exit()
        weights = weights[trainset]
    
    if '--regression' in sys.argv:
        params = {}
        params['device'] = get_device()
        print('Device', params['device'])
        if len(sys.argv) > sys.argv.index('--regression') + 1:
            parameters = sys.argv[sys.argv.index('--regression')+1]
            if '+' in parameters:
                parameters = parameters.split('+')
            else:
                parameters = [parameters]
            for p in parameters:
                if ':' in p and '=' in p:
                    p = p.split('=',1)
                elif ':' in p:
                    p = p.split(':',1)
                elif '=' in p:
                    p = p.split('=',1)
                params[p[0]] = check(p[1])
        params['outname'] = outname
        
        model = torch_Regression(**params)
    
    model.fit(X[trainset], Y[trainset], XYval = [X[valset], Y[valset]], weights = weights)
    Y_pred = model.predict(X[testset])
    
    if '--random_model' in sys.argv:
        could_add_modelthatlearns_on_random_data = 0
    
    if '--save_predictions' in sys.argv:
        np.savetxt(outname+'_pred.txt', np.append(names[testset][:, None], Y_pred, axis = 1), header = ' '.join(experiments), fmt = '%s')
        print('SAVED', outname+'_pred.txt')    

    if '--split_outclasses' in sys.argv:
        testclasses = np.genfromtxt(sys.argv[sys.argv.index('--split_outclasses')+1], dtype = str)
        tsort = []
        for exp in experiments:
            tsort.append(list(testclasses[:,0]).index(exp))
        testclasses = testclasses[tsort][:,1]
    else:
        testclasses = np.zeros(len(Y[0]), dtype = np.int8).astype(str)
    
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
    
    if '--feature_statistics' in sys.argv:
        # saves sign(coef)*log10(pvalues)
        logpvalues = model.statistical_weight(X[trainset], Y[trainset])
        np.savetxt(outname+'_feature_stats.dat', np.append(np.array(features).reshape(-1,1),np.around(logpvalues.T,2), axis = 1).astype(str), fmt = '%s', header = ' '.join(experiments))
    
    if '--feature_weights' in sys.argv:
        np.savetxt(outname+'_features.dat', np.append(np.array(features).reshape(-1,1),np.around(model.coef_.T[:-1],5), axis = 1).astype(str), fmt = '%s', header = ' '.join(experiments))
        
        
    








