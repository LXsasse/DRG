#test_gradients.py
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from init import get_device, MyDataset, DataLoader
import torch.optim as optim
from functions import correlation
from torch.autograd import Variable
import sys, os
from cnn_model import cnn
from train import load_model
from data_processing import readinfasta, quick_onehot, check
import time
import matplotlib as plt
from modules import loss_dict

# Enable scoring with set of models from different folds or different intializations
# 

# Use different attribution methods for GA:
    # Gradient
    # Hessian
    # Forward pass


def load_cnn_model(parameters, device = 'cpu', **kwargs):
    parameterfile = parameters.replace('model_params.dat', 'parameter.pth')
    obj = open(parameters,'r').readlines()
    parameters = []
    for l, line in enumerate(obj):
        if line[0] != '_' and line[:7] != 'outname':
            parameters.append(line.strip().replace(' ', ''))
    params = {}
    for p in parameters:
        p = p.split(':',1)
        params[p[0]] = check(p[1])
    for kw in kwargs:
        params[kw] = kwargs[kw]
    params['device'] = device
    model = cnn(**params)
    load_model(model, parameterfile, device)
    return model


# Generate random sequences
def random_sequences(n, l):
    randseq = np.zeros((n,4,l), dtype = int)
    for i in range(n):
        randseq[i,np.random.randint(4, size = l), np.arange(l,dtype = int)] = 1
    return randseq


def optimize_sequences(start, model, targ, loss_function = 'MSE', scoring = 'forward', updates = 'genetic', patience = 100, max_iter = 1000, device = 'cpu', **kwargs):
    
   
    start, targ = torch.Tensor(start).unsqueeze(0), torch.Tensor(targ).unsqueeze(0)
    bestseq = np.copy(start)
    loss = loss_dict[loss_function]
    loss.reduction = 'mean'
    
    positional_scoring = positional_scoring_function(scoring)
    update_sequence = update_algs(updates, init_scaling = 0.01*max_iter, **kwargs)
    
    opt_score, change_pos, change_base, orig_base = [], [], [], []
    
    cseq = torch.clone(start)
    cseq.requires_grad = True
    
    ppred = model.forward(cseq)
    start_score = ppred.detach().numpy()[0]
    #opt_score.append(start_score)
    
    nloss = loss(ppred, targ)
    oloss = nloss.item()
    t = 0
    not_imp = 0
    while True:
        t += 1
        
        cseq.requires_grad = True
        pscore = positional_scoring(nloss, cseq, model = model, loss = loss, target = targ)
        if np.sum(pscore > 0) == 0:
            return bestseq, opt_score[:t-not_imp], start_score, change_pos[:t-not_imp], change_base[:t-not_imp], orig_base[:t-not_imp]
        
        cseq, pos, obase, nbase = update_sequence(cseq, pscore)
        #print(pos, obase, nbase)
        #print(''.join(np.array(list('ACGT'))[np.where(cseq[0].detach().numpy().T>0)[1]]))
        pred = model.forward(cseq)
        nloss = loss(pred, targ)
        
        
        opt_score.append(pred.cpu().detach().numpy()[0])
        #print(opt_score[-1])
        change_pos.append(pos)
        change_base.append(nbase)
        orig_base.append(obase)
        
        if nloss.item() < oloss:
            not_imp = 0
            bestseq = cseq.detach().cpu().numpy()
            oloss = np.copy(nloss.item())
        else:
            not_imp += 1
            
        if oloss == 0 or not_imp == patience or t == max_iter:
            return bestseq, opt_score[:t-not_imp], start_score, change_pos[:t-not_imp], change_base[:t-not_imp], orig_base[:t-not_imp]
        

class positional_scoring_function(nn.Module):
    def __init__(self, scoretype):
        super(positional_scoring_function, self).__init__()
        self.scoretype = scoretype
        
    def forward(self, diffloss, nseq, model = None, loss = None, target = None):
        if self.scoretype == 'forward':
            # remove for loop with extension of axis = 0 in nseq, then set all positions to 0 individually and then insert the one all at once
            if model is None or loss is None or target is None:
                print("Provide model, loss, and target")
                sys.exit()
            loss.reduction = 'none'
            setto = np.array(np.where(nseq[0] == 0))
            icur = np.array(np.where(nseq[0] == 1))
            with torch.no_grad():
                target = target.expand(len(setto[0])+1,-1)
                tseq = torch.clone(nseq.detach())
                tseq = tseq.repeat(len(setto[0])+1,1,1)
                iscur = []
                for i in range(len(setto[0])):
                    iscur.append([icur[0][icur[1] == setto[1][i]][0] , setto[1][i]])
                    tseq[i, :, setto[1][i]] = 0
                    tseq[i, setto[0][i], setto[1][i]] = 1
                iscur = np.array(iscur).T
                tpred = model.forward(tseq)
                ismloss = loss(tpred,target)
                ismloss = torch.mean(ismloss, axis = 1)
                impact = np.zeros(nseq.size())
                diffloss = ismloss[-1]
                impact[0, setto[0], setto[1]] = diffloss - ismloss[:-1]
                loss.reduction = 'mean'
            '''
            lenseq = nseq.size(dim=-1)
            with torch.no_grad():
                impact = np.zeros(nseq.size())
                for l in range(lenseq):
                    iscur = np.where(nseq[...,l] == 1)[1][0]
                    setto1 = np.where(nseq[...,l] == 0)[1]
                    for i in setto1:
                        tseq = torch.clone(nseq)
                        tseq[0,i,l] = 1
                        tseq[0,iscur, setto1] = 0
                        ismloss = loss(model.forward(tseq), target).item()
                        impact[0,i,l] = diffloss.item() - ismloss
                        #print(i,l, diffloss, ismloss, impact[0,i,l])
            '''
            return impact
        elif self.scoretype == 'random_forward':
            if model is None or loss is None or target is None:
                print("Provide model, loss, and target")
                sys.exit()
            lenseq = nseq.size(dim=-1)
            with torch.no_grad():
                impact = np.zeros(nseq.size())
                allset = list(zip(np.where(nseq[0] == 0)[0], np.where(nseq[0] == 0)[1]))
                for l in np.random.permutation(len(allset)):
                    tseq = torch.clone(nseq)
                    tseq[0,:,allset[l][1]] = 0
                    tseq[0,allset[l][0],allset[l][1]] = 1
                    ismloss = loss(model.forward(tseq), target).item()
                    imp = diffloss.item() - ismloss
                    if imp > 0:
                        impact[0,allset[l][0],allset[l][1]] = imp
                        return impact
            return impact
       
        elif self.scoretype == 'gradient':
            if nseq.grad is not None:
                nseq.grad.data.zero_()
            gradin = diffloss.backward(retain_graph = True)
            grad = -nseq.grad.numpy()
            grad[np.where(nseq > 0)] = 0
            #print(grad[np.where(nseq>0)])
            return grad
        
        elif self.scoretype == 'gradient_difference':
            if nseq.grad is not None:
                nseq.grad.data.zero_()
            gradin = diffloss.backward(retain_graph = True)
            grad = nseq.grad.numpy()
            grad = -grad + np.transpose(grad,axes = (0,2,1))[np.where(nseq.transpose(2,1) > 0)][None,:]
            return grad
        elif self.scoretype == 'random':
            return np.random.random(nseq.size())


# set scaling to small fixed value and baseline to 0., so that every positive position has same chance for "real" geentic algorithm
class update_algs(nn.Module):
    def __init__(self, uptype, init_scaling = .25, temp_update = 'linear', nts = np.array(list('ACGT')), baseline = 1.):
        super(update_algs, self).__init__()
        
        self.uptype = uptype
        self.init_scaling = init_scaling
        self.t = 1
        self.nts = nts
        self.last_pos = None
        self.baseline = baseline
        self.temp_update = temp_update
        
    def forward(self, cseq, pscore):
        if self.last_pos is not None:
            pscore[0,self.last_pos[0], self.last_pos[1]] = np.amin(pscore)
        
        if self.uptype == 'genetic':
            chosenpos = np.argmax(np.transpose(pscore, axes = (0,2,1)).flatten())
        else:
            pscore[pscore < self.baseline ] = self.baseline
            pscore[cseq > 0] = 0
            if self.temp_update == 'equalchance':
                proba = pscore > 0
                proba = proba.astype(float)/np.sum(proba) 
            else:
                if self.t == 1:
                    self.init_scaling = np.amax(pscore)*self.init_scaling
                self.scaling = self.init_scaling/self.t
                proba = pscore/self.scaling
                proba = np.exp(proba)
                mask = np.isinf(proba)
                if mask.any():
                    proba[mask] = 1
                    proba[~mask] = 0
                proba = proba/np.amax(proba)
                proba = proba/np.sum(proba)
            proba = np.transpose(proba, axes = (0,2,1)).flatten()
            chosenpos = np.random.choice(np.arange(len(proba)), p = proba)
            
        pos = int(chosenpos/4)
        self.last_pos = [np.where(cseq[0,:,pos]>0)[0][0], pos]
        #print(cseq[0, :, pos])
        obase = self.nts[cseq[0, :, pos].detach().numpy() > 0][0]
        nbase = self.nts[chosenpos%4]
        with torch.no_grad():
            cseq[0, :, pos] = 0
            cseq[0, chosenpos%4, pos] = 1
        #print(cseq[0, :, pos])
        self.t += 1
        return cseq, pos, obase, nbase
    


if __name__ == '__main__':
    
    np.random.seed(1)    
    
    # Interesting to see if set of multiple models could optimize this process
    predictor = sys.argv[1]
    
    device = 'cpu'
    if '--gpu' in sys.argv:
        device = sys.argv[sys.argv.index('--gpu')+1]
    model = load_cnn_model(predictor, device = device)
    
    if '--outname' in sys.argv:
        outname = sys.argv[sys.argv.index('--outname')+1]
    else:
        outname = os.path.splitext(predictor)[0]
    
    target_tracks = sys.argv[2]
    outname += target_tracks.replace(',','_')
    
    tset = None
    if target_tracks == "ALL":
        target_tracks = np.arange(model.n_classes, dtype = int)
    elif ',' in target_tracks:
        if '-' in target_tracks:
            target_tracks = target_tracks.split(',')
            tset = []
            for tt in target_tracks:
                if '-' in tt:
                    tset.append(np.arange(int(tt.split('-')[0]), int(tt.split('-')[1])+1))
                else:
                    tset.append([int(tt)])
            target_tracks = np.concatenate(tset)
        else:
            target_tracks = np.array(target_tracks.split(','), dtype = int)
    else:
        if '-' in target_tracks:
            target_tracks = np.arange(int(target_tracks.split('-')[0]), int(tartet_tracks.split('-')[1]))
        else:
            target_tracks = [int(target_tracks)]
    
    model.classifier.Linear.weight = nn.Parameter(model.classifier.Linear.weight[target_tracks])
    model.classifier.Linear.bias = nn.Parameter(model.classifier.Linear.bias[target_tracks])
    
    target_values = sys.argv[3]
    outname += '_'+target_values.replace(',','-')
    
    if ',' in target_values:
        target_values = target_values.split(',')
        if len(target_values) != len(target_tracks) and tset is not None:
            tsetvalues = []
            for t, tval in enumerate(target_values):
                tsetvalues.append(np.ones(len(tset[t]))*float(tval))
            target_values = np.concatenate(tsetvalues)
    else:
        target_values = [target_values]
    target_values = np.array(target_values, dtype = float)
    
    print(target_values)
    
    n_starts = int(sys.argv[4]) # Number of sequences that should be generated from one seed
    
    outname += '_'+str(n_starts)
    
    scoring = sys.argv[5] # forward: mutational scanning, exchange one base and run forward pass
    # gradient: compute gradient for input and update position with highest gradient
    # gradient_difference: compute gradient for input and update position with highest gradient difference to base that is currently there
    # forward_gradient: compute gradient after forward pass and compare to afterwards
    # random 
    
    update_alg = sys.argv[6] # genetic: genetic algorithm
    # annealing: simulated annealing
    # metropolis: informed sampling without annealing
    # random_positive
    kwargs = {}
    if update_alg == 'random_positive':
        kwargs = {'baseline':0,'temp_update':'equalchance', 'max_iter':1000, 'patience':100}
    
    loss_function = sys.argv[7]
    
    outname += '_'+loss_function+scoring+update_alg
    
    if '--load_sequences' in sys.argv:
        seed_names, starting_sequences = readinfasta(sys.argv[sys.argv.index('--load_sequences')+1])
        seed_seqs, nts = quick_onehot(starting_sequences)
        outname += os.path.splitext(os.path.split(sys.argv[sys.argv.index('--load_sequences')+1])[1])[0]
        
        
    elif '--random_sequences' in sys.argv:
        N = int(sys.argv[sys.argv.index('--random_sequences')+1])
        outname += '_randseq'+str(N)
        L = model.l_seqs
        seed_seqs = random_sequences(N,L)
        obj = open(outname + '_start_seq.fasta', 'w')
        seed_names = np.array(['Seed'+str(s) for s in range(len(seed_seqs))])
        nts = np.array(['A', 'C', 'G', 'T'])
        for s, sseq in enumerate(seed_seqs):
            obj.write('>'+seed_names[s]+'\n'+''.join(nts[np.where(sseq.T > 0)[1]])+'\n')
        obj.close()
    
    
    outfile = open(outname + '_optimization_steps.txt', 'w')
    optimized_sequences = []
    for s, seq in enumerate(seed_seqs):
        for n in range(n_starts):
            t1 = time.time()
            opt_seq, opt_score, start_score, change_pos, change_base, orig_base = optimize_sequences(seq, model, target_values, loss_function = loss_function, scoring = scoring, updates = update_alg, **kwargs)
            t2 = time.time()
            if len(opt_score) == 0:
                opt_score = np.array([start_score])
            print(s, n, round(t2-t1, 1), len(change_pos), np.around(opt_score[-1],2))
            optimized_sequences.append([seed_names[s]+'_'+str(n), opt_seq, np.around(opt_score[-1],2)])
            
            if '--mean_targets' in sys.argv:
                ut_values = np.unique(target_values)
                opt_score = np.array(opt_score)
                nopt_score = np.zeros((len(opt_score), len(ut_values)))
                nstart_score = np.zeros(len(ut_values))
                for u, utv in enumerate(ut_values):
                    nopt_score[:,u] = np.mean(opt_score[:, target_values == utv],axis = 1)
                    nstart_score[u] = np.mean(start_score[target_values == utv])
                opt_score= nopt_score
                start_score = nstart_score
                
            outfile.write(seed_names[s]+','+'-'.join(np.around(start_score,2).astype(str))+';')
            for c in range(len(change_base)):
                if c == len(change_base) -1:
                    outfile.write(str(change_pos[c])+','+orig_base[c]+'/'+change_base[c]+','+'-'.join(np.around(opt_score[c],2).astype(str))+'\n')
                else:
                    outfile.write(str(change_pos[c])+','+orig_base[c]+'/'+change_base[c]+','+'-'.join(np.around(opt_score[c],2).astype(str))+';')
            
    outfile.close()
    outfile = open(outname + '_optimized_seqs.fasta', 'w')
    for s, seq in enumerate(optimized_sequences):
        outfile.write('>'+seq[0]+'('+','.join(seq[2].astype(str))+')\n'+''.join(nts[np.where(seq[1][0].T>0)[1]])+'\n')
    outfile.close()
    
    
    
    
    
    
    
    
