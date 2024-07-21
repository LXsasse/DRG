

from torch.utils.data import TensorDataset, DataLoader, Dataset
# model_utils.py 
'''
Contains functions that are required for model initialization and loading
'''

import os
import numpy as np
import torch
import torch.nn as nn




def load_parameters(model, PATH, translate_dict = None, allow_reduction = False, exclude = [], include = False):
    # if path is a string, then load this path, otherwise if it is the model.state_dict then use it directley 
    if isinstance(PATH, str):
        state_dict = torch.load(PATH, map_location = 'cpu')
    else:
        state_dict = PATH
    # this is the state_dict() of the current model 
    cstate_dict = model.state_dict()
    # iterate over the parameters of the current model
    for n, name0 in enumerate(cstate_dict):
        # name is the key that one is looking for in the loaded state_dict
        # intially this is the same as the current model
        name = name0
        # if translate_dict is not None, then change the name from the name in the current model to the given name.
        if translate_dict is not None:
            if name in translate_dict:
                name = translate_dict[name]
        # Generally only replace parameters that can be found in the loaded model's state_dict
            # Either with the name from the current model or with the translated name.
        # If include is False (default): if the name0 of the current model is not given specifically in exclude, it will be considered for replacement.
        # if include is True: only perform this if name0 is specifically in exclude, exclude is the list of layers in the current model that are considered for replacement.
        if name in state_dict and ((name0 in exclude) == include):
            ntens = None
            if cstate_dict[name0].size() == state_dict[name].size():
                cstate_dict[name0] = state_dict[name]
                print("Loaded", name0 ,'with', name)
            # Reduction can be used if number of kernels in current model differs from number of kernels in loaded models. 
            elif allow_reduction:
                print('REDUCED size')
                if (cstate_dict[name0].size(dim = 0) > state_dict[name].size(dim = 0)) and ((cstate_dict[name0].size(dim = -1) == state_dict[name].size(dim = -1)) or ((len(cstate_dict[name0].size()) ==1) and (len(state_dict[name].size())==1))):
                    ntens = cstate_dict[name0]
                    ntens[:state_dict[name].size(dim = 0)] = state_dict[name]
                elif cstate_dict[name0].size(dim = 0) <= state_dict[name].size(dim = 0) and ((cstate_dict[name0].size(dim = -1) == state_dict[name].size(dim = -1)) or ((len(cstate_dict[name0].size()) ==1) and (len(state_dict[name].size())==1))):
                    ntens = state_dict[name][:cstate_dict[name0].size(dim = 0)]
                cstate_dict[name0] = ntens
            else:
                print('\nATTENTION:', name, 'different size. Current model', cstate_dict[name0].size(),  'Saved model', state_dict[name].size(), '\n')
        else:
            print('\nATTENTION:', name, 'not in state_dict\n')
            
    model.load_state_dict(cstate_dict)
    

# For custom data sets that also return the indices for the batch
class MyDataset(Dataset):
    def __init__(self, data, targets, axis = 0, yaxis = 0):
        self.data = data
        self.targets = targets
        self.axis = axis
        self.yaxis = yaxis
        if yaxis == 1:
            self.ndatapoints = len(targets[0])
        else:
            self.ndatapoints = len(targets)
        
    def __getitem__(self, index):
        
        if self.yaxis == 0:
            y = self.targets[index]
        elif self.yaxis == 1:
            y = [dy[index] for dy in self.targets]
        
        if self.axis == 0:
            x = self.data[index]
        elif self.axis == 1:
            x = [dx[index] for dx in self.data]
        
        return x, y, index
        
    def __len__(self):
        return self.ndatapoints



        
# Either use cpu or use gpu with largest free memory
def get_device():
    tot = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Total').readlines()
    used = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used').readlines()
    tot = np.array([int(x.split()[2]) for x in tot])
    used = np.array([int(x.split()[2]) for x in used])
    memory_available = tot - used
    if len(memory_available) > 0:
        bcuda = np.argmax(memory_available)
    else:
        bcuda = 0
    device = torch.device("cuda:"+str(bcuda) if torch.cuda.is_available() else "cpu")
    return device



        

    
