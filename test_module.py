# -*- coding: utf-8 -*-
"""
Created on Thu May 15 01:08:33 2025

@author: danie
"""
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader



def init_par(_x):
    global x 
    x = _x



def fcn_inner(X):
    r = np.random.randn(X.shape[0])
    y =  np.array([0 if x < 0 else 1 for x in r])
    mdl = LogisticRegression()
    mdl.fit(X, y)
    return -1



def fcn_outer(n_feat, n_obs):
    start_fcn = time.time()
    X = np.random.randn(n_obs, n_feat)
    k = fcn_inner(X)
    stop_fcn = time.time()
    dur = stop_fcn - start_fcn
    return k, dur



def init_train_mdl(X_torch, labels_torch, n_hidden, n_epochs=100, beta_ce=1, beta_sp=1, 
    p_norm=2, lr=0.001, sigma_init=1, sigma_noise=0.1, batch_size=64, gpu=False):
    
    n_inpt = X_torch.shape[1]
    n_classes = len(torch.unique(labels_torch))
    
    # Initialize model:
    mdl = Mdl(n_inpt, n_hidden, n_classes, sigma_init)

    if gpu:
        mdl.to('cuda')

    # Train model:
    train_model(mdl, X_torch, labels_torch, n_epochs=n_epochs, beta_ce=beta_ce,
        beta_sp=beta_sp, p_norm=p_norm, lr=lr, sigma_init=sigma_init, sigma_noise=sigma_noise,
        batch_size=batch_size, gpu=gpu)
    
    return 'foo'
    


def init_train_mdl_proc(X_torch, labels_torch, n_hidden, results_dict, i, n_epochs=100, beta_ce=1, beta_sp=1, 
    p_norm=2, lr=0.001, sigma_init=1, sigma_noise=0.1, batch_size=64, gpu=False):
    
    start = time.time()
    
    n_inpt = X_torch.shape[1]
    n_classes = len(torch.unique(labels_torch))
    
    # Initialize model:
    mdl = Mdl(n_inpt, n_hidden, n_classes, sigma_init)

    if gpu:
        mdl.to('cuda')

    # Train model:
    train_model(mdl, X_torch, labels_torch, n_epochs=n_epochs, beta_ce=beta_ce,
        beta_sp=beta_sp, p_norm=p_norm, lr=lr, sigma_init=sigma_init, sigma_noise=sigma_noise,
        batch_size=batch_size, gpu=gpu)

    stop = time.time()    
    results_dict[i] = stop - start

    
    
class Mdl(nn.Module):
    def __init__(self,n_inp,n_hidden, n_classes, sigma_init):    
        super(Mdl,self).__init__()
        self.n_inp=n_inp
        self.n_hidden=n_hidden
        self.n_classes=n_classes
        self.sigma_init=sigma_init
        self.enc=torch.nn.Linear(n_inp,n_hidden)
        self.dec=torch.nn.Linear(n_hidden,n_classes)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.sigma_init)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=self.sigma_init)

    def forward(self,x,sigma_noise,gpu=False):
        if not gpu:
            x_hidden = F.relu(self.enc(x))+sigma_noise*torch.randn(x.size(0),self.n_hidden)
        else:
            print('gpu=True')
            print('x.is_cuda={}'.format(x.is_cuda))
            preactivation = self.enc(x)
            eps = sigma_noise*torch.randn(x.size(0),self.n_hidden).to('cuda')
            print('preactivation.is_cuda={}'.format(preactivation.is_cuda))
            print('eps.is_cuda={}'.format(eps.is_cuda))
            x_hidden = F.relu(preactivation)+eps
        x = self.dec(x_hidden)
        return x,x_hidden



def train_model(mdl, X_torch, labels_torch, n_epochs=100, beta_ce=1, beta_sp=1, 
    p_norm=2, lr=0.001, sigma_init=1, sigma_noise=0.1, batch_size=64, gpu=False):

    # Get dimensions:
    #n_inp = X.shape[1]
    #n_classes = labels.shape[1]

    # Initialize model, optimizer:
    #mdl = Mdl(n_inp, n_hidden, n_classes, sigma_init)
    loss_ce = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mdl.parameters(), lr=lr)

    # Convert data to torch:
    #X_torch = Variable(torch.from_numpy(X))
    #labels_torch = Variable(torch.from_numpy(labels.astype(np.float32)))

    # Move stuff to GPU if necessary:
    #if gpu:
    #    mdl.to('cuda')
    #    X_torch = X_torch.to('cuda')
    #    labels_torch = labels_torch.to('cuda')

    # Initialize loader:
    dset = torch.utils.data.TensorDataset(X_torch, labels_torch)
    data_loader = DataLoader(dset, batch_size=batch_size, shuffle=True)
    
    # Iterate over training epochs:
    for t in np.arange(n_epochs):

        # Iterate over training batches:
        for batch_idx, (curr_X, curr_labels) in enumerate(data_loader):
    
            #print('Training epoch {} of {}, batch {} of {}...'.format(t+1, n_epochs, batch_idx+1, len(data_loader)))
            
            # Reset optimizer:
            optimizer.zero_grad()
    
            # Forward pass:
            #curr_X.to('cuda')
            #print('curr_X.is_cuda={}'.format(curr_X.is_cuda))
            output = mdl(curr_X, sigma_noise, gpu=gpu)
    
            # Compute loss:
            L_ce = loss_ce(output[0], np.squeeze(curr_labels))
            L_sp = sparsity_loss(output[1], p_norm)
            L = beta_ce*L_ce + beta_sp*L_sp 
    
            # Backprop:
            L.backward()
            optimizer.step()
            
    return 'foo'        



def sparsity_loss(data,p):
    loss=torch.mean(torch.pow(abs(data),p),axis=(0,1))
    return loss