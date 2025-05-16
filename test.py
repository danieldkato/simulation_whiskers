# -*- coding: utf-8 -*-
"""
Created on Thu May 15 01:08:33 2025

@author: danie
"""
import numpy as np
import time
from sklearn.linear_model import LogisticRegression


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