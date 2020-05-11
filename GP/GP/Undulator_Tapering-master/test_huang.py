# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:30:37 2020

@author: Admin
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules.bayes_optimization import BayesOpt, negUCB, negExpImprove
# from modules.bayes_optimization_lik_opt import BayesOpt, negUCB, negExpImprove

from modules.OnlineGP import OGP
import numpy as np
import importlib


mi_module = importlib.import_module('machine_interfaces.undulator_interface')
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

saveResultsQ = True

#how long to wait between acquisitions
acquisition_delay = 0 

#create the machine interface
dev_ids = ['und_k']
start_point = 1    
mi = mi_module.machine_interface(dev_ids = dev_ids, start_point = start_point)

ndim = len(dev_ids)
#1. gp_lengthscales - learning rate
#gp_lengthscales = np.array([1])
#gp_lengthscales = np.array([0.0001])

gp_lengthscales = np.array([0.0001])

#gp_precisionmat = 1/np.diag(gp_lengthscales**(2))

gp_precisionmat = np.diag(np.log(1./gp_lengthscales**(2)))

#2. gp_amp
#gp_amp = 0.1
gp_amp = 10
#3. gp_noise
#gp_noise = 0.0001 
gp_noise =  10**(-10)

hyps = [gp_precisionmat, np.log(gp_amp), np.log(gp_noise**2)] #format the hyperparams for the OGP
gp = OGP(ndim, hyps)

#create the bayesian optimizer that will use the gp as the model to optimize the machine 
#opt = BayesOpt(gp, mi, acq_func="UCB", start_dev_vals = mi.x, dev_ids = dev_ids)
#mi.setX(-0.004)
#opt = BayesOpt(gp, mi, acq_func="UCB", start_dev_vals = -0.004, dev_ids = dev_ids, bounds=((-np.inf, 0)))


mi.setX(-0.001)
s=mi.getState()
opt = BayesOpt(gp, mi, acq_func="UCB", dev_ids = dev_ids, bounds=((-0.00875,0),))

opt.ucb_params = [2,None]
print('ucb_params',opt.ucb_params)

opt.OptIter()