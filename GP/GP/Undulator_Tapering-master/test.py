
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules.bayes_optimization import BayesOpt, negUCB, negExpImprove
# from modules.bayes_optimization_lik_opt import BayesOpt, negUCB, negExpImprove

from modules.OnlineGP import OGP
import numpy as np
import importlib
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

mi_module = importlib.import_module('machine_interfaces.undulator_interface')


saveResultsQ = True

#how long to wait between acquisitions
acquisition_delay = 0 

#create the machine interface
dev_ids = ['und_k']
start_point = 1    
mi = mi_module.machine_interface(dev_ids = dev_ids, start_point = start_point)

def convert_long(l,N=200):
    #result=[3.5]*66
    result=[]
    step = N//len(l)
    for i in range(len(l)-1):
        result = result + [l[i]]*step
    result = result + [l[-1]]*(N-step*(len(l)-1))
    return result

def convert_short(l,n=32):
    step=len(l)//n
    result=[]
    for i in range(len(l)):
        if(i%step==(step-1)):
            result.append(l[i])
    return result[11:n]

def taper_get(param_K):
    '''
    Input:
    unduK is an array to represent taper profile (recommend to be a shape of (200,)
    
    Output:
    z is the position array along the undulator
    power_z is the output power along undulator
    '''
    
    starting_step_short = 11
    z_steps_short = 32
    const_K = 3.5
    n_z = 2
    unduK_1=const_K*np.ones(int(starting_step_short))
    
    '''modified by huang'''
    unduK_2=const_K+param_K*np.arange(1,1+z_steps_short-starting_step_short)**n_z
    
    #unduK_2=const_K+param_K*np.arange(z_steps_short-starting_step_short)**n_z
    undu_K=np.concatenate((unduK_1,unduK_2))
    unduK = np.array(convert_long(undu_K))
    return unduK

def taper_get2(param_K, best_K=None):
    '''
    Input:
    unduK is an array to represent taper profile (recommend to be a shape of (200,)
    
    Output:
    z is the position array along the undulator
    power_z is the output power along undulator
    '''
    
    starting_step_short = 11
    z_steps_short = 32
    const_K = 3.5
    n_z = 2
    unduK_1=const_K*np.ones(int(starting_step_short))
    
    
    '''modified by huang'''
    param_K = param_K.flatten()
    
    if best_K is None:
    
        '''modified by huang'''
        unduK_2=const_K+param_K*np.arange(1,1+z_steps_short-starting_step_short)**n_z
        
        #unduK_2=const_K+param_K*np.arange(z_steps_short-starting_step_short)**n_z
        undu_K=np.concatenate((unduK_1,unduK_2))
        unduK = np.array(convert_long(undu_K))
    
    else:
        '''modified by huang: fine tune parameter in the best quad case'''
        unduK_2=const_K+best_K*np.arange(1,1+z_steps_short-starting_step_short)**n_z
        assert unduK_2.shape == param_K.shape
        unduK_2 += param_K
        undu_K=np.concatenate((unduK_1,unduK_2))
        unduK = np.array(convert_long(undu_K))
    return unduK
                         
tap = taper_get(np.array([-0.00038487]))
convert_short(3.5-tap)[0]

tap_short = convert_short(tap)
length_scale_zoom = convert_short(3.5-tap)

ndim = len(dev_ids)
ndim = 21
#1. gp_lengthscales - learning rate
#gp_lengthscales = np.array([1])
#gp_lengthscales = np.array([0.0001])

gp_lengthscales = np.array([0.1])*length_scale_zoom
t = [0.00001] + [0.0001]*(len(length_scale_zoom)-1)
gp_lengthscales = np.array(t)

#gp_precisionmat = 1/np.diag(gp_lengthscales**(2))

#gp_precisionmat = np.diag(np.log(1./gp_lengthscales**(2)))
gp_precisionmat = np.log(1./gp_lengthscales**(2))

#2. gp_amp
#gp_amp = 0.1
gp_amp = 1
#3. gp_noise
#gp_noise = 0.0001 
gp_noise =  10**(-4)

hyps = [gp_precisionmat, np.log(gp_amp), np.log(gp_noise**2)] #format the hyperparams for the OGP
gp = OGP(ndim, hyps)

bounds=((-0.00875,0),)
bounds = tuple([(-x,y) for x,y in zip(tap_short, length_scale_zoom)])


mi.set_best_K(-0.00038487)
mi.setX(np.array([0]*21))
s=mi.getState()
opt = BayesOpt(gp, mi, acq_func="UCB", dev_ids = dev_ids, bounds=bounds)

opt.ucb_params = [2,None]
print('ucb_params',opt.ucb_params)

Obj_state_s=[]

import time
end = time.time()
Niter = 10
for i in range(Niter):
#    clear_output(wait=True) 
    print ('iteration =', i)
    print ('current position:', mi.x, 'current objective value:', mi.getState()[1])
    
    x_best, y_best = opt.best_seen()
    print('best position:', x_best, 'best objective value:', y_best)
    Obj_state_s.append(mi.getState()[1][0])
    print('time=', time.time()-end)
    end = time.time()
    opt.OptIter()
