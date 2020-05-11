# -*- coding: iso-8859-1 -*-
import numpy as np
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, WhiteKernel
from taper_output import taper_output

class machine_interface:
    def __init__(self, dev_ids, start_point = None):
        self.pvs = np.array(dev_ids)
        self.name = 'undulator_interface' #name your machine interface. doesn't matter what you call it as long as it isn't 'MultinormalInterface'.
        if type(start_point) == type(None):
            current_x = np.zeros(len(self.pvs)) #replace with expression that reads current ctrl pv values (x) from machine
            self.setX(current_x)
        else: 
            self.setX(start_point)

    def setX(self, x_new):
        self.x = np.array(x_new, ndmin=2)

    def getState(self): 
#         self.x is und k
#         print(self.x[0][0])
        z,power_z = taper_output(self.x[0][0])
        objective_state = (power_z[-1])*10**(-10)
        return np.array(self.x, ndmin = 2), np.array([[objective_state]])
