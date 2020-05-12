    
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.io
import random
import sase1d_input_part

def convert_short(l,n=32):
    step=len(l)//n
    result=[]
    for i in range(len(l)):
        if(i%step==(step-1)):
            result.append(l[i])
    return result[11:n]

def convert_long(l,N=200):
    #result=[3.5]*66
    result=[]
    step = N//len(l)
    for i in range(len(l)-1):
        result = result + [l[i]]*step
    result = result + [l[-1]]*(N-step*(len(l)-1))
    return result

def taper_output(param_K):
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
    
    #unduK = np.ones(z_steps)*K          # tapered undulator parameter, K [ ]
    #if unduK.shape[0]!=z_steps:
    #    print('Wrong! Number of steps should always be the same as z_steps')
    
    ################## Change above, get an unduK which is 200 dim array##################
    
    # Basic parameters
    Nruns=1                             # Number of runs
    npart   = 512                       # n-macro-particles per bucket 
    s_steps = 200#200#30   300                 # n-sample points along bunch length
    z_steps = 200#unduK.shape[0]#200#30   300                 # n-sample points along undulator
    energy  = 4313.34*1E6               # electron energy [eV]
    eSpread = 0#1.0e-4                  # relative rms energy spread [ ]
    emitN   = 1.2e-6                    # normalized transverse emittance [m-rad]
    currentMax = 3900                   # peak current [Ampere]
    beta = 26                           # mean beta [meter]
    unduPeriod = 0.03                   # undulator period [meter]
    #unduK = 3.5                        # undulator parameter, K [ ]

    unduL = 70#30                       # length of undulator [meter]
    radWavelength = 1.5e-9              # seed wavelength? [meter], used only in single-freuqency runs

    mc2 = 0.51099906E6#510.99906E-3      # Electron rest mass in eV

    gamma0  = energy/mc2                                    # central energy of the beam in unit of mc2

    radWavelength=unduPeriod*(1+unduK[0]**2/2.0)\
                        /(2*gamma0**2)                          # resonant wavelength

    dEdz = 0                            # rate of relative energy gain or taper [keV/m], optimal~130
    iopt = 'sase'                       # 'sase' or 'seeded'
    P0 = 10000*0.0                      # small seed input power [W]
    constseed = 1                       # whether we want to use constant random seed for reproducibility, 1 Yes, 0 No
    particle_position=genfromtxt('machine_interfaces/SASE_particle_position.csv', delimiter=',') # or None  
    # particle information with positions in meter and eta,\
    # if we want to load random particle positions and energy, then set None
    hist_rule='square-root'             # 'square-root' or 'sturges' or 'rice-rule' or 'self-design', number \
                                        #  of intervals to generate the histogram of eta value in a bucket
        
    

    inp_struct={'Nruns':Nruns,'npart':npart,'s_steps':s_steps,'z_steps':z_steps,'energy':energy,'eSpread':eSpread,\
                'emitN':emitN,'currentMax':currentMax,'beta':beta,'unduPeriod':unduPeriod,'unduK':unduK,'unduL':\
                    unduL,'radWavelength':radWavelength,'dEdz':dEdz,'iopt':iopt,'P0':P0,'constseed':constseed,'particle_position':particle_position,'hist_rule':hist_rule}
    
    z,power_z,s,power_s,rho,detune,field,\
        field_s,gainLength,resWavelength,\
        thet_out,gam_out,bunching,spectrum,freq,Ns,history=sase1d_input_part.sase(inp_struct)
    
    
    # check unduK is biger than 0, and smaller than 3.5
    
    '''modified by huang, do linear  extrapolation instead of zero map, which makes truth function interrupt in the margin'''
    if unduK[-1]<0:
        power_z[-1]= (1 - (unduK[-1]-0)*100)*10**(10)
    elif param_K>0:
        power_z[-1]= (1.8428 - (param_K-0)*100)*10**(10)
    
    
    return z,power_z    