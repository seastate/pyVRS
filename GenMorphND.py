
from attrdict import AttrDict
from pyVRSmorph import MorphPars, SimPars, Morphology, MorphologyND
from pyVRSflow import VRSsim
from matplotlib import pyplot
pyplot.ion()
from mpl_toolkits import mplot3d
from matplotlib.colors import LightSource
# Import modules
import numpy as np
from math import pi
import os
# set up path to submodules
import sys
#sys.path.append('../../../submodules/')
#sys.path.append('../../../submodules/pyVRS')
#import pyVRSmorph as mrph
#import pyVRSflow as flw
from meshSpheroid import chimeraSpheroid
import pickle
from copy import deepcopy

#===========================================================================
# Generate dimensional and nondimensional scenarios using default parameters

mp = MorphPars()

set_break = False

alpha_set = np.linspace(1,3.5,6)
beta_set = np.linspace(1,3,5)
eta_set = np.linspace(0.25,0.75,5)

simnum = alpha_set.size * beta_set.size * eta_set.size
simcount = 0

sigma = 0.9

for alpha in alpha_set:
    if set_break:
        break
    for beta in beta_set:
        if set_break:
            break
        for eta in eta_set:
            if set_break:
                break
            simcount += 1
            print(f'\n\n********** Calculating morphology {simcount}/{simnum} ************\n\n')
            mp = MorphPars()
            xi = (1-eta)*(sigma - ((beta-1)/beta)**(1/3))
            mp.shape_pars = AttrDict({'alpha_s': alpha, 'eta_s': eta, 'alpha_i': alpha, 'eta_i': eta,
                                      'xi': xi, 'beta': beta, 'gamma': 0.1,
                                      'rho_t': 10.388349514563107, 'rho_i': 9.70873786407767,
                                      'd_s': 0.11285593694928399, 'nlevels_s': (16, 16),
                                      'd_i': 0.09404661412440334, 'nlevels_i': (12, 12)})
            mp.gen_morphND()
            fpath = mp.save_morphND(path='TMP')
            
            #if simcount == 1:
            #    set_break = True







