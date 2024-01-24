#
#   Data handling utilities and functions for the pyVRS hydrodynamics codes
#
from attrdict import AttrDict
from pyVRSmorph import MorphPars, SimPars, Morphology, MorphologyND, base_densities
from pyVRSflow import VRSsim
from matplotlib import pyplot as plt
import numpy as np
from math import pi
import os
#from meshSpheroid import chimeraSpheroid
import pickle
from copy import deepcopy


#==============================================================================
class DataManager():
    """
        A class to faciliate batch collection, analysis and plotting of swimming
        performance metrics for the pyVRS swimming embryo model.
    """
    def __init__(self,pars={},scale_pars={},**kwargs):
        """ Create a DataManager instance
        """
        # Create a dictionary to store parameters and add default parameters
        self.pars = AttrDict()
        p = self.pars  # Create a shortcut
        # Set default parameters so that the methods can be tested out of the box
        p.alpha_set = np.linspace(1,3.5,6)
        p.beta_set = np.linspace(1,3,5)
        p.eta_set = np.linspace(0.25,0.75,5)
        p.sigma = 0.9
        p.gamma = 0.1
        p.rho_t = 10.388349514563107
        p.rho_i = 9.70873786407767
        p.d_s =  0.11285593694928399
        p.nlevels_s = (16, 16)
        p.d_i = 0.09404661412440334
        p.nlevels_i = (12, 12)
        # update with passed pars
        p.update(pars)
        # Set up scale dictionary, defaulting to nondimensional equivalent
        self.scale_pars = AttrDict({'V_t':1.,'rho_med':1,'mu':1.,'g':1})
        self.scale_pars.update(scale_pars)
        


    def genMorphMatrix(self,mrph_path='MorphFiles',break_num=0):
        """ Set up a matrix of embryo morphologies. For debugging, break>0
            halts execution after break_num morphologies have been calculated.
        """
        p = self.pars  # Create a shortcut
        # Create nested lists for filenames, results etc.
        self.mrph_files = [[[None for i_e in p.eta_set] for i_b in p.beta_set] for i_a in p.alpha_set]
        #
        simnum = p.alpha_set.size * p.beta_set.size * p.eta_set.size
        simcount = 0
        sigma = p.sigma
        set_break = False
        #
        for i_a,alpha in enumerate(p.alpha_set):
            if set_break:
                break
            for i_b,beta in enumerate(p.beta_set):
                if set_break:
                    break
                for i_e,eta in enumerate(p.eta_set):
                    if set_break:
                        break
                    simcount += 1
                    print(f'\n\n********** Calculating morphology {simcount}/{simnum} ************\n\n')
                    self.mp = MorphPars()
                    #
                    xi = (1-eta)*(sigma - ((beta-1)/beta)**(1/3))
                    
                    #
                    self.mp.shape_pars = AttrDict({'alpha_s': alpha, 'eta_s': eta, 'alpha_i': alpha, 'eta_i': eta,
                                      'xi': xi, 'beta': beta, 'gamma': p.gamma,
                                      'rho_t': p.rho_t, 'rho_i': p.rho_i,
                                      'd_s': p.d_s, 'nlevels_s': p.nlevels_s,
                                      'd_i': p.d_i, 'nlevels_i': p.nlevels_i})
                    self.mp.gen_morphND()
                    fpath = self.mp.save_morphND(path=mrph_path)
                    self.mrph_files[i_a][i_b][i_e] = fpath

                    # If requested, break for debugging
                    if simcount == break_num:
                        set_break = True


    def genSimMatrix(self):
        """ Set up a matrix of swimming simulations
        """
        pass
    
