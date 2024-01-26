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

#==============================================================================
class MorphMatrix():
    """
        A class to faciliate batch calculations and manipulations with swimming embryo
        morphologies.
    """
    def __init__(self,pars={},scale_pars={},**kwargs):
        """ Create a MorphMatrix instance
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

        
    def genMatrix(self,morph_path='MorphFiles',pars={}):
        """ Set up a matrix of embryo morphologies. 
        """
        p = self.pars  # Create a shortcut
        
        # Create nested lists for filenames, results etc.
        self.mrph_files = [[[None for i_e in p.eta_set] for i_b in p.beta_set] for i_a in p.alpha_set]
        #
        simnum = p.alpha_set.size * p.beta_set.size * p.eta_set.size
        simcount = 0
        sigma = p.sigma
        #
        for i_a,alpha in enumerate(p.alpha_set):
            for i_b,beta in enumerate(p.beta_set):
                for i_e,eta in enumerate(p.eta_set):
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
                    fpath = self.mp.save_morphND(path=morph_path)
                    self.mrph_files[i_a][i_b][i_e] = fpath

#==============================================================================
class SimMatrix():
    """
        A class to faciliate batch calculations and statistics with swimming embryo
        morphologies, using a MorphPars object.
    """
    def __init__(self,mp=None,spars={},scale_pars={}):
        """ Create a SimMatrix instance
        """
        # Store the MorphPars object
        self.mp = mp
        # Create a dictionary to store parameters and add default parameters
        self.spars = AttrDict({'flow_type': 'rotation',
                               'shear_set': np.linspace(0,.01,7),
                               'Vcil_set': np.linspace(0.,0.4,5),
                               'flow_type': 'rotation',
                               'XEinits': [[0.,0.,0.,pi/4,0.,pi]],
                               'Tmax': 100, 'dt': 0.5,
                               'dt_stat': 1., 'skip_stat': 0,
                               'plot_intvl': 25, 'plotSim': 'intvl'})
        self.spars.update(spars)
        # Set up scale dictionary, defaulting to nondimensional equivalent
        #self.scale_pars = AttrDict({'V_t':1.,'rho_med':1,'mu':1.,'g':1})
        #self.scale_pars.update(scale_pars)

        # Initialize a nested list for results
        self.reset()

    def reset(self):
        # (Re)create a nested list for results
        s = self.spars
        self.sim_data = [[[None for i_x in range(len(s.XEinits))] \
                                for i_v in s.Vcil_set] for i_s in s.shear_set]
        
    def genMatrix(self,spars={}):
        """ Set up a matrix of swimming simulations, and collect basic statistics.

            spars is a dictionary (or AttrDict) used optionally to update the
            current set of simulation parameters.
        """
        # Set up a shortcut and update dictionary
        s = self.spars
        s.update(spars)
        # Assume that if simulation parameters have changed, need a reset of data list
        if len(spars) > 0:
            self.reset()
        # The total number of simulations, and a counter...
        simnum = s.Vcil_set.size * s.shear_set.size * len(s.XEinits)
        simcount = 0
        # Loops through simulation parameters
        for i_s,shear in enumerate(s.shear_set):
            for i_v,Vcil in enumerate(s.Vcil_set):
                for i_x,XEinit in enumerate(s.XEinits):
                    simcount += 1
                    print(f'\n\n********** Calculating simulation {simcount}/{simnum} ************\n\n')
                    if s.flow_type == 'rotation':
                        dudz = -shear
                        dwdx = shear
                    elif s.flow_type == 'horizshear':
                        dudz = shear
                        dwdx = 0.
                    elif s.flow_type == 'vertshear':
                        dudz = 0.
                        dwdx = shear
                        
                    sim_parsND =  AttrDict({'cil_speed': Vcil, 'dudz':dudz,
                                            'dwdx': dwdx, 'dvdz': 0.0, 'Tmax': s.Tmax,
                                            'theta': XEinit[3], 'phi': XEinit[4], 'psi': XEinit[5],
                                            'x0': XEinit[0], 'y0': XEinit[1], 'z0': XEinit[2],
                                            'dt':s.dt,'dt_stat':s.dt_stat,
                                            'plot_intvl':s.plot_intvl,'plotSim':s.plotSim})
                    self.mp.sim_parsND = sim_parsND
                    print('sim_parsND = ',sim_parsND)
                    
                    self.mp.gen_simND(run=True)#,plotSim=s.plotSim)

                    timeND = np.asarray(self.mp.SimND.time)[s.skip_stat:]
                    velocityND = np.asarray(self.mp.SimND.velocity)[s.skip_stat:,:]
                    positionND = np.asarray(self.mp.SimND.position)[s.skip_stat:,:]
                    extflowND = np.asarray(self.mp.SimND.extflow)[s.skip_stat:,:]
                
                    rel_velND = velocityND[:,0:3]-extflowND
                    avg_rel_velND = np.mean(rel_velND,axis=0)
                    velND = velocityND[:,0:3]
                    avg_velND = np.mean(velocityND,axis=0)

                    new_data = np.asarray([avg_velND[0],avg_velND[1],avg_velND[2],
                                           avg_rel_velND[0],avg_rel_velND[1],avg_rel_velND[2]])
                    print('new_data = ',new_data)
                    self.sim_data[i_s][i_v][i_x] = new_data


#==============================================================================
class DataManager():
    """
        A class to faciliate batch collection, analysis and plotting of swimming
        performance metrics for the pyVRS swimming embryo model.
    """
    def __init__(self,dpars={}):
        """ Create a DataManager instance
        """
        # Create a dictionary to store parameters and add default parameters
        self.dpars = AttrDict()
        d = self.dpars  # Create a shortcut
        # Set default parameters so that the methods can be tested out of the box
        #d.morph_path =
        d.mm_path =  'MorphFiles'
        d.mm_filename = 'MorphMatrix.mm'
        self.spars = AttrDict()
        
    def loadMorphMatrix(self,mm_path=None,mm_filename=None):
        """ Load an existing MorphMatrix object.
        """
        d = self.dpars  # Create a shortcut
        # Update parameters if requested
        if mm_path is not None:
            d.mm_path = mm_path
        if mm_filename is not None:
            d.mm_filename = filename
        # Load requested file
        fullpath = os.path.join(d.mm_path,d.mm_filename)
        with open(fullpath, 'rb') as handle:
            self.MM = pickle.load(handle)
            print(f'Loaded MorphMatrix file {fullpath}')

    def genMorphMatrix(self,MM=None,dpars={},mpars={},scale_pars={}):
        """ Set up a matrix of embryo morphologies.
        """
        d = self.dpars  # Create a shortcut and update parameters dictionary
        d.update(dpars)
        # Store or create MorphMatrix object
        if MM is not None:
            self.MM = MM
        else:
            self.MM = MorphMatrix(pars=mpars,scale_pars=scale_pars)
        #
        self.MM.genMatrix(morph_path=d.mm_path,pars={})

        # If requested, save the MorphMatrix object:
        #              None (default): Save to morph_path
        #              path: save to path
        #              False: do not save
        if d.mm_path == False:
            return
        if d.mm_path != None:
            fullpath = os.path.join(d.mm_path,d.mm_filename)
        else:
            fullpath = os.path.join(d.morph_path,d.mm_filename)
        with open(fullpath, 'wb') as handle:
            #print(f'Saving morphology as {fc_s.selected}')
            pickle.dump(self.MM, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved MorphMatrix as {fullpath}')


    def genSimMatrix(self,MM=None,spars={}):
        """ Set up a matrix of swimming simulations, and collect basic statistics.
        """
        # The MorphMatix object should either be passed or already an attribute
        if MM is not None:
            self.MM = MM
        # Create shortcuts
        s = self.spars
        p = self.MM.pars
        # Create a MorphPars object to handle parameters
        self.mp = MorphPars()
        # Default to run the complete matrix of morphologies; this can be modified in
        # the spars argument.
        self.spars = AttrDict({'alpha_num': range(p.alpha_set.size),
                               'beta_num': range(p.beta_set.size),
                               'eta_num': range(p.eta_set.size),
                               'shear_set': np.linspace(0,.01,7),
                               'Vcil_set': np.linspace(0.,0.4,5),
                               'flow_type': 'rotation',
                               'XEinits': [[0.,0.,0.,pi/4,0.,pi]],
                               'Tmax': 100, 'dt': 0.5,
                               'dt_stat': 1., 'skip_stat': 0,
                               'plot_intvl': 25, 'plotSim': 'intvl'})
        self.spars.update(spars)
        s = self.spars # Create shortcuts
        # Create nested lists for results etc.
        results_template = [[[None for i_x in range(len(s.XEinits))] for i_v in s.Vcil_set] for i_s in s.shear_set]
        self.sim_data = [[[results_template for i_e in s.eta_num] for i_b in s.beta_num] for i_a in s.alpha_num]
        # The total number of simulations, and a counter...
        simnum = p.alpha_set.size * p.beta_set.size * p.eta_set.size * \
                 s.Vcil_set.size * s.shear_set.size * len(s.XEinits)
        simcount = 0

        # Loops through morphologies
        for i_a in s.alpha_num:
            for i_b in s.beta_num:
                for i_e in s.eta_num:
                    # Load selected morphology
                    fpath = self.MM.mrph_files[i_a][i_b][i_e]
                    self.mp.load_morphND(fullpath=fpath,plotMorph=False)
                    self.SM = SimMatrix(mp=self.mp,spars=s)
                    self.SM.genMatrix()
                    self.sim_data[i_a][i_b][i_e] = self.SM.sim_data



