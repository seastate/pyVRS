#
#   Data handling utilities and functions for the pyVRS hydrodynamics codes
#
#from attrdict import AttrDict
from MinimalAttrDict import AttrDict
#from pyVRSmorph import MorphPars, SimPars, Morphology, MorphologyND, base_densities
from pyVRSmorph import Morphology
from pyVRSchimera import chimeraParams, shape_scaleParams, chimeraMorphology, print_cp
from pyVRSchimera import ScaleParams, ShapeParams, MeshParams
from pyVRSflow import VRSsim, SimPars
from matplotlib import pyplot as plt
import numpy as np
from math import pi
import os
from copy import deepcopy
#from meshSpheroid import chimeraSpheroid
import pickle


# Define a default library of materials to run out of the box
Materials = AttrDict({'freshwater':AttrDict({'material':'freshwater','density':1000.,'color':np.asarray([0.1,0.3,0.3])}),
                      'seawater':AttrDict({'material':'seawater','density':1030.,'color':np.asarray([0.3,0.3,0.3]),'mu':1030.*1.17e-6}),
                      'brackish':AttrDict({'material':'brackish','density':1015.,'color':np.asarray([0.,1.0,0.])}),
                      'tissue':AttrDict({'material':'tissue','density':1070.,'color':'purple'}),
                      'lipid':AttrDict({'material':'lipid','density':900.,'color':np.asarray([0.,1.,1.])}),
                      'calcite':AttrDict({'material':'calcite','density': 2669., 'color': 'gray'})})
                      #'other': AttrDict({'material':'other','density':None, 'color': None}),

# Define a method and use it to define a default set of material parameters.
# By default, these correspond to nondimensionalized excess density
def get_MatlParams(Materials=Materials,reference_material='seawater',Delta_rho=1.):
    """
       A method to generate a set of nondimensional materials parameters. By default the dimensional
       material properties are taken from the predefined Materials AttrDict, the reference material
       is seawater, and the reference excess density is taken from the predefined ScaleParams AttrDict.
    """
    matl_pars = deepcopy(Materials)
    print('Materials[reference_material].density = ',Materials[reference_material].density)
    for key in matl_pars.keys():
        print(f'Materials[{key}] = {Materials[key]}')
        # the nondimensional excess density
        matl_pars[key].density = (matl_pars[key].density-Materials[reference_material].density)/Delta_rho
        #mat_pars[key].density /= Materials.gamma * Materials[reference_material].density
        # normalize viscosities; almost always only Layer 0 (the medium) will have mu as a key
        if 'mu' in Materials[key].keys():
            Materials[key].mu /= Materials[reference_material].mu
        print(f'matl_pars.{key} = {matl_pars[key]}')
    return matl_pars


#==============================================================================
class MorphMatrix():
    """
        A class to faciliate batch calculations and manipulations with swimming embryo
        morphologies. The default is nondimensional material properties, but dimensional
        ones or other variations can be submitted as an argument.
    """
    def __init__(self,scale_pars=None,msh_pars=None,densities=None,colors=None,
                 alpha_set=None,beta_set=None,eta_set=None,sigma=None):
        """ Create a MorphMatrix instance
        """
        # Create a dictionary to store parameters and add default parameters
        self.pars = AttrDict()
        p = self.pars  # Create a shortcut
        # Record scale, mesh, density and color parameters
        p.scale_pars = scale_pars
        p.msh_pars = msh_pars
        p.densities = densities
        p.colors = colors
        # Record ranges of shape parameters
        p.alpha_set = alpha_set
        p.beta_set = beta_set
        p.eta_set = eta_set
        # a parameter from which to calculate inclusion height (xi --> h_i)
        # from other shape parameters
        p.sigma = sigma
        
    def genMatrix(self,morph_path=None,prefix=None,suffix='mrph',fpars='baer',
                  plotMorph=True,saveMorph=True,calcBody=True,calcFlow=True,fignum=63):
        """ Set up a matrix of embryo morphologies. 
        """
        p = self.pars  # Create a shortcut
        #print('genMatrix: self.pars = ',p)
        # Create nested lists for filenames, results etc.
        self.morph_files = [[[None for i_e in p.eta_set] for i_b in p.beta_set] for i_a in p.alpha_set]
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
                    # calculate an inclusion height parameter
                    xi = (1-eta)*(sigma - ((beta-1)/beta)**(1/3))
                    # create a shape parameter object
                    p.shape_pars = ShapeParams(alpha_s=alpha,eta_s=eta,alpha_i=alpha,eta_i=eta,xi=xi,beta=beta)
                    # calculate the corresponding chimera template, with the default parameter set
                    self.cp = chimeraParams(shape_pars=p.shape_pars,scale_pars=p.scale_pars,
                                       mesh_pars=p.msh_pars,densities=p.densities,
                                       colors=p.colors)
                    #print_cp(cp)
                    # create the filepath 
                    self.morph_files[i_a][i_b][i_e] = \
                        self.name_morph(morph_path=morph_path,prefix=prefix,suffix=suffix,fpars=fpars)
                    # if requested, save the Morphology
                    if saveMorph:
                    # create a chimeraMorphology, and plot it if requested
                        self.cM = chimeraMorphology(chimera_params=self.cp,plotMorph=plotMorph,
                                                    calcFlow=calcFlow,calcBody=calcBody)
                        self.save_morph(fullpath=self.morph_files[i_a][i_b][i_e])

    def name_morph(self,morph_path=None,prefix=None,suffix='mrph',fpars='baer'):
        """
        A function to facilitate saving the current Morphology object
        with an informative name. The parameter fpars determines which 
        parameters will be included in the filename: a = alpha, b = beta, e=eta, 
        d = density (assumed to be for the inclusion layer).
        
        Values for those parameters are taken from the current shape_pars and 
        densities dictionaries.
        """
        # make shortcuts
        p = self.pars  
        sh = p.shape_pars
        # Construct filename and path
        filename = prefix
        for c in fpars:
            if c=='a':
                filename += '_a'+str(round(sh.alpha_s,3))
            if c=='b':
                filename += '_b'+str(round(sh.beta,3))
            if c=='e':
                filename += '_e'+str(round(sh.eta_s,3))
            if c=='d':
                filename += '_d'+str(round(p.densities[2],3))
        # complete the filename with the suffix
        filename += '.' + suffix
        fullpath = os.path.join(morph_path,filename)
        #
        return fullpath

    def save_morph(self,fullpath=None,morph_path=None,prefix=None,suffix='mrph',fpars='baer',fignum=63):
        """
        A function to facilitate saving the current Morphology object.
        with an informative name. The full path can be provided in the
        argument fullpath. Alternatively, it can be supplied by the 
        name_morph method. In that case, the parameter fpars determines which 
        parameters will be included in the filename: a = alpha, b = beta, e=eta, 
        d = density (assumed to be for the inclusion layer).
        
        Values for those parameters are taken from the current shape_pars and densities dictionaries.
        This method should be run after chimeraMorphology and before shape parameters are
        changed, to insure the saved metadata  correct.
        """
        # make shortcuts
        p = self.pars  
        sh = p.shape_pars
        # Get the full path to save the morphologu filename and path
        if fullpath == None:
            fullpath = self.name_morph(self,morph_path=morph_path,prefix=prefix,suffix=suffix,fpars=fpars)
        # save the morphology as a pickle file
        with open(fullpath, 'wb') as handle:
            pickle.dump(self.cM, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Saved morphology as {fullpath}')
        #
        return fullpath

# A function to simplify loading pickled Morphology objects
def load_morph(morph_path=None,filename=None,fignum=None):
    """
        Load the requested Morphology object. filename can either be a full path, or
        the completion of a partial path indicated by morph_path. 
    """
    if morph_path != None:
        fullpath = os.path.join(morph_path,filename)
    else:
        fullpath = filename
    #
    with open(fullpath, 'rb') as handle:
        M = pickle.load(handle)
        print(f'Loaded morphology file {fullpath}')
        if fignum != None:
            # Plot the loaded morphology
            figureM = plt.figure(num=fignum)
            axesM = figureM.add_subplot(projection='3d')
            M.plot_layers(axes=axesM)
            figureM.canvas.draw()
            figureM.canvas.flush_events()
            plt.pause(0.25)
        # return the morphology object
        return M
    
#==============================================================================
class SimMatrix():
    """
        A class to faciliate batch calculations and statistics for a Morphology
        object representing a swimming embryo. spars is a dictionary (or AttrDict)
        of simulation parameters.
    """
    def __init__(self,M=None):   #,spars=None):
        """ Create a SimMatrix instance
        """
        # Store the Morphology object
        self.M = M
        ## Create a dictionary to store parameters and add default parameters
        #self.spars = AttrDict({'shear_set': np.linspace(0,.01,7),
        #                       'Vcil_set': np.linspace(0.,0.4,5),
        #                       'flow_type': 'rotation',
        #                       'XEinits': [[0.,0.,0.,pi/4,0.,pi]],
        #                       'Tmax': 100, 'dt': 0.5,
        #                       'dt_stat': 1., 'skip_stat': 0,
        #                       'plot_intvl': 25, 'plotSim': 'intvl'})
        #if spars != None:
        #    self.spars.update(spars)

        # Initialize a nested list to store results
        #self.reset()

    def gen_data_list(self):
        # (Re)create a nested list to store results
        s = self.spars
        self.sim_data = [[[None for i_x in range(len(s.XEinits))] \
                                for i_v in s.Vcil_set] for i_s in s.shear_set]
        
    def genMatrix(self,spars=None,fignum=47):
        """ Set up a matrix of swimming simulations, and collect basic statistics.
            spars is a dictionary (or AttrDict) used optionally to update the
            current set of simulation parameters.
        """
        # Set up a shortcut and update dictionary
        self.spars = spars
        s = self.spars
        s.update(spars)
        # create list to store results
        self.gen_data_list()
        # Assume that if simulation parameters have changed, need a reset of data list
        #if len(spars) > 0:
        #    self.reset()
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
                    # create a set of simulation parameters
                    self.sp = SimPars(dudz=dudz,dvdz=0.,dwdx=dwdx,U0=0.,U1=0.,U2=0.,
                                 Tmax=s.Tmax,cil_speed=Vcil,
                                 x0=XEinit[0],y0=XEinit[1],z0=XEinit[2],
                                 phi=XEinit[3],theta=XEinit[4],psi=XEinit[5],
                                 dt=s.dt,dt_stat=s.dt_stat,first_step=s.first_step,
                                 plotSim=s.plotSim,plot_intvl=s.plot_intvl,resume=s.resume)
                    # print the current SimPars values
                    print('XEinit = ',XEinit)
                    self.sp.print()
                    # create a simulation object
                    self.vs = VRSsim(morph=self.M,simPars=self.sp,fignum=fignum)
                    # run the simulation
                    self.vs.runSP()   #simPars=self.sp)

                    
                    timeND = np.asarray(self.vs.time)[s.skip_stat:]
                    velocityND = np.asarray(self.vs.velocity)[s.skip_stat:,:]
                    positionND = np.asarray(self.vs.position)[s.skip_stat:,:]
                    extflowND = np.asarray(self.vs.extflow)[s.skip_stat:,:]
                
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
    def __init__(self,MM=None,spars=None):
        """ Create a DataManager instance, using the MorphMatrix object MM if
            is supplied. Record simulation parameters spars if they are supplied.
        """
        self.MM = MM
        # Create an AttrDict to store simulation parameters
        self.spars = AttrDict({})
        # update with parameters from the argument, if any
        if spars != None:
            self.spars.update(spars)
        
    def loadMorphMatrix(self,filename=None):
        """ Load an existing MorphMatrix object.
        """
        with open(filename, 'rb') as handle:
            self.MM = pickle.load(handle)
            print(f'Loaded MorphMatrix file {filename}')

    def saveMorphMatrix(self,filename=None):
        """ Save the current MorphMatrix object.
        """
        with open(filename, 'wb') as handle:
            pickle.dump(self.MM, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Saved MorphMatrix as {filename}')

    def genSimMatrix(self,MM=None,spars=None):
        """ Set up a matrix of swimming simulations, and collect basic statistics,
            for the morphologies in the MorphMatrix, MM.
            spars is an AttrDict of simulation parameters. 
        """
        # Update the MorphMatix object and simulation parameters, if requested
        if MM is not None:
            self.MM = MM
        if spars is not None:
            self.spars.update(spars)
        # Create shortcuts
        s = self.spars
        p = self.MM.pars
        self.alpha_num = range(p.alpha_set.size)
        self.beta_num = range(p.beta_set.size)
        self.eta_num = range(p.eta_set.size)
        # Create nested lists for results etc.
        results_template = [[[None for i_x in range(len(s.XEinits))] for i_v in s.Vcil_set] for i_s in s.shear_set]
        self.sim_data = [[[results_template for i_e in self.eta_num] for i_b in self.beta_num] for i_a in self.alpha_num]
        # The total number of simulations, and a counter...
        simnum = p.alpha_set.size * p.beta_set.size * p.eta_set.size * \
                 s.Vcil_set.size * s.shear_set.size * len(s.XEinits)
        simcount = 0

        # Loops through morphologies
        for i_a in self.alpha_num:          # the alpha index in the simulation matrix
            for i_b in self.beta_num:       # the beta index in the simulation matrix
                for i_e in self.eta_num:    # the eta index in the simulation matrix
                    simcount += 1
                    print(f'********************************************************') 
                    print(f'******Simulating Morphology {simcount} of {simnum}******') 
                    print(f'********************************************************') 
                    # Load selected morphology
                    self.M = load_morph(filename=self.MM.morph_files[i_a][i_b][i_e])
                    self.SM = SimMatrix(M=self.M)
                    self.SM.genMatrix(spars=self.spars)
                    self.sim_data[i_a][i_b][i_e] = self.SM.sim_data


'''
    def genMorphMatrix(self,spars=None):
        """ Set up a matrix of embryo morphologies. MM, a MorphMatrix
            object, is created if it is not passed.

            Note that if a MorphMatrix is passed, it is used unchanged:
            spars and mpars are ignored. This may change in future 
            versions.
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
        if saveMM:
            self.saveMorphMatrix()

'''
