#
#   Submodule containing class definitions and methods to create and perform
#   calculations with "constitutive chimera" morphologies for Volume Rendered
#   Swimmer hydrodynamic calculations.
#
import numpy as np
from matplotlib import pyplot as plt
from math import pi
from attrdict import AttrDict
import pickle
from copy import deepcopy
import os

#from pyVRSutils import n2s_fmt
#from pyVRSflow import Stokeslet_shape, External_vel3, larval_V, solve_flowVRS, R_Euler, VRSsim
from meshSpheroid import chimeraSpheroid
from pyVRSmorph import Morphology
#from pyVRSmorph import Layer, Surface, Inclusion, Medium


# Define a library of materials to run out of the box (these can be modified or replaced when modules are invoked)
Materials = AttrDict({'freshwater':AttrDict({'material':'freshwater','density':1000.,'color':np.asarray([0.1,0.3,0.3])}),
                      'seawater':AttrDict({'material':'seawater','density':1030.,'color':np.asarray([0.3,0.3,0.3]),'mu':1030.*1.17e-6}),
                      'brackish':AttrDict({'material':'brackish','density':1015.,'color':np.asarray([0.,1.0,0.])}),
                      'tissue':AttrDict({'material':'tissue','density':1070.,'color':'purple'}),
                      'lipid':AttrDict({'material':'lipid','density':900.,'color':np.asarray([0.,1.,1.])}),
                      'calcite':AttrDict({'material':'calcite','density': 2669., 'color': 'gray'})})
                      #'other': AttrDict({'material':'other','density':None, 'color': None}),
                      #'g': 9.81,'Delta_rho':10.})

# Define a default set of scale parameters, corresponding to the nondimensional case
ScaleParams = AttrDict({'V_t':1.,'mu':1.,'Delta_rho':1.,'g':1.})
#ScaleParams = AttrDict({'V_t':1.,'mu':1030.*1.17e-6,'Delta_rho':1.,'g': 9.81})

# Define a default set of shape parameters. From the pyVRSdata mockup, use a formula to set xi below from sigma
ShapeParams = Attrdict({'alpha_s':2.,'eta_s':0.3,'alpha_i':2.,'eta_i':0.3,'sigma':0.9,'beta':1.2})
# the original defaults...
#ShapeParams = Attrdict({'alpha_s':2.,'eta_s':0.3,'alpha_i':2.,'eta_i':0.3,'xi':0.3,'beta':1.2})

# Define a default set of mesh parameters (these determine triangulation of the surface and inclusion)
MeshParams = Attrdict({'d_s':0.11285593694928399,'nlevels_s':(16,16),'d_i':0.09404661412440334,'nlevels_i':(16,16)})

# Define a method and use it to define a default set of material parameters.
# By default, these correspond to nondimensionalized excess density
def get_MatlParams(Materials=Materials,reference_material='seawater',Delta_rho=ScaleParams['Delta_rho']):
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
        matl_pars[key].density = (matl_pars[key].density-Materials[reference_material].density)/Delta_rho]
        #mat_pars[key].density /= Materials.gamma * Materials[reference_material].density
        # normalize viscosities; almost always only Layer 0 (the medium) will have mu as a key
        if 'mu' in Materials[key].keys():
            Materials[key].mu /= Materials[reference_material].mu
        print(f'matl_pars.{key} = {matl_pars[key]}')
    return matl_pars

MatlParams = get_MatlParams()

def chimeraParams(shape_pars=None,scale_pars=None,
                  mesh_pars=None,densities=[None,None,None],
                  colors=[None,None,None],
                  chimera_pars=AttrDict({})):
    """
       A function to calculate geometric and  material parameters, as expected by the chimeraSpheroid 
       class to generate a constitutive chimera. 

       When scale parameters are all equal to 1, the result is the nondimensional case. When scale 
       parameters are dimensional, the result is in physical units.
    
       Currently, parameters are expected for three layers, in order:
           Layer 0: Medium
           Layer 1: Surface
           Layer 2: Inclusion
       A future version may enable multiple Inclusions, or other variations.

       By default, chimera parameters are returned as an AttrDict. Alternatively, a dictionary or
       AttrDict can be passed in the chimera_pars argument.

       NOTE: Parameters returned in the Surface and Inclusion fields of Layers (i.e., items 1 and 2)
             are a superset of the geometric parameters to be passed to chimeraSpheroid, so their
             interpretation and format are set by that class.

       To avoid ambiguity, parameters provided as arguments are not duplicated in the returned
       parameter set, except when necessary (e.g., mu) to implement simulations.
    """
    # Define some shortcuts
    V_t = scale_pars['V_t']
    Delta_rho = scale_pars['Delta_rho']
    g = scale_pars['g']
    beta = shape_pars['beta']
    mu = scale_pars['mu'] 
    # Calcuate length and time scales.
    l = V_t**(1./3.)
    tau = mu / (Delta_rho * g * l)
    # Retain these in the returned parameters dictionary because they will be needed
    # to parameterize simulations.
    chimera_pars['l'] = l
    chimera_pars['tau'] = tau
    # Create a list of AttrDicts to carry parameters for the three layers
    chimera_pars['Layers'] = [AttrDict({}),AttrDict({}),AttrDict({})]
    # Consistent with Morphology, the 0th Layer is the medium
    ccM = chimera_pars['Layers'][0]  # make a shortcut
    ccM.layer_type = 'medium'
    ccM.color = colors[0]
    ccM.density = Delta_rho * densities[0]
    ccM.mu = mu
    # Create a shortcut for the AttrDict of surface parameters
    ccS = chimera_pars['Layers'][1]
    ccS.layer_type = 'surface'
    ccS.color = colors[1]
    # parse parameters
    ccS.density = Delta_rho * densities[1]
    alpha_s = shape_pars['alpha_s']
    eta_s = shape_pars['eta_s']
    # Calculate surface volume
    ccS.V_s = beta * V_t
    # Calculate dimensions of surface and inclusion chimeras
    ccS.D = (6.*beta/(pi*alpha_s))**(1./3.) * l
    ccS.L0 = alpha_s * ccS.D
    ccS.L2 = eta_s * ccS.L0
    ccS.L1 = (1.-eta_s) * ccS.L0
    # Add triangulation (mesh) parameters  TODO: check for scaling of d and possibly nlevels
    ccS.d = mesh_pars['d_s']
    ccS.nlevels = mesh_pars['nlevels_s']
    # Create a shortcut for the AttrDict of inclusion parameters
    ccI = chimera_pars['Layers'][2]
    ccI.layer_type = 'inclusion'
    ccI.color = colors[2]
    # parse parameters
    ccI.density = Delta_rho * densities[2]
    alpha_i = shape_pars['alpha_i']
    eta_i = shape_pars['eta_i']
    sigma = shape_pars['sigma']
    xi = (1-eta_i)*(sigma - ((beta-1)/beta)**(1./3.))
    # Calculate inclusion volume
    ccI.V_i = (beta-1.) * V_t
    # Calculate dimensions of surface and inclusion chimeras
    ccI.D = (6.*beta/(pi*alpha_i))**(1./3.) * l
    ccI.L0 = alpha_i * ccI.D
    ccI.L2 = eta_i * ccI.L0
    ccI.L1 = (1.-eta_i) * ccI.L0
    # Calculate vertical offset, and package as a translation vector as expected
    # by chimeraSpheroid
    ccI.h_i = xi * ccI.L0
    ccI.translate = [0.,0.,ccI.h_i]
    # Add triangulation (mesh) parameters  TODO: check for scaling of d and possibly nlevels
    ccI.d = mesh_pars['d_i']
    ccI.nlevels = mesh_pars['nlevels_i']
    #
    return chimera_pars


def chimeraMorphology(M=Morphology(),shape_pars=None,scale_pars=None,mesh_pars=None,
                      densities=[None,None,None],colors=[None,'purple',np.asarray([0.,1.,1.])],
                      plotMorph=True,calcFlow=True,calcBody=True):
    """
       A function to facilitate generating a Morphology in the form of a constitutive
       chimera using the supplied shape, scale, material and mesh parameters.
    
       If a Morphology object is not provided, it is generated.
    """
    # Calculate dimensional chimera parameters
    chimera_parameters = chimeraParams(shape_pars=shape_pars,scale_pars=scale_pars,
                                       mesh_pars=mesh_pars,densities=densities,colors=colors)
    # create shortcuts
    cp = chimera_parameters
    cpM = cp.Layers[0]  # the medium layer parameters
    cpS = cp.Layers[1]  # the surface layer parameters
    cpI = cp.Layers[2]  # the inclusion layer parameters
    # scaling parameters
    l = cp.l
    tau = cp.tau
    # Set up the Medium layer in the morphology M
    M. gen_medium(density=cpM.density,mu=cpM.mu)
    # Set up Surface layer
    surface_chimera = chimera_spheroid(geomPars=cpS)
    M.gen_surface(vectors=surface_chimera.vectors,density=cpS.density,immersed_in=0,
                    color=cpS.color,get_points=True,check_normals=False,get_points=True,
                    tetra_project=0.03,tetra_project_min=0.01e-6)
    # If V_i > 0, set up the Inclusion layer
    if cpI.V_i > 0.:
        inclusion_chimera = chimera_spheroid(geomPars=cpI)
        M.gen_inclusion(vectors=inclusion_chimera.vectors,density=cpI.density,
                        immersed_in=1,color=cpI.color)
    # If requested, plot morphology
    if plotMorph:
        figureMND = plt.figure(num=67)
        figureMND.clf()
        axesMND = figureMND.add_subplot(projection='3d')
        self.MND.plot_layers(axes=axesMND)
        figureMND.canvas.draw()
        figureMND.canvas.flush_events()
        plt.pause(0.25)
        if calcBody:  # if requested, calculate buoyancy & gravity forces
            M.body_calcs()
        if calcFLow:  # if requested, calculate flow around the surface
            M.flow_calcs(surface_layer=1)
    # Return the Morphology object
    return M

