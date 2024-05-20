
from copy import deepcopy
import numpy as np
from math import pi
#from attrdict import AttrDict
from MinimalAttrDict import AttrDict
from pprint import pprint as pprnt
import pickle

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
plt.ion()

from pyVRSchimera import chimeraParams, shape_scaleParams, chimeraMorphology, print_cp
from pyVRSflow import SimPars, VRSsim
from pyVRSchimera import ScaleParams, ShapeParams, MeshParams
from pyVRSdata import Materials, get_MatlParams, MorphMatrix, load_morph, SimMatrix, DataManager

#====================================================================
# Scaling test, Step 1: Create an example dimensional parameter set
# Because we don't know the shape/scaling parameters, create a dummy set
# of parameters, modify them to reflect the example, then get shape &
# scale parameters.
#====================================================================
# create parameter AttrDicts, using default parameters
scaleParams = ScaleParams(Delta_rho = 1.0)
shapeParams = ShapeParams()
meshParams = MeshParams()
MatlParams = get_MatlParams(Delta_rho=scaleParams.Delta_rho)
# extract basic properties
excess_densities = [MatlParams['seawater'].density,MatlParams['tissue'].density,MatlParams['freshwater'].density]
colors_ = [MatlParams['seawater'].color,MatlParams['tissue'].color,MatlParams['freshwater'].color]

# set up series of shape parameters
alpha_set = np.linspace(1,2.5,4)
beta_set = np.linspace(1,2,3)
eta_set = np.linspace(0.25,0.75,3)
sigma = 0.9

MM = MorphMatrix(scale_pars=scaleParams,msh_pars=meshParams,densities=excess_densities,colors=colors_,
                 alpha_set=alpha_set,beta_set=beta_set,eta_set=eta_set,sigma=sigma)

# precalculate morphology properties
MM.genMatrix(morph_path='Test',prefix='fresh',suffix='mrph',fpars='baed',
                  plotMorph=True,saveMorph=True,calcBody=True,calcFlow=True)

# save the MorphMatrix object
filename = 'Test/freshMM.mmtx'
with open(filename, 'wb') as handle:
    pickle.dump(MM, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved MorphMatrix as {filename}')
