
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


# set up series of simulation parameters
spars = AttrDict({'shear_set': np.linspace(0,.1,3),
                  'Vcil_set': np.linspace(0.,1.5,3),
                  'flow_type': 'rotation',
                  'XEinits': [[0.,0.,0.,pi/5,pi/3.,pi/7.]],
                  'Tmax': 200., 'dt': 0.5,
                  'dt_stat': 10., 'skip_stat': 0,
                  'plot_intvl': 25, 'plotSim': 'end',
                  'first_step':0.05,'resume':False})

DM = DataManager()
DM.loadMorphMatrix(filename='Test/freshMM.mmtx')
DM.genSimMatrix(spars=spars)

# save the DataManager object
filename = 'Freshwater/freshDM.dmgr'
with open(filename, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved DataManager as {filename}')
