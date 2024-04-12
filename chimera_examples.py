from copy import deepcopy
import numpy as np
#from attrdict import AttrDict
from MinimalAttrDict import AttrDict
from pprint import pprint as pprnt

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
plt.ion()

from pyVRSchimera import chimeraParams, shape_scaleParams, chimeraMorphology, print_cp
from pyVRSflow import SimPars, VRSsim
from pyVRSchimera import ScaleParams
from pyVRSchimera import ShapeParams
from pyVRSchimera import MeshParams

# Define a default library of dimensional material properties to run out of the box
Materials = AttrDict({'freshwater':AttrDict({'material':'freshwater','density':1000.,'color':np.asarray([0.1,0.3,0.3])}),
                      'seawater':AttrDict({'material':'seawater','density':1030.,'color':np.asarray([0.3,0.3,0.3]),'mu':1030.*1.17e-6}),
                      'brackish':AttrDict({'material':'brackish','density':1015.,'color':np.asarray([0.,1.0,0.])}),
                      'tissue':AttrDict({'material':'tissue','density':1070.,'color':'purple'}),
                      'lipid':AttrDict({'material':'lipid','density':900.,'color':np.asarray([0.,1.,1.])}),
                      'calcite':AttrDict({'material':'calcite','density': 2669., 'color': 'gray'})})
                      #'other': AttrDict({'material':'other','density':None, 'color': None}),


# Define a method and use it to define a default set of material parameters.
# By default, these correspond to nondimensionalized excess density
def get_MatlParams(Materials=Materials,reference_material='seawater',Delta_rho=None):
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
        # normalize viscosities; almost always only Layer 0 (the medium) will have mu as a key
        if 'mu' in Materials[key].keys():
            Materials[key].mu /= Materials[reference_material].mu
        print(f'matl_pars.{key} = {matl_pars[key]}')
    return matl_pars


# create parameter AttrDicts, using default parameters
scaleParams = ScaleParams()
shapeParams = ShapeParams()
meshParams = MeshParams()
MatlParams = get_MatlParams(Delta_rho=scaleParams.Delta_rho)
# extract basic properties
densities_ = [MatlParams['seawater'].density,MatlParams['tissue'].density,MatlParams['freshwater'].density]
colors_ = [MatlParams['seawater'].color,MatlParams['tissue'].color,MatlParams['freshwater'].color]


# calculate the corresponding chimera parameter set
cp = chimeraParams(shape_pars=shapeParams,scale_pars=scaleParams,
                  mesh_pars=meshParams,densities=densities_,
                  colors=colors_)
print_cp(cp)

# Modify the chimeraParams object with some reasonable dimensional parameters
# medium properties
cp.Layers[0].mu = 1030.*1.17e-6
cp.Layers[0].density = 1030.
# Surface properties
#cp.Layers[1].V_s
cp.Layers[1].D = 150.e-6
cp.Layers[1].L1 = 100.e-6
cp.Layers[1].L2 = 60.e-6
cp.Layers[1].L0 = cp.Layers[1].L1 +  cp.Layers[1].L2
cp.Layers[1].density = Materials.tissue.density
cp.Layers[1].d = 6.e-6
cp.Layers[1].nlevels = (16,16)
cp.Layers[1].translate = None

#Inclusion properties
#cp.Layers[2].V_i
cp.Layers[2].D = 75.e-6
cp.Layers[2].L1 = 60.e-6
cp.Layers[2].L2 = 40.e-6
cp.Layers[2].L0 = cp.Layers[2].L1 +  cp.Layers[2].L2
cp.Layers[2].d = 5.e-6
cp.Layers[2].h_i = 20.e-6
cp.Layers[2].nlevels = (16,16)
cp.Layers[2].density = Materials.freshwater.density
cp.Layers[2].translate = [0.,0.,cp.Layers[2].h_i]

print_cp(cp)

cM = chimeraMorphology(chimera_params=cp,plotMorph=False,calcFlow=True,calcBody=True)

figureM = plt.figure(num=27)
figureM.clf()
axesM = figureM.add_subplot(projection='3d')
cM.plot_layers(axes=axesM,showFaces=True,showEdges=False)

figureM2 = plt.figure(num=37)
figureM2.clf()
axesM2 = figureM2.add_subplot(projection='3d')
cM.plot_layers(axes=axesM2,showFaces=False,showEdges=True,autoscale=True)


sh,sc,msh,dens,clrs = shape_scaleParams(cp)
pprnt(dict(sh))
pprnt(dict(sc))
pprnt(dens)
pprnt(clrs)








