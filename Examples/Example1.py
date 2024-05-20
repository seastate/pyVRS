"""
This code demonstrates creating model of an early embryo using two 
constitutive chimeras representing the exterior surface and an inclusion.
It shows the basic usage of codes calculating surface and body forces on 
the embryo, and its swimming in a specified flowfield. It also demonstrates
scaling properties, testing whether the dimensional analysis correctly 
identifies different scenarios that are scales models of each other.

"""

# import required modules
from copy import deepcopy
import numpy as np
from math import pi
from MinimalAttrDict import AttrDict
from pprint import pprint as pprnt

from matplotlib import pyplot as plt
plt.ion()

from pyVRSchimera import chimeraParams, shape_scaleParams, chimeraMorphology, print_cp
from pyVRSchimera import ScaleParams, ShapeParams, MeshParams
from pyVRSflow import SimPars, VRSsim
from pyVRSdata import Materials, get_MatlParams

#====================================================================
# Scaling test, Step 1: Create an example dimensional parameter set
# Because we don't know the shape/scaling parameters, create a dummy set
# of parameters, modify them to reflect the example, then get shape &
# scale parameters.
#====================================================================
# create parameter AttrDicts, using default parameters
scaleParams = ScaleParams()
shapeParams = ShapeParams()
meshParams = MeshParams()
# Get material parameters (by default, scaled by Delta-rho=1, so
# unless this is changed the numerical values for density do not change).
# However the viscosity is normalized by the viscosity of the reference
# material (by default, seawater).
MatlParams = get_MatlParams()
# Extract basic properties, in the order of Layers (Layer 0: medium;
# Layer 1: tissue; Layer 2: inclusion).
excess_densities = [MatlParams['seawater'].density,
                    MatlParams['tissue'].density,
                    MatlParams['freshwater'].density]
colors_ = [MatlParams['seawater'].color,
           MatlParams['tissue'].color,
           MatlParams['freshwater'].color]

# calculate the corresponding chimera template, with the default parameter set
cp = chimeraParams(shape_pars=shapeParams,scale_pars=scaleParams,
                  mesh_pars=meshParams,densities=excess_densities,
                  colors=colors_)
# cp is now the template (default) chimera parameter set; print values
# using the print_cp utility:
print_cp(cp)

# Modify the chimeraParams object with some reasonable dimensional parameters
# medium properties
cp.Layers[0].mu = 1030.*1.17e-6
#cp.Layers[0].density = 1030.
# Surface properties
#cp.Layers[1].V_s
cp.Layers[1].D = 150.e-6
cp.Layers[1].L1 = 100.e-6
cp.Layers[1].L2 = 60.e-6
cp.Layers[1].L0 = cp.Layers[1].L1 +  cp.Layers[1].L2
#cp.Layers[1].density = Materials.tissue.density
#cp.Layers[1].d = 6.e-6
#cp.Layers[1].nd = 12 #16
#cp.Layers[1].nlevels = (16,16)
cp.Layers[1].translate = None

#Inclusion properties
#cp.Layers[2].V_i
cp.Layers[2].D = 80.e-6
cp.Layers[2].L1 = 60.e-6
cp.Layers[2].L2 = 50.e-6
cp.Layers[2].L0 = cp.Layers[2].L1 +  cp.Layers[2].L2
cp.Layers[2].h_i = 30.e-6
#cp.Layers[2].d = 5.e-6
#cp.Layers[2].nd = 12 #16
#cp.Layers[2].nlevels = (16,16)
#cp.Layers[2].density = Materials.freshwater.density
cp.Layers[2].translate = [0.,0.,cp.Layers[2].h_i]
# print the modified chimera parameters, reflecting a plausible example
print_cp(cp)

#============================================================================
# Calculate the corresponding shape/scale params; print values to confirm
sh,scD,msh,dens,clrs = shape_scaleParams(cp)
print('Dimensional parameters:')
print(dict(sh))
print(dict(scD))
print(dens)
print(dict(msh))
print(clrs)
# calculate chimera parameters corresponding to the dimensional characteristics
cpD = chimeraParams(shape_pars=sh,scale_pars=scD,
                  mesh_pars=msh,densities=dens,
                  colors=clrs)
# print to compare to cp:
print_cp(cpD)
# record dimensional length and time scales
l_D = cpD.l
tauD = cpD.tau

# generate the dimensional chimera morphology
cM = chimeraMorphology(chimera_params=cpD,plotMorph=False,calcFlow=True,calcBody=True)
# visualize the surface and tiling
figureM = plt.figure(num=27)
figureM.clf()
axesM = figureM.add_subplot(projection='3d')
cM.plot_layers(axes=axesM,showFaces=True,showEdges=False)

figureM2 = plt.figure(num=37)
figureM2.clf()
axesM2 = figureM2.add_subplot(projection='3d')
cM.plot_layers(axes=axesM2,showFaces=False,showEdges=True,autoscale=True)


#===========================================
# create a set of simulation parameters
sp = SimPars()
#sp.Tmax = 3.
sp.Tmax = 50.
sp.cil_speed = 500.e-6
sp.dt = 0.1
sp.dudz = -0.25
sp.dwdx = 0.25
#sp.plotSim = 'all'
sp.plotSim = 'end'
#sp.plotSim = 'intvl'
sp.plot_intvl = 50
sp.print()
# create a simulation object
vs = VRSsim(morph=cM,simPars=sp,fignum=47)
# run the simulation
vs.runSP(simPars=sp)

