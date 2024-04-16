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
MatlParams = get_MatlParams(Delta_rho=scaleParams.Delta_rho)
# extract basic properties
densities_ = [MatlParams['seawater'].density,MatlParams['tissue'].density,MatlParams['freshwater'].density]
colors_ = [MatlParams['seawater'].color,MatlParams['tissue'].color,MatlParams['freshwater'].color]

# calculate the corresponding chimera template, with the default parameter set
cp = chimeraParams(shape_pars=shapeParams,scale_pars=scaleParams,
                  mesh_pars=meshParams,densities=densities_,
                  colors=colors_)
# cp is now the template (default) chimera parameter set
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
#cp.Layers[1].d = 6.e-6
cp.Layers[1].nd = 12 #16
cp.Layers[1].nlevels = (16,16)
cp.Layers[1].translate = None

#Inclusion properties
#cp.Layers[2].V_i
cp.Layers[2].D = 80.e-6
cp.Layers[2].L1 = 60.e-6
cp.Layers[2].L2 = 50.e-6
cp.Layers[2].L0 = cp.Layers[2].L1 +  cp.Layers[2].L2
cp.Layers[2].h_i = 30.e-6
#cp.Layers[2].d = 5.e-6
cp.Layers[2].nd = 12 #16
cp.Layers[2].nlevels = (16,16)
cp.Layers[2].density = Materials.freshwater.density
cp.Layers[2].translate = [0.,0.,cp.Layers[2].h_i]
# print the modified chimera parameters, reflecting a plausible example
print_cp(cp)

#============================================================================
# Calculate the corresponding shape/scale params
sh,scD,msh,dens,clrs = shape_scaleParams(cp)
print('Dimensional parameters:')
print(dict(sh))
print(dict(scD))
print(dens)
print(dict(msh))
print(clrs)

cpD = chimeraParams(shape_pars=sh,scale_pars=scD,
                  mesh_pars=msh,densities=dens,
                  colors=clrs)
# print to compare to cp:
print_cp(cpD)
# record dimensional length and time scales
l_D = cpD.l
tauD = cpD.tau

cM = chimeraMorphology(chimera_params=cpD,plotMorph=False,calcFlow=True,calcBody=True)

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
sp.Tmax = 50.
sp.cil_speed = 500.e-6
sp.dt = 0.1
#sp.plotSim = 'end'
sp.plotSim = 'intvl'
sp.plot_intvl = 50
sp.print()
# create a simulation object
vs = VRSsim(morph=cM,simPars=sp,fignum=47)
# run the simulation
vs.runSP(simPars=sp)


============================================================================
# test that scaling/nondimensionalization works as it should
scND = ScaleParams(V_t=1.,mu=1.,Delta_rho=1.,g=1.)  # the nondimensional (default) scaling parameters
# create a corresponding nondimensional version of cp; parameters are the same, except scaling parameters
cpND = chimeraParams(shape_pars=sh,scale_pars=scND,
                  mesh_pars=msh,densities=dens,
                  colors=clrs)
print_cp(cpND)
cMND = chimeraMorphology(chimera_params=cpND,plotMorph=False,calcFlow=True,calcBody=True)

figureMND = plt.figure(num=28)
figureMND.clf()
axesMND = figureMND.add_subplot(projection='3d')
cMND.plot_layers(axes=axesMND,showFaces=True,showEdges=False)

figureMND2 = plt.figure(num=38)
figureMND2.clf()
axesMND2 = figureMND2.add_subplot(projection='3d')
cMND.plot_layers(axes=axesMND2,showFaces=False,showEdges=True,autoscale=True)

#===========================================
# create a set of simulation parameters
spND = deepcopy(sp)
# convert dimensional simulation parameters to nondimensional simulation parameters,
# using the dimensional length and time scale
spND.toND(l=l_D,tau=tauD)
spND.plotSim = 'all'
#spND.dt = 0.01
spND.print()
# create a simulation object
vsND = VRSsim(morph=cMND,simPars=spND,fignum=48)
# run the simulation
vsND.runSP(simPars=spND)



#===========================================
# graph some result comparisons
skip_stat = 0
# dimensional results
timeD = np.asarray(vs.time)[skip_stat:]
velocityD = np.asarray(vs.velocity)[skip_stat:,:]
positionD = np.asarray(vs.position)[skip_stat:,:]
extflowD = np.asarray(vs.extflow)[skip_stat:,:]
# dimensional results
timeND = np.asarray(vsND.time)[skip_stat:]
velocityND = np.asarray(vsND.velocity)[skip_stat:,:]
positionND = np.asarray(vsND.position)[skip_stat:,:]
extflowND = np.asarray(vsND.extflow)[skip_stat:,:]

figureR = plt.figure(num=8)
figureR.clf()
axesR1 = figureR.add_subplot(1,2,1,projection='3d')
# plot dimensional positions
axesR1.plot(positionD[:,0],positionD[:,1],positionD[:,2],'r')
# plot rescaled nondimensional positions
axesR1.plot(l*positionND[:,0],l*positionND[:,1],l*positionND[:,2],'b-.')
axesR1.set_xlabel('X')
axesR1.set_ylabel('Y')
axesR1.set_zlabel('Z')

axesR2 = figureR.add_subplot(1,2,2,projection='3d')
# plot dimensional velocities
axesR2.plot(velocityD[:,0],velocityD[:,1],velocityD[:,2],'r')
# plot rescaled nondimensional velocities
axesR2.plot(l/tau*velocityND[:,0],l/tau*velocityND[:,1],l/tau*velocityND[:,2],'b-.')
axesR2.set_xlabel('U')
axesR2.set_ylabel('V')
axesR2.set_zlabel('W')

#self.axes1 = self.figV.add_subplot(1,2,1,projection='3d')
#self.axes2 = self.figV.add_subplot(1,2,2,projection='3d')
