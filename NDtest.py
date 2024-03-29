#
#   Code to facilitate testing and debugging the nondimensional vs. dimensional
#   versions of the pyVRS codes.
#

from attrdict import AttrDict
from pyVRSmorph import MorphPars

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
import pyVRSmorph as mrph
import pyVRSflow as flw
from meshSpheroid import chimeraSpheroid
import pickle
from copy import deepcopy
#===========================================================================
# Generate dimensional and nondimensional scenarios using default parameters

mp = MorphPars()
#mp.new_geom_dim(d_s=0.5*6e-6,nlevels_s=[2*16,2*12],d_i=0.5*5e-6,nlevels_i=[2*12,2*8])      # create geom_pars dict
#mp.new_geom_dim(L1_s=25e-6,L2_s=25e-6,L1_i=1e-9,L2_i=1e-9,D_i=1e-9)      # create geom_pars dict
#mp.new_geom_dim(L1_i=1e-9,L2_i=1e-9,D_i=1e-9)      # create geom_pars dict
#mp.new_geom_dim(L1_s=2*25e-6,L2_s=2*25e-6,L1_i=1e-9,L2_i=1e-9,D_i=1e-9)      # create geom_pars dict
mp.new_geom_dim()      # create geom_pars dict

mp.calc_geom_dim()     # calculate derived variables in geom_pars
mp.new_sim_dim(pars={'phi':0,'Tmax':100,'dudz':-0.4,'dwdx':0.4})    # create sim_pars dict (can be modified without recalculation)
#mp.new_sim_dim(pars={'phi':0,'Tmax':100,'dudz':-0.4,'dwdx':0.4})   # create sim_pars dict (can be modified without recalculation)

mp.calc_geom2shape()   # create/calculate shape_pars dict from geom_pars
mp.new_geom2scale()    # create scale_pars dict from geom_pars
mp.calc_geom_nondim()   # calculate new geom_parsND from geom and shape parameters

#sav_dim_scale = mp.scale_pars
mp.calc_sim_nondim()   # create new sim_parsND from simulation and scale parameters

# As a test, reconstitute dimensional morphology parameters
mp.geom_pars=AttrDict()  # erase old values
mp.new_shape2geom()    # create new geom_pars from shape and scale parameters
mp.calc_sim_dim()    # create new geom_pars from shape and scale parameters

'''
# print geom_pars
all_keys = set(list(mp.geom_pars.keys()))
for key in all_keys:
    try:
        print(key,mp.geom_pars[key],sav_geom_pars[key],mp.geom_parsND[key])
    except:
        print(key,mp.geom_pars[key],sav_geom_pars[key])

# print sim_pars
all_keys = set(list(mp.sim_pars.keys()))
for key in all_keys:
    print(key,mp.sim_pars[key],sav_sim_pars[key],mp.sim_parsND[key])


# mu = ndt.mp.scale_pars.mu
# l = ndt.mp.geom_pars.l
# tau = ndt.mp.geom_pars.tau
# key='L0_i';[ndt.mp.geom_parsND[key]*l,ndt.mp.geom_pars[key]]
'''
#===============================================================
# Create shortcuts

gpars = mp.geom_pars
spars = mp.sim_pars
gparsND = mp.geom_parsND
sparsND = mp.sim_parsND

sh_pars = mp.shape_pars
sc_pars = mp.scale_pars

print('gpars = ',gpars)
print('gparsND = ',gparsND)
print('shape_pars = ',sh_pars)
print('scale_pars = ',sc_pars)

#===============================================================
# set up the dimensional morphology
CEsurf = chimeraSpheroid(D=gpars.D_s,L1=gpars.L1_s,L2=gpars.L2_s,d=gpars.d_s,nlevels=gpars.nlevels_s)
CEincl = chimeraSpheroid(D=gpars.D_i,L1=gpars.L1_i,L2=gpars.L2_i,d=gpars.d_i,nlevels=gpars.nlevels_i,
                         translate=[0,0,gpars.h_i])

M = mrph.Morphology()
M.check_normals = False
M.gen_surface(vectors=CEsurf.vectors)
# materials parameter can be 'seawater', 'tissue', 'lipid' or 'calcite' 
M.gen_inclusion(vectors=CEincl.vectors,material='freshwater',immersed_in=1)

'''
# Plot the dimensional morphology
figureM = pyplot.figure(num=57)
axesM = figureM.add_subplot(projection='3d')
M.plot_layers(axes=axesM)

figureM.canvas.draw()
figureM.canvas.flush_events()
pyplot.pause(0.25)
'''
#===============================================================
# set up the nondimensional morphology
CEsurfND = chimeraSpheroid(D=gparsND.D_s,L1=gparsND.L1_s,L2=gparsND.L2_s,d=gparsND.d_s,nlevels=gparsND.nlevels_s)
CEinclND = chimeraSpheroid(D=gparsND.D_i,L1=gparsND.L1_i,L2=gparsND.L2_i,d=gparsND.d_i,nlevels=gparsND.nlevels_i,
                         translate=[0,0,gparsND.h_i])

MND = mrph.MorphologyND(gamma=gparsND.gamma)
MND.check_normals = False
MND.gen_surface(vectors=CEsurfND.vectors)
# materials parameter can be 'seawater', 'tissue', 'lipid' or 'calcite' 
MND.gen_inclusion(vectors=CEinclND.vectors,material='freshwater',immersed_in=1)

'''
# Plot the nondimensional morphology
figureMND = pyplot.figure(num=67)
axesMND = figureMND.add_subplot(projection='3d')
MND.plot_layers(axes=axesMND)

figureMND.canvas.draw()
figureMND.canvas.flush_events()
pyplot.pause(0.25)
'''

#===============================================================
'''
# Add rescaled nondimensional morphology to dimensional plot for comparison
MNDc = deepcopy(MND)
MNDc.layers[1].vectors *= mp.geom_pars.l
MNDc.layers[2].vectors *= mp.geom_pars.l

MNDc.plot_layers(axes=axesM)

figureM.canvas.draw()
figureM.canvas.flush_events()
pyplot.pause(0.25)
'''
#===============================================================
# Calculate flow
print('\n\nCalculating dimensional flow properties')
M.body_calcs()
M.flow_calcs(surface_layer=1)

print('\n\nCalculating nondimensional flow properties')
MND.body_calcs()
MND.flow_calcs(surface_layer=1)

#horse
#===============================================================
# borrow machinery from ChimeraSwim4.ipynb

class SimPars():
    """
    A simple class to facilitate acquiring and passing VRS simulation
    parameters with interactive_output widgets.
    """
    def __init__(self,dudz=0.,dvdz=0.,dwdx=0.,U0=0.,U1=0.,U2=0.,
                 Tmax=20.,cil_speed=0.5*1000*1e-6,
                 phi=pi/3.,theta=-pi/4.,psi=pi):
        self.dudz = dudz
        self.dvdz = dvdz
        self.dwdx = dwdx
        self.U0 = U0
        self.U1 = U1
        self.U2 = U2
        self.Tmax = Tmax
        self.cil_speed = cil_speed
        self.S_fixed = np.asarray([0.,0.,dudz,0.,0.,dvdz,dwdx,0.,0.])
        self.U_const_fixed = np.asarray([U0,U1,U2])
        self.XEinit = np.asarray([0.,0.,0.,phi,theta,psi])

#===============================================================
# run the dimensional simulation

Sim_Pars=SimPars(dudz=spars.dudz,dvdz=spars.dvdz,dwdx=spars.dwdx,
                 Tmax=spars.Tmax,cil_speed=spars.cil_speed,
                 phi=spars.phi,theta=spars.theta,psi=spars.psi)

Sim = flw.VRSsim(morph=M,fignum=58)
Sim.run(XEinit=Sim_Pars.XEinit,Tmax=1*Sim_Pars.Tmax,cil_speed=1*Sim_Pars.cil_speed,
        U_const_fixed=Sim_Pars.U_const_fixed,S_fixed=Sim_Pars.S_fixed)

#===============================================================
# run the nondimensional simulation

Sim_ParsND=SimPars(dudz=sparsND.dudz,dvdz=sparsND.dvdz,dwdx=sparsND.dwdx,
                 Tmax=sparsND.Tmax,cil_speed=sparsND.cil_speed,
                 phi=sparsND.phi,theta=sparsND.theta,psi=sparsND.psi)

print('sim_parsND = ',mp.sim_parsND)

SimND = flw.VRSsim(morph=MND,fignum=68)
#SimND.run(XEinit=Sim_ParsND.XEinit,Tmax=4400,cil_speed=0*Sim_ParsND.cil_speed,
#          U_const_fixed=Sim_ParsND.U_const_fixed,S_fixed=Sim_ParsND.S_fixed,dt=1.,dt_stat=25.)
#SimND.run(XEinit=Sim_ParsND.XEinit,Tmax=Sim_Pars.Tmax,cil_speed=1*Sim_ParsND.cil_speed,vel_scale=1/gpars.tau,
#          U_const_fixed=Sim_ParsND.U_const_fixed,S_fixed=Sim_ParsND.S_fixed)
SimND.run(XEinit=Sim_ParsND.XEinit,Tmax=Sim_ParsND.Tmax,cil_speed=1*Sim_ParsND.cil_speed,
          U_const_fixed=Sim_ParsND.U_const_fixed,S_fixed=Sim_ParsND.S_fixed,dt=1.,dt_stat=25.)




time = np.asarray(Sim.time)
velocity = np.asarray(Sim.velocity)
position = np.asarray(Sim.position)
extflow = np.asarray(Sim.extflow)

timeND = np.asarray(SimND.time)
velocityND = np.asarray(SimND.velocity)
positionND = np.asarray(SimND.position)
extflowND = np.asarray(SimND.extflow)

rel_vel = velocity[:,0:3]-extflow
rel_velND = velocityND[:,0:3]-extflowND

figureS = pyplot.figure(num=5)
pyplot.subplot(311)
#axesS1 = figureS.add_subplot()
pyplot.plot(time,velocity[:,2],label='abs_vel')
pyplot.plot(time,rel_vel[:,2],label='rel_vel')
pyplot.plot(time,extflow[:,2],label='extflow')
pyplot.legend()

pyplot.subplot(312)
tau = mp.geom_pars.tau
l = mp.geom_pars.l

#axesS1 = figureS.add_subplot()
pyplot.plot(time,velocity[:,2],label='abs_vel')
pyplot.plot(time,rel_vel[:,2],label='rel_vel')
pyplot.plot(time,extflow[:,2],label='extflow')
pyplot.plot(timeND*tau,l/tau*velocityND[:,2],label='abs_velND')
pyplot.plot(timeND*tau,l/tau*rel_velND[:,2],label='rel_velND')
pyplot.plot(timeND*tau,l/tau*extflowND[:,2],label='extflowND')
#pyplot.plot(time/tau,tau/l*velocity[:,2],label='abs_vel')
#pyplot.plot(time/tau,tau/l*rel_vel[:,2],label='rel_vel')
#pyplot.plot(time/tau,tau/l*extflow[:,2],label='extflow')
#pyplot.plot(timeND,velocityND[:,2],label='abs_velND')
#pyplot.plot(timeND,rel_velND[:,2],label='rel_velND')
#pyplot.plot(timeND,extflowND[:,2],label='extflowND')
pyplot.legend()


pyplot.subplot(313)
#axesS1 = figureS.add_subplot()
pyplot.plot(timeND,velocityND[:,2],label='abs_velND')
pyplot.plot(timeND,rel_velND[:,2],label='rel_velND')
pyplot.plot(timeND,extflowND[:,2],label='extflowND')
pyplot.legend()

"""
#===============================================================
# Try a rescaled version

mp.scale_pars.V_t *= 8
mp.new_shape2geom()    # create new geom_pars from shape and scale parameters
mp.calc_sim_dim()    # create new geom_pars from shape and scale parameters
mp.calc_geom_dim()     # calculate derived variables in geom_pars
mp.calc_sim_dim()

#===============================================================
# set up the dimensional morphology
CEsurf2 = chimeraSpheroid(D=gpars.D_s,L1=gpars.L1_s,L2=gpars.L2_s,d=gpars.d_s,nlevels=gpars.nlevels_s)
CEincl2 = chimeraSpheroid(D=gpars.D_i,L1=gpars.L1_i,L2=gpars.L2_i,d=gpars.d_i,nlevels=gpars.nlevels_i,
                         translate=[0,0,gpars.h_i])

M2 = mrph.Morphology()
M2.check_normals = False
M2.gen_surface(vectors=CEsurf2.vectors)
# materials parameter can be 'seawater', 'tissue', 'lipid' or 'calcite' 
M2.gen_inclusion(vectors=CEincl2.vectors,material='freshwater',immersed_in=1)


# Calculate flow
print('\n\nCalculating dimensional flow properties')
M2.body_calcs()
M2.flow_calcs(surface_layer=1)

Sim_Pars2=SimPars(dudz=spars.dudz,dvdz=spars.dvdz,dwdx=spars.dwdx,
                 Tmax=spars.Tmax,cil_speed=spars.cil_speed,
                 phi=spars.phi,theta=spars.theta,psi=spars.psi)

Sim2 = flw.VRSsim(morph=M2,fignum=78)
Sim2.run(XEinit=Sim_Pars2.XEinit,Tmax=1*Sim_Pars2.Tmax,cil_speed=1*Sim_Pars2.cil_speed,
        U_const_fixed=Sim_Pars2.U_const_fixed,S_fixed=Sim_Pars2.S_fixed)


#===============================================================
# Try a rescaled version

mp.scale_pars.V_t /= 8**2
mp.new_shape2geom()    # create new geom_pars from shape and scale parameters
mp.calc_sim_dim()    # create new geom_pars from shape and scale parameters
mp.calc_geom_dim()     # calculate derived variables in geom_pars
mp.calc_sim_dim()

#===============================================================
# set up the dimensional morphology
CEsurf2 = chimeraSpheroid(D=gpars.D_s,L1=gpars.L1_s,L2=gpars.L2_s,d=gpars.d_s,nlevels=gpars.nlevels_s)
CEincl2 = chimeraSpheroid(D=gpars.D_i,L1=gpars.L1_i,L2=gpars.L2_i,d=gpars.d_i,nlevels=gpars.nlevels_i,
                         translate=[0,0,gpars.h_i])

M2 = mrph.Morphology()
M2.check_normals = False
M2.gen_surface(vectors=CEsurf2.vectors)
# materials parameter can be 'seawater', 'tissue', 'lipid' or 'calcite' 
M2.gen_inclusion(vectors=CEincl2.vectors,material='freshwater',immersed_in=1)


# Calculate flow
print('\n\nCalculating dimensional flow properties')
M2.body_calcs()
M2.flow_calcs(surface_layer=1)

Sim_Pars2=SimPars(dudz=spars.dudz,dvdz=spars.dvdz,dwdx=spars.dwdx,
                 Tmax=spars.Tmax,cil_speed=spars.cil_speed,
                 phi=spars.phi,theta=spars.theta,psi=spars.psi)

Sim2 = flw.VRSsim(morph=M2,fignum=88)
Sim2.run(XEinit=Sim_Pars2.XEinit,Tmax=1*Sim_Pars2.Tmax,cil_speed=1*Sim_Pars2.cil_speed,
        U_const_fixed=Sim_Pars2.U_const_fixed,S_fixed=Sim_Pars2.S_fixed)



"""











