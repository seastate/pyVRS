
from attrdict import AttrDict
from pyVRSmorph import MorphPars, SimPars, Morphology, MorphologyND
from pyVRSflow import VRSsim
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
#import pyVRSmorph as mrph
#import pyVRSflow as flw
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
'''
mp.new_geom_dim()      # create geom_pars dict

mp.calc_geom_dim()     # calculate derived variables in geom_pars
mp.new_sim_dim(pars={'phi':0,'Tmax':100,'dudz':-0.2,'dwdx':0.2})    # create sim_pars dict (can be modified without recalculation)
#mp.new_sim_dim(pars={'phi':0,'Tmax':100,'dudz':-0.4,'dwdx':0.4})    # create sim_pars dict (can be modified without recalculation)

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
#===============================================================
# Create shortcuts
mp.shape_pars =  AttrDict({'alpha_s': 2.8000000000000003, 'eta_s': 0.2857142857142857, 'alpha_i': 2.3333333333333335, 'eta_i': 0.2857142857142857, 'xi': 0.2857142857142857, 'beta': 1.2195121951219512, 'rho_t': 10.388349514563107, 'rho_i': 9.70873786407767, 'gamma': 0.1, 'd_s': 0.11285593694928399, 'nlevels_s': (16, 12), 'd_i': 0.09404661412440334, 'nlevels_i': (12, 8)})
sh_pars = mp.shape_pars
print('shape_pars = ',sh_pars)

mp.gen_morphND()
fpath = mp.save_morphND()

mp2 = MorphPars()
mp2.load_morphND(fullpath=fpath)

'''
sim_parsND=SimPars(dudz=sparsND.dudz,dvdz=sparsND.dvdz,dwdx=sparsND.dwdx,
                 Tmax=sparsND.Tmax,cil_speed=sparsND.cil_speed,
                 phi=sparsND.phi,theta=sparsND.theta,psi=sparsND.psi)

SimND = VRSsim(morph=MND,fignum=68)
#SimND.run(XEinit=sim_parsND.XEinit,Tmax=4400,cil_speed=0*sim_parsND.cil_speed,
#          U_const_fixed=sim_parsND.U_const_fixed,S_fixed=sim_parsND.S_fixed,dt=1.,dt_plot=25.)
#SimND.run(XEinit=sim_parsND.XEinit,Tmax=sim_pars.Tmax,cil_speed=1*sim_parsND.cil_speed,vel_scale=1/gpars.tau,
#          U_const_fixed=sim_parsND.U_const_fixed,S_fixed=sim_parsND.S_fixed)
SimND.run(XEinit=sim_parsND.XEinit,Tmax=sim_parsND.Tmax,cil_speed=1*sim_parsND.cil_speed,
          U_const_fixed=sim_parsND.U_const_fixed,S_fixed=sim_parsND.S_fixed,dt=1.,dt_plot=25.)




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

'''




