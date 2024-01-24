
from attrdict import AttrDict
from pyVRSmorph import MorphPars, SimPars, Morphology, MorphologyND
from pyVRSflow import VRSsim
from matplotlib import pyplot as plt
plt.ion()
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

#fpath = '/home/dg/VRS/pyVRS/MorphFiles/morph_b3.0_a3.5_e0.25_r10.388.mrph'
#fpath = '/home/dg/VRS/pyVRS/MorphFiles/morph_b2.0_a1.0_e0.25_r10.388.mrph'
fpath = '/home/dg/VRS/pyVRS/MorphFiles/morph_b1.5_a2.0_e0.25_r10.388.mrph'
#fpath = '/home/dg/VRS/pyVRS/MorphFiles/morph_b1.5_a3.0_e0.25_r10.388.mrph'
mp.load_morphND(fullpath=fpath,plotMorph=False)


set_break = False

#alpha_set = np.linspace(1,3.5,6)
#beta_set = np.linspace(1,3,5)
#eta_set = np.linspace(0.25,0.75,5)
#simnum = alpha_set.size * beta_set.size * eta_set.size

shear_set = np.linspace(0,.01,7)
Vcil_set = np.linspace(0.,1.,5)
simnum = shear_set.size * Vcil_set.size

flow_type = 'rotation'
#flow_type = 'horizshear'
#flow_type = 'vertshear'


#Tmax = 100
Tmax = 2500
#Tmax = 5000

simcount = 0

skip_stat = 40
#data = []
data = np.full([simnum,5],None,dtype='float')
dataW = np.full([shear_set.shape[0],Vcil_set.shape[0]],None,dtype='float')
#dataW = np.full([shear_set.shape[0],Vcil_set.shape[0]],0.,dtype='float')



for i_s,shear in enumerate(shear_set):
    if set_break:
        break
    for i_v,Vcil in enumerate(Vcil_set):
        if set_break:
            break
        print(f'\n\n********** Calculating simulation {simcount}/{simnum} ************\n\n')

        if flow_type == 'rotation':
            dudz = -shear
            dwdx = shear
        if flow_type == 'horizshear':
            dudz = shear
            dwdx = 0.
        if flow_type == 'vertshear':
            dudz = 0.
            dwdx = shear

        sim_parsND =  AttrDict({'cil_speed': Vcil, 'dudz':dudz,
                                'dwdx': dwdx, 'dvdz': 0.0, 'Tmax': Tmax,
                                'theta': 0.78539, 'phi': 0, 'psi': 3.141592653589793,
                                'x0': 0.0, 'y0': 0.0, 'z0': 0.0,
#                                'dt':0.5,'dt_stat':0.25,'plot_intvl':20,'plotSim':'end'})
                                'dt':0.5,'dt_stat':1.,'plot_intvl':400,'plotSim':'intvl'})

        #mp = MorphPars()
        #mp.load_morphND(fullpath=fpath,plotMorph=False)
        mp.sim_parsND = sim_parsND
        print('sim_parsND = ',sim_parsND)
        
        mp.gen_simND(run=True,plotSim='end')

        timeND = np.asarray(mp.SimND.time)[skip_stat:]
        velocityND = np.asarray(mp.SimND.velocity)[skip_stat:,:]
        positionND = np.asarray(mp.SimND.position)[skip_stat:,:]
        extflowND = np.asarray(mp.SimND.extflow)[skip_stat:,:]
        
        rel_velND = velocityND[:,0:3]-extflowND
        avg_rel_velND = np.mean(rel_velND,axis=0)

        print(avg_rel_velND.shape)
        #new_data = [shear,Vcil,avg_rel_velND]
        #data.append(new_data)
        new_data = np.asarray([shear,Vcil,avg_rel_velND[0],avg_rel_velND[1],avg_rel_velND[2]])
        data[simcount,:] = new_data
        dataW[i_s,i_v] = avg_rel_velND[2]
        
        print('new_data = ',new_data)
        
        figW = plt.figure(num=5)
        figW.clf()
        axesW = figW.add_subplot()#projection='3d')
        pc = axesW.pcolor(dataW)
        figW.colorbar(pc,ax=axesW)
        
        simcount += 1
        #if simcount == 3:
        #    set_break = True











            

'''
sim_parsND=SimPars(dudz=sparsND.dudz,dvdz=sparsND.dvdz,dwdx=sparsND.dwdx,
                 Tmax=sparsND.Tmax,cil_speed=sparsND.cil_speed,
                 phi=sparsND.phi,theta=sparsND.theta,psi=sparsND.psi)

SimND = VRSsim(morph=MND,fignum=68)
#SimND.run(XEinit=sim_parsND.XEinit,Tmax=4400,cil_speed=0*sim_parsND.cil_speed,
#          U_const_fixed=sim_parsND.U_const_fixed,S_fixed=sim_parsND.S_fixed,dt=1.,dt_stat=25.)
#SimND.run(XEinit=sim_parsND.XEinit,Tmax=sim_pars.Tmax,cil_speed=1*sim_parsND.cil_speed,vel_scale=1/gpars.tau,
#          U_const_fixed=sim_parsND.U_const_fixed,S_fixed=sim_parsND.S_fixed)
SimND.run(XEinit=sim_parsND.XEinit,Tmax=sim_parsND.Tmax,cil_speed=1*sim_parsND.cil_speed,
          U_const_fixed=sim_parsND.U_const_fixed,S_fixed=sim_parsND.S_fixed,dt=1.,dt_stat=25.)




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

figureS = plt.figure(num=5)
plt.subplot(311)
#axesS1 = figureS.add_subplot()
plt.plot(time,velocity[:,2],label='abs_vel')
plt.plot(time,rel_vel[:,2],label='rel_vel')
plt.plot(time,extflow[:,2],label='extflow')
plt.legend()

plt.subplot(312)
tau = mp.geom_pars.tau
l = mp.geom_pars.l

#axesS1 = figureS.add_subplot()
plt.plot(time,velocity[:,2],label='abs_vel')
plt.plot(time,rel_vel[:,2],label='rel_vel')
plt.plot(time,extflow[:,2],label='extflow')
plt.plot(timeND*tau,l/tau*velocityND[:,2],label='abs_velND')
plt.plot(timeND*tau,l/tau*rel_velND[:,2],label='rel_velND')
plt.plot(timeND*tau,l/tau*extflowND[:,2],label='extflowND')
#plt.plot(time/tau,tau/l*velocity[:,2],label='abs_vel')
#plt.plot(time/tau,tau/l*rel_vel[:,2],label='rel_vel')
#plt.plot(time/tau,tau/l*extflow[:,2],label='extflow')
#plt.plot(timeND,velocityND[:,2],label='abs_velND')
#plt.plot(timeND,rel_velND[:,2],label='rel_velND')
#plt.plot(timeND,extflowND[:,2],label='extflowND')
plt.legend()


plt.subplot(313)
#axesS1 = figureS.add_subplot()
plt.plot(timeND,velocityND[:,2],label='abs_velND')
plt.plot(timeND,rel_velND[:,2],label='rel_velND')
plt.plot(timeND,extflowND[:,2],label='extflowND')
plt.legend()

'''




