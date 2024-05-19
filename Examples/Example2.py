"""
This code demonstrates scaling properties, testing whether the dimensional analysis
 correctly identifies different scenarios that are scales models of each other.
"""
# Import libraries, classes etc. from Example1
from Examples.Example1 import *

#============================================================================
# test that scaling/nondimensionalization works as it should, by calculating
# and simulating the nondimensional version of the scenario above.
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
#spND.plotSim = 'all'
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
# nondimensional results
timeND = np.asarray(vsND.time)[skip_stat:]
velocityND = np.asarray(vsND.velocity)[skip_stat:,:]
positionND = np.asarray(vsND.position)[skip_stat:,:]
extflowND = np.asarray(vsND.extflow)[skip_stat:,:]

# Dimensional vs. rescaled nondimensional results
figureR = plt.figure(num=8)
figureR.clf()
axesR1 = figureR.add_subplot(1,2,1,projection='3d')
# plot dimensional positions
axesR1.plot(positionD[:,0],positionD[:,1],positionD[:,2],'r')
# plot rescaled nondimensional positions
axesR1.plot(l_D*positionND[:,0],l_D*positionND[:,1],l_D*positionND[:,2],'b-.')
axesR1.set_xlabel('X')
axesR1.set_ylabel('Y')
axesR1.set_zlabel('Z')
axesR2 = figureR.add_subplot(1,2,2,projection='3d')
# plot dimensional velocitiestau
axesR2.plot(velocityD[:,0],velocityD[:,1],velocityD[:,2],'r')
# plot rescaled nondimensional velocities
axesR2.plot(l_D/tauD*velocityND[:,0],l_D/tauD*velocityND[:,1],l_D/tauD*velocityND[:,2],'b-.')
axesR2.set_xlabel('U')
axesR2.set_ylabel('V')
axesR2.set_zlabel('W')


# Nondimensional vs. rescaled dimensional results
figureS = plt.figure(num=9)
figureS.clf()
axesS1 = figureS.add_subplot(1,2,1,projection='3d')
# plot nondimensional positions
axesS1.plot(positionND[:,0],positionND[:,1],positionND[:,2],'b')
# plot rescaled nondimensional positions
axesS1.plot(1/l_D*positionD[:,0],1/l_D*positionD[:,1],1/l_D*positionD[:,2],'r-.')
axesS1.set_xlabel('X')
axesS1.set_ylabel('Y')
axesS1.set_zlabel('Z')

axesS2 = figureS.add_subplot(1,2,2,projection='3d')
# plot dimensional velocities
axesS2.plot(velocityND[:,0],velocityND[:,1],velocityND[:,2],'b')
# plot rescaled nondimensional velocities
axesS2.plot(tauD/l_D*velocityD[:,0],tauD/l_D*velocityD[:,1],tauD/l_D*velocityD[:,2],'r-.')
axesS2.set_xlabel('U')
axesS2.set_ylabel('V')
axesS2.set_zlabel('W')



