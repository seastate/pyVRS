#
#   Submodule containing class definitions and methods to create and modify
#   morphologies for Volume Rendered Swimmer hydrodynamic calculations.
#

from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from matplotlib.colors import LightSource
import numpy as np
#import math

from attrdict import AttrDict


#==============================================================================
def Stokeslet_shape(X,P,alpha,mu):
    """	This function returns the velocities at the points represented by the
	list of vectors X exerted by the Stokeslets at the points P with forces
	alpha. X, P and alpha are _x3 np arrays.
    """
    eps = np.finfo(float).eps # machine epsilon to prevent division by zero
    # dimensions of inputs
    nx = X.shape[0]
    np = P.shape[0]
    # Create arrays for resultant velocities
    U1 = np.zeros([nx,3])
    U2 = np.zeros([nx,3])
    U3 = np.zeros([nx,3])
    zzz = np.zeros([nx,1])
    # Loop through evaluation points, vectorizing singularity points
    for i = in range(np):	#	Calculate velocites at each of the points in X
	x = X[:,0] - P[i,0]	#	Distances from each point in P
	y = X[:,1] - P[i,1]
	z = X[:,2] - P[i,2]
        alpha1 = 1/(8*np.pi*mu) * alpha(i,1)*np.ones([nx,1])
        alpha2 = 1/(8*np.pi*mu) * alpha(i,2)*np.ones([nx,1])
        alpha3 = 1/(8*np.pi*mu) * alpha(i,3)*np.ones([nx,1])
	# Calculate distances. The eps should prevent blowups when X and P are the same
	R = np.sqrt(x.**2+y**2+z**2) + eps
        Rinv = np.divide(ones([nx,1]),R)
        Rinv3 = Rinv**3
        # Calculate induced velocities for alpha components separately
        u = alpha1*Rinv + alpha1*x*Rinv3*x
        v = alpha1*x*Rinv3*y
        w = alpha1*x*Rinv3*z
	U1 += np.concatenate((u,v,w),axis=1)
        u = alpha2*y*Rinv3*x
        v = alpha2*Rinv + alpha2*y*Rinv3*y
        w = alpha2*y*Rinv3*z
	U2 += np.concatenate((u,v,w),axis=1)
        u = alpha3*z*Rinv3*x
        v = alpha3*z*Rinv3*y
        w = alpha3*Rinv + alpha3*z*Rinv3*z
	U3 += np.concatenate((u,v,w),axis=1)
	#U1 = U1 + [alpha1./R zzz zzz] + repmat(alpha1.*x./R.^3,[1 3]) .* [x,y,z]
	#U2 = U2 + [zzz alpha2./R zzz] + repmat(alpha2.*y./R.^3,[1 3]) .* [x,y,z]
	#U3 = U3 + [zzz zzz alpha3./R] + repmat(alpha3.*z./R.^3,[1 3]) .* [x,y,z]
    return U1,U2,U3

#==============================================================================
def External_vel3(X,U_const,S):
    """ This function evaluates the (linearized) external flow at the points X using the
        constant velocity vector U_const and the deriviatives vector S
    """
    nx = X.shape[0]
    U_ext = np.repeat(U_const,nx,axis=0)
    U_ext[:,0] += S[0:2].dot(X.T).T
    U_ext[:,1] += S[3:5].dot(X.T).T
    U_ext[:,2] += S[6:8].dot(X.T).T
    #U_ext = repmat(U_const,nx,1) ...
    #  + [S(1)*X(:,1) + S(2)*X(:,2) + S(3)*X(:,3),...
    #     S(4)*X(:,1) + S(5)*X(:,2) + S(6)*X(:,3),...
    #     S(7)*X(:,1) + S(8)*X(:,2) + S(9)*X(:,3)];
    return U_ext

#==============================================================================
def larval_V(X,V_L,Omega_L):
    """ This function calculates the array of velocity vectors v_larva 
        of the points X on the larval morphology. V_L is the velocity of
        a reference point, (0,0,0), relative to which morphological
        points are defined. Omega_L is the larva's angular velocity, in
        radians per second.
    """
    nx = size(X,1);
    #v_larva = zeros(nx,3);
    #v_larva = zeros(X.shape)
    v_larva = V_L).repeat(nx,axis=0) \
              + np.cross(Omega_L.reshape([1,3]),X)
    #for i = 1:nx,
    #    v_larva(i,:) = V_L + cross(Omega_L,X(i,:));
    #end
    return v_larva

#==============================================================================
def solve_flowVRS(surface,V_L,Omega_L,Q_inv,U_cilia,cil_speed,U_const,S):
    """This function solves for the forces imparted on the cylinders by the flow
       to match boundary conditions for the flow defined by the 
       global parameters below.
    """
    #global Fpt U_cilia cil_speed 
    #global P_center F_center
    #global V_larva
    np = surface.mesh.ctrlpts.shape[0]
    U_ext = External_vel3(surface.mesh.ctrlpts,U_const,S)
    V_larval = larval_V(surface.mesh.ctrlpts,V_L,Omega_L)
    #U_1 = V_larva[:,0] - U_ext[:,0] + cil_speed*U_cilia[:,0]	#	The induced velocity should cancel the external velocity
    #U_2 = V_larva[:,1] - U_ext[:,1] + cil_speed*U_cilia[:,1]
    #U_3 = V_larva[:,2] - U_ext[:,2] + cil_speed*U_cilia[:,2]
    U = [U_1 ; U_2 ; U_3]
    #  U is the set of velocities at the control points    
    U = (V_larva - U_ext + cil_speed*U_cilia).T.reshape([3*np,1])
    #  F is the set of forces ON THE FLUID necessary to induces those velocities.
    F = Q_inv * U
    #  Forces on the organism are equal and opposite
    Fpt = np.ones([np,3])
    Fpt[:,0] = -F[0:np-1]
    Fpt[:,1] = -F[np:2*np-1]
    Fpt[:,2] = -F[2*np:3*np-1] 
    Fpt_ = -F.reshape([np,3])
    Fpt_T = -F.T.reshape([np,3])
    print('Fpt = ',Fpt)
    print('Fpt_ = ',Fpt_)
    print('Fpt_T = ',Fpt_T)
    #   Calculate moments from the positions of the singularities:
    print(surface.mesh.singpts.shape,Fpt.shape)
    Mpt = np.cross(surface.mesh.singpts,Fpt)
    print(Mpt.shape)
    #Mpt = cross(F_center,Fpt,2)
    return Fpt,Mpt
