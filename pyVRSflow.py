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
def Stokeslet_shape(X,C,alpha,mu):
    """	This function returns the velocities at the points represented by the
	list of vectors X exerted by the Stokeslets at the points C with forces
	alpha. X, P and alpha are _x3 np arrays.
    """
    eps = np.finfo(float).eps # machine epsilon to prevent division by zero
    # dimensions of inputs
    #print('X.shape = ',X.shape)
    #print('C.shape = ',C.shape)
    #print('alpha.shape = ',alpha.shape)
    nx = X.shape[0]
    nc = C.shape[0]
    # Create arrays for resultant velocities
    U1 = np.zeros([nx,3])
    U2 = np.zeros([nx,3])
    U3 = np.zeros([nx,3])
    zzz = np.zeros([nx,1])
    # Loop through evaluation points, vectorizing singularity points
    for i in range(nc):	#	Calculate velocites at each of the points in X
        #print(X[:,0].shape,C[i,0].shape,alpha[i,0].shape)
        x = X[:,0].reshape([nx,1]) - C[i,0]	#	Distances from each point in C
        y = X[:,1].reshape([nx,1]) - C[i,1]
        z = X[:,2].reshape([nx,1]) - C[i,2]
        #print('x shapes = ',x.shape,y.shape,z.shape)
        alpha1 = 1/(8*np.pi*mu) * alpha[i,0]*np.ones([nx,1])
        alpha2 = 1/(8*np.pi*mu) * alpha[i,1]*np.ones([nx,1])
        alpha3 = 1/(8*np.pi*mu) * alpha[i,2]*np.ones([nx,1])
        #print('alpha shapes = ',alpha1.shape,alpha2.shape,alpha3.shape)
        # Calculate distances. The eps should prevent blowups when X and C are the same
        R = np.sqrt(x**2+y**2+z**2) + eps
        #R = np.sqrt(x**2+y**2+z**2).reshape([nx,1]) + eps
        Rinv = np.divide(np.ones([nx,1]),R)
        Rinv3 = Rinv**3
        #print('Rshapes = ',R.shape,Rinv.shape,Rinv3.shape)
        # Calculate induced velocities for alpha components separately
        #print('calc shapes = ',(alpha1*Rinv).shape,(alpha1*x*Rinv3*x).shape,
        #      (alpha1*x).shape,(Rinv3*x).shape)
        u = alpha1*Rinv + alpha1*x*Rinv3*x
        v = alpha1*x*Rinv3*y
        w = alpha1*x*Rinv3*z
        #print('U shapes = ',U1.shape,u.shape,v.shape,w.shape)
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
    # Insure shape and dtype are consistent
    u_const = U_const.astype('float').reshape([1,3])
    s = S.astype('float').flatten()
    #print(X.shape)
    #print(S[0:3])
    #print(S[0:3].dot(X.T).shape)
    #print(S[0:3].dot(X.T).T.shape)
    U_ext = np.repeat(u_const,nx,axis=0)
    U_ext[:,0] += s[0:3].dot(X.T).T
    U_ext[:,1] += s[3:6].dot(X.T).T
    U_ext[:,2] += s[6:9].dot(X.T).T
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
    nx = X.shape[0]
    #v_larva = zeros(nx,3);
    #v_larva = zeros(X.shape)
    v_larva = V_L.reshape([1,3]).repeat(nx,axis=0) \
              + np.cross(Omega_L.reshape([1,3]),X)
    #for i = 1:nx,
    #    v_larva(i,:) = V_L + cross(Omega_L,X(i,:));
    #end
    return v_larva

#==============================================================================
def solve_flowVRS(surface,V_L,Omega_L,cil_speed,U_const,S):
    """This function solves for the forces imparted on the cylinders by the flow
       to match boundary conditions for the flow defined by the 
       global parameters below.
    """
    nc = surface.ctrlpts.shape[0]
    U_ext = External_vel3(surface.ctrlpts,U_const,S)
    V_larva = larval_V(surface.ctrlpts,V_L,Omega_L)
    #U_1 = V_larva[:,0] - U_ext[:,0] + cil_speed*U_cilia[:,0]	#	The induced velocity should cancel the external velocity
    #U_2 = V_larva[:,1] - U_ext[:,1] + cil_speed*U_cilia[:,1]
    #U_3 = V_larva[:,2] - U_ext[:,2] + cil_speed*U_cilia[:,2]
    #U = [U_1 ; U_2 ; U_3]
    #  U is the set of velocities at the control points    
    U = (V_larva - U_ext + cil_speed*surface.rel_Ucilia).T.reshape([3*nc,1])
    #  F is the set of forces ON THE FLUID necessary to induces those velocities.
    F = surface.Q_inv.dot(U)
    #F = surface.Q_inv * U
    print(U.shape,surface.Q_inv.shape,F.shape)
    #  Forces on the organism are equal and opposite
    Fpt = np.ones([nc,3])
    print(Fpt[:,0].shape,F[0:nc].flatten().shape)
    Fpt[:,0] = -F[0:nc].flatten()
    Fpt[:,1] = -F[nc:2*nc].flatten()
    Fpt[:,2] = -F[2*nc:3*nc].flatten()
    Fpt_ = -F.reshape([nc,3])
    Fpt_T = -F.T.reshape([nc,3])
    print('Fpt = ',Fpt)
    print('Fpt_ = ',Fpt_)
    print('Fpt_T = ',Fpt_T)
    #   Calculate moments from the positions of the singularities:
    print(surface.singpts.shape,Fpt.shape)
    Mpt = np.cross(surface.singpts,Fpt)
    print(Mpt.shape)
    #Mpt = cross(F_center,Fpt,2)
    return Fpt,Mpt
