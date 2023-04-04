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
from math import ceil
from scipy.integrate import odeint

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
    #print(U.shape,surface.Q_inv.shape,F.shape)
    #  Forces on the organism are equal and opposite
    Fpt = np.ones([nc,3])
    #print(Fpt[:,0].shape,F[0:nc].flatten().shape)
    Fpt[:,0] = -F[0:nc].flatten()
    Fpt[:,1] = -F[nc:2*nc].flatten()
    Fpt[:,2] = -F[2*nc:3*nc].flatten()
    Fpt_ = -F.reshape([nc,3])
    Fpt_T = -F.T.reshape([nc,3])
    #print('Fpt = ',Fpt)
    #print('Fpt_ = ',Fpt_)
    #print('Fpt_T = ',Fpt_T)
    #   Calculate moments from the positions of the singularities:
    #print(surface.singpts.shape,Fpt.shape)
    Mpt = np.cross(surface.singpts,Fpt)
    #print(Mpt.shape)
    #Mpt = cross(F_center,Fpt,2)
    return Fpt,Mpt

#==============================================================================
def foo():
    """A test function to experiment with function attributes as static vairables
    """
    try:
        foo.counter += 1
    except AttributeError:
        foo.counter = 1

class Foo(object):
    # Class variable, shared by all instances of this class
    counter = 0
    blah = 'blah'
    
    def __call__(self):
        Foo.counter += 1
        print(Foo.counter)

class Foo2():
    def __init__(self):
        self.counter = 0
        self.blah = 'blah'
    
    def __call__(self):
        self.counter += 1
        print(self.counter)

class Foo3():
    def __init__(self):
        self.counter = 0
        self.blah = 'blah'
    
    def __call__(self,X):
        self.counter += 1
        print(self.counter,X)

#==============================================================================
class Flowfield():
    """ This function specifies the external flowfield at the points in
        the m x 3 ndarray X. U_const_fixed and S_fixed are the external
        linearized flow parameters, in fixed (global) coordinates. In
        this function they are preserved as an attribute, initialized
        to default values when the function is called for the first time.
       Thereafter, they can be set with e..g. flowfield.S_fixed = ...
    """
    def __init__(self,U_const_fixed = np.zeros([1,3]),
                             S_fixed = np.zeros([1,9])):
        self.U_const_fixed = U_const_fixed
        self.S_fixed = S_fixed
        
    def __call__(self,X):
        m = X.shape[0]
        U_ext = U_const_fixed.reshape([1,3]).repeat(m,axis=0) \
	    + [S_fixed[0]*X[:,0] + S_fixed[1]*X[:,1] + S_fixed[2]*X[:,2],
	       S_fixed[3]*X[:,0] + S_fixed[4]*X[:,1] + S_fixed[5]*X[:,2],
	       S_fixed[6]*X[:,0] + S_fixed[7]*X[:,1] + S_fixed[8]*X[:,2]]
        return U_ext

#==============================================================================
'''
def flowfield(X):
    """ This function specifies the external flowfield at the points in
        the m x 3 ndarray X. U_const_fixed and S_fixed are the external
        linearized flow parameters, in fixed (global) coordinates. In
        this function they are preserved as an attribute, initialized
        to default values when the function is called for the first time.
       Thereafter, they can be set with e..g. flowfield.S_fixed = ...
    """
    #global U_const_fixed S_fixed
    try:
        U_const_fixed = flowfield.U_const_fixed
        S_fixed = flowfield.S_fixed
    except:
        flowfield.U_const_fixed = np.zeros([1,3])
        flowfield.S_fixed = np.zeros([1,9])
    m = X.shape[0]
    U_ext = U_const_fixed.reshape([1,3]).repeat(m,axis=0) ...
	  + [S_fixed[0]*X[:,0] + S_fixed[1]*X[:,1] + S_fixed[2]*X[:,2],
	     S_fixed[3]*X[:,0] + S_fixed[4]*X[:,1] + S_fixed[5]*X[:,2],
	     S_fixed[6]*X[:,0] + S_fixed[7]*X[:,1] + S_fixed[8]*X[:,2]]
    return U_ext
'''
#==============================================================================
def R_Euler(phi,theta,psi):
    """	This function returns the 3 x 3 rotation matrix for the Euler angles phi, theta and psi.
    """
    R = np.zeros([3,3])
    R[0,0] = cos(psi) * cos(phi) - cos(theta)*sin(phi)*sin(psi)
    R[0,1] = cos(psi) * sin(phi) + cos(theta)*cos(phi)*sin(psi)
    R[0,2] = sin(psi) * sin(theta)
    R[1,0] = -sin(psi) * cos(phi) - cos(theta) * sin(phi) * cos(psi)
    R[1,1] = -sin(psi) * sin(phi) + cos(theta) * cos(phi) * cos(psi)
    R[1,2] = cos(psi) * sin(theta)
    R[2,0] = sin(theta) * sin(phi)
    R[2,1] = -sin(theta) * cos(phi)
    R[2,2] = cos(theta)
    return R

#==============================================================================
class VRSsim():
    """ This class implements hydrodynamic simulations of swimming organisms
        with characteristics specified in a Morphology object from the 
        pyVRSmorph module. 
    """
    def __init__(self,XEinit=np.asarray([0,0,0,0,0,0]),Tmax=3*25,dt_plot=0.25,
                 dt = 0.001,morph=None,flowfield=Flowfield):
        U_const_fixed = [0 0 0]
        self.tiny = 10**-7
        self.U_const_fixed = U_const_fixed
        self.S_fixed = S_fixed
        self.dt = dt
        self.dt_plot = dt_plot
        self.XEinit = XEint
        self.Tmax = Tmax
        self.morph = morph
        self.flowfield = flowfield

        self.F_buoyancy = M.pars.F_buoyancy
        self.C_buoyancy = M.pars.C_buoyancy
        self.F_gravity = M.pars.F_gravity
        self.C_gravity = M.pars.C_gravity

    def run(self):
        self.nsteps = ceil(self.Tmax/self.dt_plot)
        XE = self.XEinit
        for istep in range(self.nsteps):
	    t_prev = istep*self.dt_plot
	    t_next = min(istep*dt_plot,Tmax)
            
	    XE_old = XE
            sol = odeint(self.Rotated_CoordsVRS,XE,[t_prev t_next])
            #[t,XEbig] = ode15s('Rotated_CoordsVRS',[t_prev t_next],XE);
            XE = XEbig[-1,:]
            VEdot = self.Rotated_CoordsVRS(XE,t_next)
    
    def Rotated_CoordsVRS(self,XE,t):
        """ This function calculates the translational velocity and time rate
            of change of Euler angles for a larva immersed in flow.
            The rotation matrix and its inverse are R and Rinv. Capital 
            coordinates (e.g., X) are universal fixed, lower case is the
            coordinate system embedded in the larva.
        """
        #global VRS_morph
        #global VW
        #global U_const_fixed	#	Constant component of external velocity, fixed coordinates
        #global S_fixed	#	Constant component of external velocity, fixed coordinates
        #global U_const S

        R = R_Euler(XE[3],XE[4],XE[5])			#	The rotation of the larva relative to XYZ
        Rinv = np.linalg.inv(R)	#	The rotation of XYZ relative to the larva

        X0 = XE[0] 	#	The position of the base
        Y0 = XE[1] 
        Z0 = XE[2]
        Xbase = np.asarray([X0,Y0,Z0]).reshape([1,3])

        U_ext_fixed = self.flowfield(Xbase) #.reshape([3,1])	#	Velocity at the base in fixed coords
        U_ext = R.dot(U_ext_fixed)				#	Velocity at the base in larval coords

        #	Increments from the position of the base for estimating derivatives
        X0p = Xbase + Rinv.dot(np.asarray([self.tiny 0 0]).reshape([3,1]))
        Y0p = Xbase + Rinv.dot(np.asarray([0 self.tiny 0]).reshape([3,1]))
        Z0p = Xbase + Rinv.dot(np.asarray([0 0 self.tiny]).reshape([3,1]))
        Up1_ext_fixed = self.flowfield(X0p.T)
        Up2_ext_fixed = self.flowfield(Y0p.T)
        Up3_ext_fixed = self.flowfield(Z0p.T)
        Up1_ext = R.dot(Up1_ext_fixed)
        Up2_ext = R.dot(Up2_ext_fixed)
        Up3_ext = R.dot(Up3_ext_fixed)

        #	Estimate local shear in larval coords
        S = np.zeros([9,1])
        S[0] = (Up1_ext[0]-U_ext[0])/self.tiny
        S[3] = (Up1_ext[1]-U_ext[1])/self.tiny
        S[6] = (Up1_ext[2]-U_ext[2])/self.tiny

        S[1] = (Up2_ext[0]-U_ext[0])/self.tiny
        S[4] = (Up2_ext[1]-U_ext[1])/self.tiny
        S[7] = (Up2_ext[2]-U_ext[2])/self.tiny

        S[2] = (Up3_ext[0]-U_ext[0])/self.tiny
        S[5] = (Up3_ext[1]-U_ext[1])/self.tiny
        S[8] = (Up3_ext[2]-U_ext[2])/self.tiny

        FM_body = np.zeros([6,1]);
        FM_body[0:3] =  R.dot(self.F_buoyancy + self.F_gravity)
        FM_body[3:6] = (cross(self.C_buoyancy,R * self.F_buoyancy) + cross(self.C_gravity,R * self.F_gravity));

        #	Translational and rotational velocities in larva's coordinates (xyz)
        vw = -self.K_VW \ (self.cil_speed * [self.F_total_cilia';self.M_total_cilia'] + self.K_C * U_ext + self.K_S * S + FM_body) 
        #	Translational and rotational velocities in fixed coordinates (XYZ)
        VW = [Rinv * vw(1:3);Rinv * vw(4:6)] 
        #	Calculate rates of change in Euler angles corresponding to 
        omega1 = vw(4)
        omega2 = vw(5)
        omega3 = vw(6)
        # omega1 = VW(4)
        # omega2 = VW(5)
        # omega3 = VW(6)

        phi = XE(4)
        theta = XE(5)
        psi = XE(6)

        dphi_dt = cos(psi)/sin(theta) * omega2 + sin(psi)/sin(theta) * omega1
        dtheta_dt = cos(psi)*omega1 - sin(psi) * omega2
        dpsi_dt = -cos(theta)*cos(psi)/sin(theta) * omega2 - cos(theta)*sin(psi)/sin(theta) * omega1 + omega3


        VEdot = [VW(1:3);[dphi_dt;dtheta_dt;dpsi_dt]]

        return VEdot
