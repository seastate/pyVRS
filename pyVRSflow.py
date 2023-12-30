#
#   Submodule containing class definitions and methods to create and modify
#   morphologies for Volume Rendered Swimmer hydrodynamic calculations.
#

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
import numpy as np

from math import ceil, sin, cos, pi, sqrt
from scipy.integrate import odeint, solve_ivp

from pyVRSutils import n2s_fmt

#==============================================================================
def Stokeslet_shape(X,C,alpha,mu):
    """	This function returns the velocities at the points represented by the
	list of vectors X exerted by the Stokeslets at the points C with forces
	alpha. X, P and alpha are _x3 np arrays.
    """
    eps = np.finfo(float).eps # machine epsilon to prevent division by zero
    # dimensions of inputs
    nx = X.shape[0]
    nc = C.shape[0]
    # Create arrays for resultant velocities
    U1 = np.zeros([nx,3])
    U2 = np.zeros([nx,3])
    U3 = np.zeros([nx,3])
    zzz = np.zeros([nx,1])
    # Loop through evaluation points, vectorizing singularity points
    for i in range(nc):	#	Calculate velocites at each of the points in X
        x = X[:,0].reshape([nx,1]) - C[i,0]	#	Distances from each point in C
        y = X[:,1].reshape([nx,1]) - C[i,1]
        z = X[:,2].reshape([nx,1]) - C[i,2]
        alpha1 = 1/(8*np.pi*mu) * alpha[i,0]*np.ones([nx,1])
        alpha2 = 1/(8*np.pi*mu) * alpha[i,1]*np.ones([nx,1])
        alpha3 = 1/(8*np.pi*mu) * alpha[i,2]*np.ones([nx,1])
        # Calculate distances. The eps should prevent blowups when X and C are the same
        R = np.sqrt(x**2+y**2+z**2) + eps
        Rinv = np.divide(np.ones([nx,1]),R)
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
    return U1,U2,U3

#==============================================================================
def External_vel3(X,U_const,S):
    """ This function evaluates the (linearized) external flow at the points X using the
        constant velocity vector U_const and the deriviatives vector S
    """
    nx = X.shape[0]
    # Insure shape and dtype are consistent
    u_const = U_const.copy().astype('float').reshape([1,3])
    s = S.copy().astype('float').flatten()
    U_ext = np.repeat(u_const,nx,axis=0)
    U_ext[:,0] += s[0:3].dot(X.T).T
    U_ext[:,1] += s[3:6].dot(X.T).T
    U_ext[:,2] += s[6:9].dot(X.T).T
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
    v_larva = V_L.reshape([1,3]).repeat(nx,axis=0) \
              + np.cross(Omega_L.reshape([1,3]),X)
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
    #  U is the set of velocities at the control points    
    U = (V_larva - U_ext + cil_speed*surface.rel_Ucilia).T.reshape([3*nc,1])
    #  F is the set of forces ON THE FLUID necessary to induces those velocities.
    F = surface.Q_inv.dot(U)
    #  Forces on the organism are equal and opposite
    Fpt = np.ones([nc,3])
    Fpt[:,0] = -F[0:nc].flatten()
    Fpt[:,1] = -F[nc:2*nc].flatten()
    Fpt[:,2] = -F[2*nc:3*nc].flatten()
    Fpt_ = -F.reshape([nc,3])
    Fpt_T = -F.T.reshape([nc,3])
    #   Calculate moments from the positions of the singularities:
    Mpt = np.cross(surface.singpts,Fpt)
    return Fpt,Mpt

#==============================================================================
def flowfield3(X,**kwargs):
    """ This function specifies the external flowfield at the points in
        the m x 3 ndarray X. U_const_fixed and S_fixed are the external
        linearized flow parameters, in fixed (global) coordinates. In
        this function they are preserved as an attribute, initialized
        to default values when the function is called for the first time.
       Thereafter, they can be set with e..g. flowfield.S_fixed = ...
    """
    try:
        flowfield3.U_const_fixed = kwargs['U_const_fixed']
    except:
        pass
    try:
        flowfield3.S_fixed = kwargs['S_fixed']
    except:
        pass
    U_const_fixed = flowfield3.U_const_fixed
    S_fixed = flowfield3.S_fixed
    # Return without a result if called without X (e.g. to install S, U parameters)
    if X is None:
        return
    m = X.shape[0]
    U_ext = U_const_fixed.reshape([1,3]).repeat(m,axis=0) + np. concatenate(
        ((S_fixed[0]*X[:,0] + S_fixed[1]*X[:,1] + S_fixed[2]*X[:,2]).reshape([m,1]),
	 (S_fixed[3]*X[:,0] + S_fixed[4]*X[:,1] + S_fixed[5]*X[:,2]).reshape([m,1]),
	 (S_fixed[6]*X[:,0] + S_fixed[7]*X[:,1] + S_fixed[8]*X[:,2]).reshape([m,1])),axis=1)
    return U_ext

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
    def __init__(self,morph=None,surface_layer=1,fig=None,fignum=None):
        self.morph = morph
        self.surface_layer = surface_layer

        self.F_buoyancy = self.morph.layers[self.surface_layer].pars.F_buoyancy
        self.F_buoyancy_vec = self.morph.layers[self.surface_layer].pars.F_buoyancy_vec
        self.C_buoyancy = self.morph.layers[self.surface_layer].pars.C_buoyancy
        self.F_gravity = self.morph.layers[self.surface_layer].pars.F_gravity
        self.F_gravity_vec = self.morph.layers[self.surface_layer].pars.F_gravity_vec
        self.C_gravity = self.morph.layers[self.surface_layer].pars.C_gravity
        
        self.F_total_cilia = self.morph.layers[self.surface_layer].F_total_cilia
        self.M_total_cilia = self.morph.layers[self.surface_layer].M_total_cilia

        self.K_VW = self.morph.layers[self.surface_layer].K_VW
        self.K_S = self.morph.layers[self.surface_layer].K_S
        self.K_C = self.morph.layers[self.surface_layer].K_C
        
        # Set up graphics
        self.fignum = fignum
        if fig is not None:
            self.figV = fig
        else:
            print('Creating new figure...')
            self.figV = plt.figure(num=self.fignum)
        self.axes1 = self.figV.add_subplot(1,2,1,projection='3d')
        self.axes2 = self.figV.add_subplot(1,2,2,projection='3d')
        
        plt.pause(1e-3)
        
    def run(self,XEinit=None,resume=False,
                 U_const_fixed = np.asarray([0.,0.,0.]),
                 S_fixed = np.asarray([0.,0.,0.,0.,0.,0.,0.,0.,0.]),
                 cil_speed = 0.,
                 Tmax=1,dt_plot=0.25,
                 dt = 0.001,morph=None,surface_layer=1,flowfield=flowfield3):
        self.tiny = 10**-6
        self.U_const_fixed = U_const_fixed.copy()
        self.S_fixed = S_fixed.copy()
        # Call flowfield, to set flow parameters S, U
        self.flowfield = flowfield
        self.flowfield(None,U_const_fixed=self.U_const_fixed,S_fixed=self.S_fixed)
        self.cil_speed = cil_speed
        self.dt = dt
        self.dt_plot = dt_plot
        if XEinit is not None:
            self.XE = XEinit.reshape([6,])
        self.Tmax = Tmax
        self.nsteps = ceil(self.Tmax/self.dt_plot)
        self.axes1.scatter(self.XE[0],self.XE[1],self.XE[2],c='red')
        if not resume:
            self.t_prev = -self.dt_plot
        for istep in range(self.nsteps):
            self.t_prev += self.dt_plot
            t_next = min(self.t_prev+self.dt_plot,self.Tmax)
            XE_old = self.XE
            sol = solve_ivp(self.Rotated_CoordsVRS,[self.t_prev,t_next],self.XE,method='RK45',
                            first_step=1e-4,max_step=1e-2)
            self.XE = sol.y[:,-1]
            self.VEdot = self.Rotated_CoordsVRS(t_next,self.XE)
            title_str1 = 'time = {}\nposition = {}\nvelocity = {}'.format(t_next,n2s_fmt(self.XE[0:3]),n2s_fmt(self.VEdot[0:3]))
            self.axes1.set_title(title_str1)
            self.axes1.plot([XE_old[0],self.XE[0]],[XE_old[1],self.XE[1]],[XE_old[2],self.XE[2]],c='blue')
            self.axes1.set_xlabel('$X$ position')
            self.axes1.set_ylabel('$Y$ position')
            self.axes1.set_zlabel('$Z$ position')
            tx = self.axes1.xaxis.get_offset_text().get_text()
            if tx=='':
                tx = '1'
            ty = self.axes1.yaxis.get_offset_text().get_text()
            if ty=='':
                ty = '1'
            tz = self.axes1.zaxis.get_offset_text().get_text()
            if tz=='':
                tz = '1'
            scale_txt = 'scale =  {},   {},   {}'.format(tx,ty,tz)
            try:
                self.axes1.texts[0].remove()
            except:
                pass
            self.axes1.text2D(0.05, 0.95, scale_txt, transform=self.axes1.transAxes)
            
            self.axes2.cla()
            self.morph.plot_layers(axes=self.axes2,XE=self.XE)
            speed = sqrt((self.VEdot[0:3]**2).sum())
            title_str2 = '{}\n{}'.format(n2s_fmt(self.XE[3:6]),n2s_fmt(speed))
            self.axes2.set_title(title_str2)

            self.figV.canvas.draw()
            self.figV.canvas.flush_events()
            
            plt.pause(1e-3)
    
    #def Rotated_CoordsVRS(self,XE,t):
    def Rotated_CoordsVRS(self,t,XE):
        """ This function calculates the translational velocity and time rate
            of change of Euler angles for a larva immersed in flow.
            The rotation matrix and its inverse are R and Rinv. Capital 
            coordinates (e.g., X) are universal fixed, lower case is the
            coordinate system embedded in the larva.
        """
        R = R_Euler(XE[3],XE[4],XE[5])			#	The rotation of the larva relative to XYZ
        Rinv = np.linalg.inv(R)	#	The rotation of XYZ relative to the larva

        X0 = XE[0] 	#	The position of the base
        Y0 = XE[1] 
        Z0 = XE[2]
        Xbase = np.asarray([X0,Y0,Z0]).reshape([3,1])

        U_ext_fixed = self.flowfield(Xbase.T).reshape([3,1])	#	Velocity at the base in fixed coords
        U_ext = R.dot(U_ext_fixed)				#	Velocity at the base in larval coords
        #	Increments from the position of the base for estimating derivatives
        X0p = Xbase + Rinv.dot(np.asarray([self.tiny,0.,0.]).reshape([3,1]))
        Y0p = Xbase + Rinv.dot(np.asarray([0.,self.tiny,0.]).reshape([3,1]))
        Z0p = Xbase + Rinv.dot(np.asarray([0.,0.,self.tiny]).reshape([3,1]))
        Up1_ext_fixed = self.flowfield(X0p.T).reshape([3,1])
        Up2_ext_fixed = self.flowfield(Y0p.T).reshape([3,1])
        Up3_ext_fixed = self.flowfield(Z0p.T).reshape([3,1])
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
        FM_body[0:3] = R.dot(self.F_buoyancy_vec + self.F_gravity_vec)
        FM_body[3:6] = (np.cross(self.C_buoyancy.reshape([1,3]),R.dot(self.F_buoyancy_vec).T) + \
                       np.cross(self.C_gravity.reshape([1,3]),R.dot(self.F_gravity_vec).T)).T
        #	Translational and rotational velocities in larva's coordinates (xyz)
        self.vw = -np.linalg.solve(self.K_VW,self.cil_speed * np.concatenate((self.F_total_cilia.reshape([3,1]),
                                                                        self.M_total_cilia.reshape([3,1])),
                                                                        axis=0) + 
                              self.K_C.dot(U_ext) + self.K_S.dot(S) + FM_body )
        #	Translational and rotational velocities in fixed coordinates (XYZ)
        #VW = np.concatenate(Rinv.dot(vw[0:3].reshape([1,3])),Rinv.dot(vw[3:6].reshape([1,3])),axis=0)
        self.VW = np.concatenate((Rinv.dot(self.vw[0:3].reshape([3,1])),Rinv.dot(self.vw[3:6].reshape([3,1]))),axis=0)
        #	Calculate rates of change in Euler angles corresponding to 
        omega1 = self.vw[3]
        omega2 = self.vw[4]
        omega3 = self.vw[5]

        phi = XE[3]
        theta = XE[4]
        psi = XE[5]

        dphi_dt = cos(psi)/sin(theta) * omega2 + sin(psi)/sin(theta) * omega1
        dtheta_dt = cos(psi)*omega1 - sin(psi) * omega2
        dpsi_dt = -cos(theta)*cos(psi)/sin(theta) * omega2 - cos(theta)*sin(psi)/sin(theta) * omega1 + omega3
        VEdot = np.concatenate((self.VW[0:3].reshape([3,1]),np.asarray([dphi_dt,dtheta_dt,dpsi_dt]).reshape([3,1])),axis=0)

        return VEdot.reshape([6,])
