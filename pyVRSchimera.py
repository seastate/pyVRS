#
#   Submodule containing class definitions and methods to create and perform
#   calculations with "constitutive chimera" morphologies for Volume Rendered
#   Swimmer hydrodynamic calculations.
#
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import numpy as np
from math import pi, sqrt, cos, sin, ceil
from attrdict import AttrDict
from copy import deepcopy, copy
import os

#from pyVRSutils import n2s_fmt
#from pyVRSflow import Stokeslet_shape, External_vel3, larval_V, solve_flowVRS, R_Euler, VRSsim
#from meshSpheroid import chimeraSpheroid
from pyVRSmorph import Morphology
#from pyVRSmorph import Layer, Surface, Inclusion, Medium


# Define a default set of scale parameters, corresponding to the nondimensional case
#ScaleParams = AttrDict({'V_t':1.,'mu':1.,'Delta_rho':1.,'g':1.})
#ScaleParams = AttrDict({'V_t':1.,'mu':1030.*1.17e-6,'Delta_rho':1.,'g': 9.81})
# Define as a function that returns an AttrDict, to avoid unintentional binding between parameters sets
def ScaleParams(V_t=1.,mu=1.,Delta_rho=1.,g=1.):
    return AttrDict({'V_t':V_t,'mu':mu,'Delta_rho':Delta_rho,'g':g})

# Define a default set of shape parameters. From the pyVRSdata mockup, use a formula to set xi below from sigma
def ShapeParams(alpha_s=2.,eta_s=0.3,alpha_i=2.,eta_i=0.3,sigma=0.9,beta=1.2):
    return AttrDict({'alpha_s':alpha_s,'eta_s':eta_s,'alpha_i':alpha_i,'eta_i':eta_i,'sigma':sigma,'beta':beta})
# the original defaults...
#ShapeParams = Attrdict({'alpha_s':2.,'eta_s':0.3,'alpha_i':2.,'eta_i':0.3,'xi':0.3,'beta':1.2})

# Define a default set of mesh parameters (these determine triangulation of the surface and inclusion)
def MeshParams(d_s=0.11285593694928399,nlevels_s=(16,16),d_i=0.09404661412440334,nlevels_i=(16,16)):
    return AttrDict({'d_s':d_s,'nlevels_s':nlevels_s,'d_i':d_i,'nlevels_i':nlevels_i})



def chimeraParams(shape_pars=None,scale_pars=None,
                  mesh_pars=None,densities=[None,None,None],
                  colors=[None,None,None],
                  chimera_pars=AttrDict({})):
    """
       A function to calculate geometric and  material parameters, as expected by the chimeraSpheroid 
       class to generate a constitutive chimera. 

       When scale parameters are all equal to 1, the result is the nondimensional case. When scale 
       parameters are dimensional, the result is in physical units.
    
       Currently, parameters are expected for three layers, in order:
           Layer 0: Medium
           Layer 1: Surface
           Layer 2: Inclusion
       A future version may enable multiple Inclusions, or other variations.

       By default, chimera parameters are returned as an AttrDict. Alternatively, a dictionary or
       AttrDict can be passed in the chimera_pars argument.

       NOTE: Parameters returned in the Surface and Inclusion fields of Layers (i.e., items 1 and 2)
             are a superset of the geometric parameters to be passed to chimeraSpheroid, so their
             interpretation and format are set by that class.

       To avoid ambiguity, parameters provided as arguments are not duplicated in the returned
       parameter set, except when necessary (e.g., mu) to implement simulations.
    """
    # Define some shortcuts
    V_t = scale_pars['V_t']
    Delta_rho = scale_pars['Delta_rho']
    g = scale_pars['g']
    beta = shape_pars['beta']
    mu = scale_pars['mu'] 
    # Calcuate length and time scales.
    l = V_t**(1./3.)
    tau = mu / (Delta_rho * g * l)
    # Retain these in the returned parameters dictionary because they will be needed
    # to parameterize simulations.
    chimera_pars['l'] = l
    chimera_pars['tau'] = tau
    chimera_pars['g'] = g
    # Create a list of AttrDicts to carry parameters for the three layers
    chimera_pars['Layers'] = [AttrDict({}),AttrDict({}),AttrDict({})]
    # Consistent with Morphology, the 0th Layer is the medium
    ccM = chimera_pars['Layers'][0]  # make a shortcut
    ccM.layer_type = 'medium'
    ccM.color = colors[0]
    ccM.density = Delta_rho * densities[0]
    ccM.mu = mu
    #----------------------------------------------------------
    # Create a shortcut for the AttrDict of surface parameters
    ccS = chimera_pars['Layers'][1]
    ccS.layer_type = 'surface'
    ccS.color = colors[1]
    # parse parameters
    ccS.density = Delta_rho * densities[1]
    alpha_s = shape_pars['alpha_s']
    eta_s = shape_pars['eta_s']
    # Calculate surface volume
    ccS.V_s = beta * V_t
    # Calculate dimensions of surface chimera
    ccS.D = (6.*beta/(pi*alpha_s))**(1./3.) * l
    ccS.L0 = alpha_s * ccS.D
    ccS.L2 = eta_s * ccS.L0
    ccS.L1 = (1.-eta_s) * ccS.L0
    ccS.translate = None
    # Add triangulation (mesh) parameters  TODO: check for scaling of d and possibly nlevels
    ccS.d = mesh_pars['d_s']
    ccS.nlevels = mesh_pars['nlevels_s']
    #----------------------------------------------------------
    # Create a shortcut for the AttrDict of inclusion parameters
    ccI = chimera_pars['Layers'][2]
    ccI.layer_type = 'inclusion'
    ccI.color = colors[2]
    # parse parameters
    ccI.density = Delta_rho * densities[2]
    alpha_i = shape_pars['alpha_i']
    eta_i = shape_pars['eta_i']
    sigma = shape_pars['sigma']
    xi = (1-eta_i)*(sigma - ((beta-1)/beta)**(1./3.))
    # Calculate inclusion volume
    ccI.V_i = (beta-1.) * V_t
    # Calculate dimensions of inclusion chimera
    ccI.D = (6.*(beta-1.)/(pi*alpha_i))**(1./3.) * l
    ccI.L0 = alpha_i * ccI.D
    ccI.L2 = eta_i * ccI.L0
    ccI.L1 = (1.-eta_i) * ccI.L0
    # Calculate vertical offset, and package as a translation vector as expected
    # by chimeraSpheroid
    ccI.h_i = xi * ccI.L0
    ccI.translate = [0.,0.,ccI.h_i]
    # Add triangulation (mesh) parameters  TODO: check for scaling of d and possibly nlevels
    ccI.d = mesh_pars['d_i']
    ccI.nlevels = mesh_pars['nlevels_i']
    #
    return chimera_pars

#                      densities=[None,None,None],colors=[None,'purple',np.asarray([0.,1.,1.])],

def chimeraMorphology(M=Morphology(),chimera_params=None,shape_pars=None,scale_pars=None,mesh_pars=None,
                      plotMorph=True,calcFlow=True,calcBody=True):
    """
       A function to facilitate generating a Morphology in the form of a constitutive
       chimera using (if provided) a set of precalculated chimera parameters, or 
       otherwise the supplied shape, scale, density, color and mesh parameters.
    
       If a Morphology object is not provided, it is generated.
    """
    if chimera_params != None:
        chimera_parameters = chimera_params
    else:
        # Calculate dimensional chimera parameters
        chimera_parameters = chimeraParams(shape_pars=shape_pars,scale_pars=scale_pars,
                                           mesh_pars=mesh_pars,densities=densities,colors=colors)
    # Attach chimera parameters to the Morphology metadata, for reference and error checking
    M.metadata["chimera_parameters"] = chimera_parameters
    # Attach gravitational acceleration to the Morphology, for body force calculations
    M.g = chimera_params.g
    # create shortcuts
    cp = chimera_parameters
    cpM = cp.Layers[0]  # the medium layer parameters
    cpS = cp.Layers[1]  # the surface layer parameters
    cpI = cp.Layers[2]  # the inclusion layer parameters
    # scaling parameters
    l = cp.l
    tau = cp.tau
    # Set up the Medium layer in the morphology M
    M. gen_medium(density=cpM.density,mu=cpM.mu)
    # Set up Surface layer
    print(M.layers[0].pars)
    surface_chimera = chimeraSpheroid(geomPars=cpS)
    M.gen_surface(vectors=surface_chimera.vectors,density=cpS.density,immersed_in=0,
                    color=cpS.color,get_points=True,check_normals=False,
                    tetra_project=0.03,tetra_project_min=0.01e-6)
    print(M.layers[0].pars)
    print(M.layers[1].pars)
    # If V_i > 0, set up the Inclusion layer
    if cpI.V_i > 0.:
        inclusion_chimera = chimeraSpheroid(geomPars=cpI)
        M.gen_inclusion(vectors=inclusion_chimera.vectors,density=cpI.density,
                        immersed_in=1,color=cpI.color)
        print(M.layers[0].pars)
        print(M.layers[1].pars)
        print(M.layers[2].pars)

    # If requested, plot morphology
    if plotMorph:
        figureM = plt.figure(num=67)
        figureM.clf()
        axesM = figureM.add_subplot(projection='3d')
        M.plot_layers(axes=axesM)
        figureM.canvas.draw()
        figureM.canvas.flush_events()
        plt.pause(0.25)
        if calcBody:  # if requested, calculate buoyancy & gravity forces
            M.body_calcs()
        if calcFlow:  # if requested, calculate flow around the surface
            M.flow_calcs(surface_layer=1)
    # Return the Morphology object
    return M

Exy = lambda t_,a_,b_: (a_*cos(t_),b_*sin(t_))

class chimeraSpheroid():
    """ A class to facilitate creating and modifying meshes forming chimeras 
        of two semi-spheroids as approximations of microorganism forms for 
        hydrodynamic modeling. Parameters are contained in geom_pars, an
        argument that is either an AttrDict or dictionary instance.

        The resulting chimera has diameter D, with the upper semi-spheroid of 
        height L1 and the lower of height L2.

        Tiling for the chimera is produced separately for the upper and lower
        semispheroids, using the semiSpheroid class methods. Note that this 
        class defined the horizontal section with radius rather than diameter,
        and defines the lower semispheroid with a negative height.
    """
    def __init__(self,geomPars=None,levels=True,tile=True):
        # Add tiling parameters as characteristics
        print('chimeraSpheroid: defining with geomPars')
        self.geomPars = geomPars
        #print(self.geomPars)
        self.levels = levels
        # If requested (true by default), calculate tiles
        if tile:
            print('chimeraSpheroid: calculating tiles...')
            self.get_tiles()
        
    def get_tiles(self):
        # Generate tiles for the upper semispheroid, using radius a = D/2, and 
        # the upper height parameter b=L1 projecting upwards
        print('tiling upper semispheroid...')
        SE = semiSpheroid(a=self.geomPars.D/2,b = self.geomPars.L1,
                          d = self.geomPars.d,nlevel=self.geomPars.nlevels[0],levels=self.levels)
        SE.tile_quadrant()   # Create tiles for 1/8 of the spheroid
        SE.reflect_tiles()   # Reflect and mirror to complete upper semispheroid
        SE.mirror_tiles(directions=['x','y'])
        SE.get_normals()
        # Generate tiles for the lower semispheroid, using radius a = D/2, and
        # the lower height parameter b=-L2 projecting downwards (preserves exact
        # symmetries with upper semispheroid)
        print('tiling lower semispheroid...')
        SE2 = semiSpheroid(a=self.geomPars.D/2,b = -self.geomPars.L2,
                           d = self.geomPars.d,nlevel=self.geomPars.nlevels[1],levels=self.levels)
        SE2.tile_quadrant()   # Create tiles for 1/8 of the spheroid
        SE2.reflect_tiles()   # Reflect and mirror to complete upper semispheroid
        SE2.mirror_tiles(directions=['x','y'])
        SE2.get_normals()
        # Combine to form a complete closed shape
        self.vectors = np.append(SE.vectors,SE2.vectors,axis=0)
        # Translate in xyz, if requested
        if self.geomPars.translate is not None:
            # trigger an error if translate does not have 3 entries
            t0 = self.geomPars.translate[0]
            t1 = self.geomPars.translate[1]
            t2 = self.geomPars.translate[2]
            m = self.vectors.shape[0]
            self.vectors.reshape([3*m,3])[:,0] += t0
            self.vectors.reshape([3*m,3])[:,1] += t1
            self.vectors.reshape([3*m,3])[:,2] += t2
         
    def plot_tiles(self,cla=True,axes=None):
        if axes is None:
            figureC = plt.figure()
            axes = figureC.add_subplot(projection='3d')
        if cla:
            axes.cla()
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.vectors,edgecolors='blue',alpha=0.5))
        scale = self.vectors.flatten()
        axes.auto_scale_xyz(scale, scale, scale)
        axes.set_aspect('equal')
        
        axes.set_xlabel('$X$ position')
        axes.set_ylabel('$Y$ position')
        axes.set_zlabel('$Z$ position')


class semiSpheroid():
    """ A class to facilitate creating and modifying meshes forming semi-spheroids
        to be combined to contruct approximations of microorganism forms for
        hydrodynamic modeling. A constraint is observed that constructed meshes
        must have 4-fold symmetry, to prevent artifacts such as twisting in 
        the swimming of modeled morphologies. Another constraint is that the
        order of vertices in faces conform to the normal formed by the cross
        product of the first two vertex pairs be outward-pointing.

        a: radius of intersection with the xy plane
        b: height of semispheroid above the xy plane
            b>0 implies the semispheroid projects in the positive z-direction
            b<0 implies the semispheroid projects in the negative z-direction
    """
    def __init__(self,a=None,b=None,d=None,nlevel=32,levels=True,**kwargs):

        self.a = a
        self.b = b
        self.d = d
        self.ds = sqrt(3.)/2. * self.d
        self.nlevel = nlevel

        self.rows = []
        self.zs = np.zeros(nlevel)
        self.rs = np.zeros(nlevel)
        self.ns =np.zeros(nlevel).astype('int')
        self.Ss = np.zeros(nlevel)
        self.vectors = np.zeros([0,3,3])
        self.faces = []

        if levels:
            self.set_levels()
    
    def set_levels(self,nlevel=None):
        """Calculate a set of angles, t, measured in radians upwards from
           the xy plane. Tiles are to be distributed as t (translated into 
           the vertical axis, z). Currently the only option is uniform 
           spacing in t, but this attribute can be modified directly during
           mesh construction.
        """
        if nlevel is not None:
            self.nlevel = nlevel
        self.ts = np.linspace(0.,pi/2,num=self.nlevel)

        for i in range(0,self.nlevel):
            self.rs[i],self.zs[i] = Exy(self.ts[i],self.a,self.b)
            self.Ss[i] = 2. * pi * self.rs[i]/8.
            self.ns[i] = ceil(self.Ss[i]/self.ds)

        self.peak = np.asarray([0.,0.,self.b])

    def tile_quadrant(self,clear=True,trange=None):
        if clear:
            self.vectors = np.zeros([0,3,3])

        #for i in range(1,ts.shape[0]-1):
        #for i in range(1,self.nlevel-7):
        if trange is None:
            w0 = 0
            w1 = self.nlevel-1
        else:
            w0 = trange[0]
            w1 = trange[1]
        for i in range(w0,w1):
            #print('i = ',i)

            angle_offset = pi/8.
            #s0 = np.ones(self.ns[i]).cumsum() * self.Ss[i]/(self.ns[i]-1.)
            s0 = (np.ones(self.ns[i]).cumsum() - (self.ns[i]+1.)/2.) * self.Ss[i]/(self.ns[i]-1.)
            z0 = self.zs[i] * np.ones(s0.shape)
            self.row0 = np.zeros([self.ns[i],3])
            self.row0[:,0] = self.rs[i] * np.cos(angle_offset+s0/(self.rs[i]))
            self.row0[:,1] = self.rs[i] * np.sin(angle_offset+s0/(self.rs[i]))
            self.row0[:,2] = z0
            

            if self.ns[i+1] == 1:
                # reached the last row that can be propagated;
                # create a final tile to peak to close the shape.
                if self.ns[i] != 2:
                    print('Warning: Abrupt tile transition at peak! Errors are likely!!')
                tri1 = np.zeros([1,3,3])
                tri1[0,0,:] = self.row0[0]
                tri1[0,1,:] = self.peak
                tri1[0,2,:] = self.row0[1]
                self.vectors = np.append(self.vectors,tri1,axis=0)
                #print('Added peak tile...')
                break
                 
            #s1 = np.ones(self.ns[i+1]).cumsum() * self.Ss[i+1]/(self.ns[i+1]-1.)
            s1 = (np.ones(self.ns[i+1]).cumsum() - (self.ns[i+1]+1.)/2.) * self.Ss[i+1]/(self.ns[i+1]-1.)
            z1 = self.zs[i+1] * np.ones(s1.shape)
            self.row1 = np.zeros([self.ns[i+1],3])
            self.row1[:,0] = self.rs[i+1] * np.cos(angle_offset+s1/(self.rs[i+1]))
            self.row1[:,1] = self.rs[i+1] * np.sin(angle_offset+s1/(self.rs[i+1]))
            self.row1[:,2] = z1
    

            if self.ns[i] == self.ns[i+1] or self.ns[i] == self.ns[i+1]+1:
                offset = 1
            elif self.ns[i] == self.ns[i+1]-1:
                offset = -1
            else:
                print('row length difference > 1')

            for ii in range(self.ns[i]):
                try:
                    if ii + offset <0:
                        continue
                    tri1 = np.zeros([1,3,3])
                    tri1[0,0,:] = self.row0[ii]
                    tri1[0,1,:] = self.row1[ii]
                    tri1[0,2,:] = self.row0[ii+offset]
                    self.vectors = np.append(self.vectors,tri1,axis=0)
                except:
                    pass

            for ii in range(self.ns[i+1]):
                try:
                    if ii + offset <0:
                        continue
                    tri1 = np.zeros([1,3,3])
                    tri1[0,0,:] = self.row1[ii]
                    tri1[0,1,:] = self.row1[ii+offset]
                    tri1[0,2,:] = self.row0[ii+offset]
                    self.vectors = np.append(self.vectors,tri1,axis=0)
                    #print('B success: ',i)
                except:
                    pass

    def mirror_tiles(self,concat=True,directions=[]):
        for d in directions:
            self.vectors2 = self.vectors.copy()
            if d in ['x','X']:
                for i,v in enumerate(self.vectors2):
                    self.vectors2[i,:,0] *= -1
                if concat:
                    self.vectors = np.append(self.vectors,self.vectors2,axis=0)
            if d in ['y','Y']:
                for i,v in enumerate(self.vectors2):
                    self.vectors2[i,:,1] *= -1
                if concat:
                    self.vectors = np.append(self.vectors,self.vectors2,axis=0)
            if d in ['z','Z']:
                for i,v in enumerate(self.vectors2):
                    self.vectors2[i,:,2] *= -1
                if concat:
                    self.vectors = np.append(self.vectors,self.vectors2,axis=0)

    def reflect_tiles(self,concat=True):
        self.vectors2 = self.vectors.copy()
        for i,v in enumerate(self.vectors2):
            self.vectors2[i,:,0:2] = np.flip(self.vectors2[i,:,0:2],axis=1)
        if concat:
            self.vectors = np.append(self.vectors,self.vectors2,axis=0)

    def plot_tiles(self,cla=True,axes=None):
        if axes is None:
            figureE = plt.figure()
            axes = figureE.add_subplot(projection='3d')
        if cla:
            axes.cla()
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.vectors,edgecolors='blue',alpha=0.5))
        scale = self.vectors.flatten()
        axes.auto_scale_xyz(scale, scale, scale)
        axes.set_aspect('equal')
        
        axes.set_xlabel('$X$ position')
        axes.set_ylabel('$Y$ position')
        axes.set_zlabel('$Z$ position')

    def get_normals(self,check_normals=True,ref_point=None,outwards=True,correct=True):
        v0 = self.vectors[:, 0]
        v1 = self.vectors[:, 1]
        v2 = self.vectors[:, 2]
        n = self.vectors.shape[0]
        self.normals = np.cross(v1 - v0, v2 - v0)
        self.areas = .5 * np.sqrt((self.normals ** 2).sum(axis=1,keepdims=True))
        self.centroids = np.mean([v0,v1,v2], axis=0)
        self.lengths = np.sqrt((self.normals**2).sum(axis=1,keepdims=True)).repeat(3,axis=1)
        self.unormals = self.normals / self.lengths
        # checking normals is time-consuming, so do it only when check_normals is True
        # (only for Surface layers).
        if check_normals:
            #counts = self.count_intersections()
            #evens = counts % 2==0
            #odds = counts % 2!=0
            #s = np.zeros(counts.shape)
            #s[odds] = -1
            #s[evens] = 1
            # correct directions for inwards pointing normals
            #self.unormals *= s.reshape([s.shape[0],1]).repeat(3,axis=1)
            # The original kludge, for centered ellisoids only...
            #if outwards:
            if ref_point is None:
                ref_point = [0.,0.,0.] 
            self.ref_point = ref_point
            self.s = np.sign(np.sum(self.centroids * self.unormals,axis=1))
            self.unormals *= self.s.reshape([self.s.shape[0],1]).repeat(3,axis=1)
            if correct:
                for i in range(self.vectors.shape[0]):
                    if self.s[i] == -1:
                        self.vectors[i,1:3,:] = np.flip(self.vectors[i,1:3,:],axis=0)





        ## The original kludge, for centered ellisoids only...
        #if outwards:
        #    if ref_point is None:
        #        #ref_point = self.mesh.max_ + 0.5*(self.mesh.max_-self.mesh.min_)
        #        print('Using temporary algorithm to correct unit normal directions...\n')
        #        ref_point = [0.,0.,0.] 
        #    self.pars.ref_point = ref_point
        #    s = np.sign(np.sum(self.mesh.centroids * self.unormals,axis=1))
        #    self.unormals *= s.reshape([s.shape[0],1]).repeat(3,axis=1)
