#
#   Submodule containing class definitions and methods to create and modify
#   morphologies for Volume Rendered Swimmer hydrodynamic calculations.
#

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
from math import pi
#import math

from attrdict import AttrDict

#from pyVRSutils import n2s_fmt
from pyVRSflow import Stokeslet_shape, External_vel3, larval_V, solve_flowVRS, R_Euler, VRSsim
from meshSpheroid import chimeraSpheroid
import pickle
from copy import deepcopy
import os

#==============================================================================
# Set up defaults for densities of named materials
#base_densities={'freshwater':1000.,
#                'seawater':1030.,
#                'brackish':1015.,
#                'tissue':1070.,
#                'lipid':900.,
#                'calcite':2669.,
#                'other':None}

#==============================================================================
# Set up defaults for dictionary of named materials and their properties. The
# relevant properties differ depending on the layer. For example, the Medium layer
# requires a viscosity field but no color, etc. Materials such as water of
# various types require all the fields implied by various usages, e.g. viscosity
# if used as a medium and color if used as an inclusion.
# The Surface is plotted with colors reflecting ciliary velocity, so 'tissue' color
# is not used unless it is an inclusion within another inclusion.
#

# Define a library of materials to run out of the box (these can be modified or replaced when modules are invoked)
Materials = AttrDict({'freshwater':AttrDict({'material':'freshwater','density':1000.,'color':np.asarray([0.1,0.3,0.3])}),
                      'seawater':AttrDict({'material':'seawater','density':1030.,'color':np.asarray([0.3,0.3,0.3]),'mu':1030.*1.17e-6}),
                      'brackish':AttrDict({'material':'brackish','density':1015.,'color':np.asarray([0.,1.0,0.])}),
                      'tissue':AttrDict({'material':'tissue','density':1070.,'color':'purple'}),
                      'lipid':AttrDict({'material':'lipid','density':900.,'color':np.asarray([0.,1.,1.])}),
                      'calcite':AttrDict({'material':'calcite','density': 2669., 'color': 'gray'})})
                      #'other': AttrDict({'material':'other','density':None, 'color': None}),
                      #'g': 9.81,'Delta_rho':10.})

# Define a default set of scale parameters, corresponding to the nondimensional case
ScaleParams = AttrDict({'V_t':1.,'mu':1.,'Delta_rho':1.,'g':1.})
#ScaleParams = AttrDict({'V_t':1.,'mu':1030.*1.17e-6,'Delta_rho':1.,'g': 9.81})

# Define a default set of shape parameters. From the pyVRSdata mockup, use a formula to set xi below from sigma
ShapeParams = Attrdict({'alpha_s':2.,'eta_s':0.3,'alpha_i':2.,'eta_i':0.3,'sigma':0.9,'beta':1.2})
# the original defaults...
#ShapeParams = Attrdict({'alpha_s':2.,'eta_s':0.3,'alpha_i':2.,'eta_i':0.3,'xi':0.3,'beta':1.2})

# Define a default set of mesh parameters (these determine triangulation of the surface and inclusion)
MeshParams = Attrdict({'d_s':0.11285593694928399,'nlevels_s':(16,16),'d_i':0.09404661412440334,'nlevels_i':(16,16)})

# Define a method and use it to define a default set of material parameters.
# By default, these correspond to nondimensionalized excess density
def get_MatlParams(Materials=Materials,reference_material='seawater',Delta_rho=ScaleParams['Delta_rho']):
    """
       A method to generate a set of nondimensional materials parameters. By default the dimensional
       material properties are taken from the predefined Materials AttrDict, the reference material
       is seawater, and the reference excess density is taken from the predefined ScaleParams AttrDict.
    """
    matl_pars = deepcopy(Materials)
    print('Materials[reference_material].density = ',Materials[reference_material].density)
    for key in matl_pars.keys():
        print(f'Materials[{key}] = {Materials[key]}')
        matl_pars[key].density = (matl_pars[key].density-Materials[reference_material].density)/Delta_rho]
        #mat_pars[key].density /= Materials.gamma * Materials[reference_material].density
        # normalize viscosities
        if 'mu' in Materials[key].keys():
            Materials[key].mu /= Materials[reference_material].mu
        print(f'matl_pars.{key} = {matl_pars[key]}')
    return matl_pars

MatlParams = get_MatlParams()

# Define a method and use it to define a default set of chimera geometry parameters.
def get_ChimeraParams(shape_pars=ShapeParams,scale_pars=ScaleParams,mesh_pars=MeshParams,
                      geom_pars=AttrDict({})):
    """
       A method to calculate geometric parameters, as expected by the chimeraSpheroid class to generate
       a constitutive chimera.

       By default, geometric parameters are returned as an AttrDict. Alternatively, a dictionary or
       AttrDict can be passed in the geom_pars argument.
    """
    # Define some shortcuts
    V_t = scale_pars['V_t']
    mu = scale_pars['mu']
    Delta_rho = scale_pars['Delta_rho']
    g = scale_pars['g']
    beta = shape_pars['beta']
    sigma = shape_pars['sigma']
    xi = (1-gpI.eta)*(sigma - ((beta-1)/beta)**(1/3))
    # Calcuate length and time scales
    geom_pars['l'] = scale_pars['V_t']**(1./3.)
    l = geom_pars['l']
    geom_pars['tau'] = mu / (Delta_rho * g * l)
    # Create a list of AttrDicts to carry parameters for each layer. For consistency with
    # Morphology.Layers, the 0th item is the medium
    geom_pars['Layers'] = [AttrDict({})]
    # Generate the AttrDict of surface parameters, and create a shortcut
    geom_pars['Layers'].append(AttrDict({}))
    gpS = geom_pars['Layers'][1]
    gpS.alpha = shape_pars['alpha_s']
    gpS.eta = shape_pars['eta_s']
    # Calculate inclusion and surface volumes
    gpS.V = beta * V_t
    # Calculate dimensions of surface and inclusion chimeras
    gpS.D = (6.*beta/(pi*gpS.alpha))**(1./3.) * l
    gpS.L0 = gpS.alpha * gpS.D
    gpS.L2 = gpS.eta * gpS.L0
    gpS.L1 = (1.-gpS.eta) * gpS.L0
    # Add triangulation (mesh) parameters
    gpS.d = mesh_pars['d_s']
    gpS.nlevels = mesh_pars['nlevels_s']
    # Generate the AttrDict of inclusion parameters, and create a shortcut
    geom_pars['Layers'].append(AttrDict({}))
    gpI = geom_pars['Layers'][2]
    gpI.alpha = shape_pars['alpha_i']
    gpI.eta = shape_pars['eta_i']
    # Calculate inclusion and surface volumes
    gpI.V = (beta-1.) * V_t
    # Calculate dimensions of surface and inclusion chimeras
    gpI.D = (6.*beta/(pi*gpI.alpha))**(1./3.) * l
    gpI.L0 = gpI.alpha * gpI.D
    gpI.L2 = gpI.eta * gpI.L0
    gpI.L1 = (1.-gpI.eta) * gpI.L0
    # Calculate vertical offset, and package as a translation vector as expected
    # by chimeraSpheroid
    gpI.h_i = xi * gpI.L0
    gpI.translate = [0.,0.,gpI.h_i]
    # Add triangulation (mesh) parameters
    gpI.d = mesh_pars['d_i']
    gpI.nlevels = mesh_pars['nlevels_i']
    #
    return geom_pars
    
'''    
    alpha_i = shape_pars['alpha_i']
    eta_i = shape_pars['eta_i']
    # Calculate inclusion and surface volumes
    geom_pars['V_i'] = (beta-1.) * V_t
    # Calculate dimensions of surface and inclusion chimeras
    geom_pars['D_i'] = (6.*(beta-1)/(pi*alpha_i))**(1./3.) * l
    D_i = geom_pars['D_i']
    geom_pars['L0_i'] = alpha_i * D_i
    L0_i = geom_pars['L0_i']
    geom_pars['L2_i'] = eta_i * L0_i
    geom_pars['L1_i'] = (1.-eta_i) * L0_i
    # Calculate vertical offset, and package as a translation vector as expected
    # by chimeraSpheroid
    geom_pars['h_i'] = shape['xi'] * L0_s
    geom_pars['translate'] = [0.,0.,geom_pars['h_i']]
    # Add triangulation (mesh) parameters
    geom_pars['d_i'] = mesh_pars['d_i']
    geom_pars['nlevels_i'] = mesh_pars['nlevels_i']
'''
GeomParams = get_ChimeraParams()
'''
def orig_gen_MaterialsND(Materials=Materials,reference_material='seawater'):
    # A function to facilitate generating the nondimensional equivalent of a Materials
    # dictionary, in which densities are normalized by the reference density.
    MaterialsND = deepcopy(Materials)
    print('Materials[reference_material].density = ',Materials[reference_material].density)
    for key in MaterialsND.keys():
        if key in ['g','gamma']:  # skip non-Material entries; these are normalized to 1, so
            MaterialsND[key] = 1. # the nondimensional case is handled the same as the dimensional
            continue
        # normalize densities
        print('key = ',key)
        print('Materials = ',Materials[key])
        MaterialsND[key].density /= Materials.gamma * Materials[reference_material].density
        # normalize viscosities
        if 'mu' in Materials[key].keys():
            Materials[key].mu /= Materials[reference_material].mu
        print(f'MaterialsND.{key} = {MaterialsND[key]}')
    # normalize constants: (now done above)
    #MaterialsND.g = 1.
    #MaterialsND.gamma = 1.
    return MaterialsND
'''
'''
def gen_MaterialsND(Materials=Materials,reference_material='seawater',Delta_rho=None):
    # A function to facilitate generating the nondimensional equivalent of a Materials
    # dictionary, in which densities are normalized by the reference density.
    # If the characteristic density (Delta_rho) is passed, that argument's value is
    # used. Otherwise, it is taken from the Materials dictionary.
    MaterialsND = deepcopy(Materials)
    print('Materials[reference_material].density = ',Materials[reference_material].density)
    for key in MaterialsND.keys():
        if key in ['g']:  # skip non-Material entries (e.g., gravitational acceleration); these are normalized to 1, so
            MaterialsND[key] = 1. # the nondimensional case is handled the same as the dimensional
            continue
        if key == 'Delta_rho':
            if Delta_rho is not None:  # the characteristic excess density is set from the argument, if given
                MaterialsND[key] = Delta_rho 
            continue
        # normalize densities
        #print('key = ',key)
        print(f'Materials[{key}] = {Materials[key]}')
        MaterialsND[key].density = (MaterialsND[key].density-Materials[reference_material].density)/MaterialsND['Delta_rho']
        #MaterialsND[key].density /= Materials.gamma * Materials[reference_material].density
        # normalize viscosities
        if 'mu' in Materials[key].keys():
            Materials[key].mu /= Materials[reference_material].mu
        print(f'MaterialsND.{key} = {MaterialsND[key]}')
    return MaterialsND

# A dictionary with normalized (nondimensional) densities:
MaterialsND = gen_MaterialsND()
'''    
#==============================================================================
# Code to find the intersection, if there is one, of a line and a triangle
# in 3D, due to @Jochemspek,
# https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d
def intersect_line_triangle(q1,q2,p1,p2,p3):
    def signed_tetra_volume(a,b,c,d):
        return np.sign(np.dot(np.cross(b-a,c-a),d-a)/6.0)

    s1 = signed_tetra_volume(q1,p1,p2,p3)
    s2 = signed_tetra_volume(q2,p1,p2,p3)

    if s1 != s2:
        s3 = signed_tetra_volume(q1,q2,p1,p2)
        s4 = signed_tetra_volume(q1,q2,p2,p3)
        s5 = signed_tetra_volume(q1,q2,p3,p1)
        if s3 == s4 and s4 == s5:
            n = np.cross(p2-p1,p3-p1)
            #t = -np.dot(q1,n-p1) / np.dot(q1,q2-q1)
            t = np.dot(p1-q1,n) / np.dot(q2-q1,n)
            return q1 + t * (q2-q1)
    return None
#==============================================================================
class Layer():
    """ A base class to facilitate creating and modifying layers (surfaces enclosing
        or excluding morphological features of a swimming organism) for
        hydrodynamic modeling. Material is a dictionary or AttrDict, e.g. an entry 
        in the Materials AttrDict. The attributes required to be in Material vary
        across layer types: viscosity for the medium, density and color for tissue 
        and inclusions, etc.
    """
    #def __init__(self,mode='cc',stlfile=None,vectors=None,pars={},layer_type=None,
    #             Material=None,immersed_in=None,geomPars=None,
    #             check_normals=True,**kwargs):
    def __init__(self,vectors=None,layer_type=None,density=None,immersed_in=None,
                 material=None,color=None,contains=[],get_points=True): #,**kwargs):
        """ Create a layer instance, using an AttrDict object.
        """
        print(f'Creating Layer of type {layer_type}...')
        # Create a template of null parameters; these will be replaced/augmented
        # by parameters in the arguments, where they are supplied.
        self.pars=AttrDict({'layer_type' = layer_type,
                            'density':density,
                            'material':material,
                            'immersed_in':immersed_in,
                            'contains':contains,
                            'color':color,})
        # Update with passed parameters
        #self.pars.update(pars)
        self.pars.transformations = []
        self.vectors = vectors
        #self.update()

    def update(self,check_normals=False):
        """ A convenience method to initialize or update mesh properties.
            The order is determined by the structure of numpy-stl.base.py,
            in which centroids are calculated separately; areas use internally
            generated normals (which are not saved); update_normals recalculates
            (and saves) normals, areas and centroids; get_unit_normals 
            uses previously determined normals and returns a scaled copy; 
            update_units copies and scales previously determined normals and 
            saves them as units.
        """
        # Calculate normals, areas and centroids. The normals resulting
        # from this calculation are stored in mesh.normals, and may be
        # either inwards or outwards pointing (leading to erroneous
        # mass property calculations). Areas, centroids and min/max are
        # not affected by this error.
        #self.mesh.update_normals()
        #self.mesh.update_min()
        #self.mesh.update_max()
        # Calculate mins and maxes
        m = self.vectors.shape[0]
        self.min_ = self.vectors.reshape([3*m,3]).min(axis=0)
        self.max_ = self.vectors.reshape([3*m,3]).max(axis=0)
        # Get unormals, a set of normals scaled to unit length and
        # corrected to all point outwards
        self.unitnormals(check_normals=check_normals)
        # The following gives erroneous values when vertex direction is
        # not consistent, which is the case for many stls. It is based
        # directly on vertex coordinates, so correcting normals does
        # not correct these calculations.
        #self.pars.volume, self.pars.cog, self.pars.inertia = self.mesh.get_mass_properties()
        # Corrected calculations for mass properties
        self.pars.total_area = self.areas.sum()
        #m = self.areas.shape[0]
        self.volumes = self.areas*(self.centroids*self.unormals).sum(axis=1).reshape([m,1])/3
        self.pars.total_volume = self.volumes.sum()
        tet_centroids = 0.75 * self.centroids
        self.pars.volume_center = (tet_centroids*self.volumes.repeat(3,axis=1)).sum(axis=0)/self.pars.total_volume

    def unitnormals(self,outwards=True,ref_point=None,check_normals=False):
        """ A method to calculate unit normals for mesh faces.  using
            numpy-stl methods. If outwards is True, the normals are
            checked to insure they are outwards-pointing. This method
            works only for simple shapes; it needs to be upgraded using
            interior/exterior tracking e.g. with the intersect_line_triangle
            code below.

            For centered ellipsoids, use the temporary code below. 
            TODO:
            ref_point is a point guaranteed to be outside the layer. If not
            provided, it is assigned using the builtin max_ and min_ mesh
            attributes. Intersections with faces are counted to determine 
            whether a point projected from each face along the unit normal
            is interior or exterior.
        """
        self.check_normals = check_normals
        # Calculate normals and rescale to unit length (deferring direction
        # checks to the next step).
        v0 = self.vectors[:, 0]
        v1 = self.vectors[:, 1]
        v2 = self.vectors[:, 2]
        n = self.vectors.shape[0]
        self.normals = np.cross(v1 - v0, v2 - v0)
        self.areas = .5 * np.sqrt((self.normals ** 2).sum(axis=1,keepdims=True))
        self.centroids = np.mean([v0,v1,v2], axis=0)
        self.lengths = np.sqrt((self.normals**2).sum(axis=1,keepdims=True)).repeat(3,axis=1)
        self.unormals = self.normals / self.lengths
        #self.unormals = self.mesh.get_unit_normals() # unit normals, possibly misdirected
        # checking normals is time-consuming, so do it only when check_normals is True
        # (only for Surface layers).
        if self.check_normals:
            counts = self.count_intersections()
            evens = counts % 2==0
            odds = counts % 2!=0
            s = np.zeros(counts.shape)
            s[odds] = -1
            s[evens] = 1
            # correct directions for inwards pointing normals
            self.unormals *= s.reshape([s.shape[0],1]).repeat(3,axis=1)

    def count_intersections(self,ref_point=None,project=0.01e-6):
        print('Counting intersections...')
        # If not provided, choose a ref_point guaranteed to be outside shape
        if ref_point is None:
            ref_point = self.max_ + np.ones(self.max_.shape)
            #ref_point = self.mesh.max_ + np.ones(self.mesh.max_.shape)
            #ref_point2 = self.mesh.min_ - np.ones(self.mesh.max_.shape)
        test_points = self.centroids + project*self.unormals
        m = self.unormals.shape[0]
        counts = np.zeros([m,1])
        q1=ref_point
        scount = 0
        icount = 0
        for i in range(m):
            q2 = test_points[i]
            q12 = np.asarray([q1,q2])
            q12min = q12.min(axis=0)
            q12max = q12.max(axis=0)
            for j in range(m):
                # Do a simple check, to save calls to
                # intersect_line_triangle, which is slow
                vecs = self.vectors[j,:,:]
                vecs_min = vecs.min(axis=0)
                if (q12max < vecs_min).any():
                    scount += 1
                    continue
                vecs_max = vecs.max(axis=0)
                if (q12min > vecs_max).any():
                    scount += 1
                    continue
                p1 = vecs[0,:]
                p2 = vecs[1,:]
                p3 = vecs[2,:]
                ilt = intersect_line_triangle(q1,q2,p1,p2,p3)
                icount += 1
                if ilt is not None:
                    counts[i] += 1
        print('...completed.')
        print('scount = {}, icount = {}'.format(scount,icount))
        return counts


class Surface(Layer):
    """ A derived class to contain a surface Layer, which additionally 
        includes singularities associated with boundary conditions and ciliary
        forces, control points on the skin, etc.

        Surface layers are always immersed in the medium, which is (pseudo)layer 0.
    """
    #def __init__(self,stlfile=None,vectors=None,pars={},layer_type='surface',
    #             Material=Materials.tissue,immersed_in=0,geomPars=None,
    #             get_points=True,check_normals=True,
    #             tetra_project=0.03,tetra_project_min=0.01e-6,**kwargs):
    def __init__(self,vectors=None,layer_type='surface',density=None,immersed_in=0,
                 material=None,color=None,get_points=True,check_normals=False,
                 tetra_project=0.03,tetra_project_min=0.01e-6):
        super().__init__(vectors,layer_type,density,immersed_in,material,color)
        self.pars.tetra_project = tetra_project
        self.pars.tetra_project_min = tetra_project_min
        print('Calculating vector properties for new Surface object with parameters:\n{}'.format(self.pars))
        self.update(check_normals=check_normals)
        if get_points:
            print('Getting control and singularity points...')
            self.get_points()
        
    def get_points(self,sing=True,control=True,
                   tetra_project=None,tetra_project_min=None):
        """ A method to generate control points and singularity (Stokeslet)
            locations.
        """
        if tetra_project is not None:
            self.pars.tetra_project = tetra_project
        if tetra_project_min is not None:
            self.pars.tetra_project_min = tetra_project_min
        self.ctrlpts = self.centroids
        scl = np.maximum(self.pars.tetra_project * np.sqrt(self.areas),
                     self.pars.tetra_project_min*np.ones(self.areas.shape)).repeat(3,axis=1)
        self.singpts = self.centroids - scl*self.unormals

        nfaces = self.areas.shape[0]
        self.normal_z_project = self.unormals[:,2]
        self.rel_Ucilia = np.asarray([0.,0.,-1.]).reshape([1,3]).repeat(nfaces,axis=0) + \
            self.unormals[:,2].reshape([nfaces,1]).repeat(3,axis=1)*self.unormals
        self.rel_speed = np.linalg.norm(self.rel_Ucilia,ord=2,axis=1,keepdims=True)

#==============================================================================
class Inclusion(Layer):
    """ A derived class to contain an inclusion Layer, which displaces
        volume from the Layer in which it is immersed. It is assumed, but
        not currently verified, that the inclusion lies entirely within 
        the specified surrounding Layer. This assumption arises in calculations
        of gravity and buoyancy centers and forces.
    """
    def __init__(self,vectors=None,layer_type='inclusion',density=None,immersed_in=None,
                 material=None,color=None,check_normals=False):
        super().__init__(vectors,layer_type,density,immersed_in,material,color)
        print('Calculating vector properties for new Inclusion object with parameters:\n{}'.format(self.pars))
        self.update(check_normals=check_normals)

#==============================================================================
class Medium(Layer):
    """ A derived class to contain the properties of the medium (ambient seawater,
        typically) in the form of a pseudo-layer (which is always the 0th layer).
    """
    def __init__(self,layer_type='medium',density=None,material='seawater',mu=None,):
        super().__init__(layer_type,density,material)
        self.pars.mu = mu
        #self.pars.mu = Material.mu
        print('Created Medium object with parameters:\n{}'.format(self.pars))

#==============================================================================
class Morphology():
    """ A class to faciliate specifications and calculations with organismal morphologies, including 
        ciliated and unciliated surfaces, inclusions and internal gaps, and various material
        densities. Medium-, Surface- and Inclusion-type layers are created by invoking those
        respective inheritance Classes.
    """
    def __init__(self,metadata={}):
    #def __init__(self,matlPars=MatlParams,shapePars=ShapeParams,scalePars=ScaleParams,meshPars=MeshParams,
    #             medium='seawater',metadata={},**kwargs):
        """ Create a morphology instance, preserving metadata and creating a placeholder
            list to contain Layers
        """
        self.metadata = metadata
        self.Layers = []
        # Preserve passed parameters
        #self.matlPars = matlPars
        #self.scalePars = scalePars
        #self.shapePars = shapePars
        #self.meshPars = meshPars
        # Calculate geometric parameters for the surface and inclusion
        #self.GeomParams = get_ChimeraParams(shape_pars=shapePars,scalePars=scalePars,mesh_pars=meshPars)
        #self.g = Materials.g # Include as an argument for units flexibility and nondimensionalization
        # Add an attribute to store Layers. The medium (typically
        # ambient seawater) is always the 0th layer
        #self.layers = [Medium(Material=self.matlPars[medium]*self.scalePars['Delta_rho'],mu=self.scalePars['mu'])]

    def gen_surface(self,vectors=None,layer_type='surface',density=None,immersed_in=0,material=None,
                    color=None,get_points=True,check_normals=False,get_points=True,
                    tetra_project=0.03,tetra_project_min=0.01e-6):
        """A method to facilitate generating Surface objects to iniate
           a morphology. The parameter immersed_in specifies the layer
           in which the surface is immersed, almost always the medium with
           layer index 0.

           Behaviors are inherited from the Layer object. The vectors argument is
           used to define the Surface. 
        """
        try:
            nlayers = len(self.layers)
            surface = Surface(vectors=vectors,layer_type=layer_type,density=density,immersed_in=immersed_in,
                              material=material,check_normals=check_normals,get_points=get_points,
                              tetra_project=tetra_project,tetra_project_min=tetra_project_min)
            print('got here')
            self.layers.append(surface)
            # Add new layer to the layer which contains it
            self.layers[surface.pars.immersed_in].pars['contains'].append(nlayers)
            print('Added Surface to layers list:')
            self.print_layer(layer_list=[-1])
        except:
            print('Failed to load file or generate a Surface object...')
        
    def gen_inclusion(self,vectors=None,layer_type='inclusion',density=None,immersed_in=None,
                 material=None,color=None,check_normals=False):
        """A method to facilitate generating Inclusion objects within a surface
           or another inclusion. The parameter immersed_in specifies the index of 
           the layer in which the inclusion is immersed, either a Surface 
           layer of layer_type tissue or an enclosing Inclusion. Common inclusions 
           include seawater, lipid and calcite.

           Behaviors are inherited from the Layer object. The vectors argument is
           used to define the Inclusion.
        """
        if immersed_in is None:
            print('Please specify immersed_in, the index of the layer \nsurrounding this inclusion.')
        try:
            nlayers = len(self.layers)
            inclusion = Inclusion(vectors=vectors,layer_type=layer_type,density=density,immersed_in=immersed_in,
                 material=material,color=color,check_normals=check_normals)
            self.layers.append(inclusion)
            # Add new layer to the layer which contains it
            print('Adding new layer to container...')
            self.layers[immersed_in].pars['contains'].append(nlayers)
            print('Added inclusion {} to layers list...'.format(len(self.layers)-1))
        except:
            print('Failed to load file to generate a Inclusion object...')

    def print_layer(self,layer_list=[],print_pars=True):
        """A method to display a summary of layer properties.
        """
        if len(layer_list) == 0:
            layer_list = range(len(self.layers))
        for l in layer_list:
            print('Layer {} of type {}'.format(l,type(self.layers[l])))
            if print_pars:
                print(self.layers[l].pars)
                    

    def plot_layers(self,axes,alpha=0.5,autoscale=True,XE=None,f=0.75):
        """A method to simplify basic 3D visualization of larval morphologies.
        """
        xyz_min = np.asarray([None,None,None],dtype='float').reshape([3,1])
        xyz_max = np.asarray([None,None,None],dtype='float').reshape([3,1])
        for i,layer in enumerate(self.layers):
            if isinstance(layer,Medium):
                continue
            elif layer.pars.layer_type == 'surface':
                nfaces = layer.areas.shape[0]
                colors = np.zeros([nfaces,3])
                colors[:,0] = layer.rel_speed.flatten()
                colors[:,2] = np.ones([nfaces])-layer.rel_speed.flatten()
            else:
                colors = layer.pars.color
            #elif layer.pars.material == 'lipid':
            #    colors = np.asarray([0.,1.,1.])
            #elif layer.pars.material == 'calcite':
            #    colors = 'gray'
            #elif layer.pars.material == 'seawater':
            #    colors = np.asarray([0.3,0.3,0.3])
            #elif layer.pars.material == 'freshwater':
            #    colors = np.asarray([0.1,0.3,0.3])
            #elif layer.pars.material == 'brackish':
            #    colors = np.asarray([0.,1.0,0.])
            #elif layer.pars.material == 'other':
            #    colors = 'orange'
            #else:
            #    print('Unknown layer material in plot_layers; skipping layer {}'.format(i))
            vectors = layer.vectors.copy()
            for m in range(vectors.shape[0]):
                if XE is not None:
                    R = R_Euler(XE[3],XE[4],XE[5])
                    Rinv = np.linalg.inv(R)
                    vectors[m] = Rinv.dot(vectors[m].T).T
                    vectors[m] += np.repeat(XE[0:3].reshape([1,3]),3,axis=0)
                xyz_max = np.fmax(np.amax(vectors[m],axis=0).reshape([3,1]),xyz_max)
                xyz_min = np.fmin(np.amin(vectors[m],axis=0).reshape([3,1]),xyz_min)
            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(vectors,#shade=False,
                                                                 facecolors=colors,
                                                                 alpha=alpha))
        if autoscale:
            xyz_range = np.max(np.abs(xyz_max - xyz_min))
            xyz_mid = (xyz_max + xyz_min)/2
            axes.set_xlim3d(xyz_mid[0]-f*xyz_range,xyz_mid[0]+f*xyz_range)
            axes.set_ylim3d(xyz_mid[1]-f*xyz_range,xyz_mid[1]+f*xyz_range)
            axes.set_zlim3d(xyz_mid[2]-f*xyz_range,xyz_mid[2]+f*xyz_range)
            axes.set_aspect('equal')
        axes.set_xlabel('$X$ position')
        axes.set_ylabel('$Y$ position')
        axes.set_zlabel('$Z$ position')
        axes.yaxis.labelpad=10
        #axes.offsetText.set(va="top", ha="right")
        tx = axes.xaxis.get_offset_text().get_text()
        if tx=='':
            tx = '1'
        ty = axes.yaxis.get_offset_text().get_text()
        if ty=='':
            ty = '1'
        tz = axes.zaxis.get_offset_text().get_text()
        if tz=='':
            tz = '1'
        scale_txt = 'scale =  {},  {},  {}'.format(tx,ty,tz)
        try:
            axes.texts[0].remove()
        except:
            pass
        axes.text2D(0.05, 0.95, scale_txt, transform=axes.transAxes)
        axes.xaxis.offsetText.set_visible(False)
        axes.yaxis.offsetText.set_visible(False)
        axes.zaxis.offsetText.set_visible(False)

    def body_calcs(self):
        """A method to calculate body forces and moments (due to gravity and buoyancy)
           for hydrodynamic simulations. It's assumed that (i) layers of type "surface"
           are exposed to the medium; (ii) layers of type "inclusion" always occur within
           layers of types surface or inclusion; and, no layer intersects another layer.
        """
        for i,layer in enumerate(self.layers):
            print('Layer {} of type {}'.format(i,type(layer)))
            # layer type "Medium" does not have body forces
            if isinstance(layer,Medium):
                continue
            # Because only surface type layers (which displace medium layers) have
            # buoyancy, accounting is done by surfaces. The "contains" list is used
            # to sequentially calculate body forces due to inclusions.
            elif layer.pars.layer_type == 'surface':
                immersed_in = layer.pars['immersed_in']
                density = layer.pars['density']
                density_immersed_in = self.layers[immersed_in].pars['density']
                # Buoyancy forces are due to displacement by the surface
                layer.pars.F_buoyancy = self.g * density_immersed_in * layer.pars['total_volume']
                layer.pars.C_buoyancy = layer.pars['volume_center']
                print('F_buoyancy = ',layer.pars.F_buoyancy)
                print('C_buoyancy = ',layer.pars.C_buoyancy)
                # begin calculation of gravity forces; CoG's of included layers are weighted by mass
                layer.pars.F_gravity = -self.g * density * layer.pars['total_volume']
                layer.pars.C_gravity = self.g*density*layer.pars['total_volume'] * layer.pars['volume_center']
                # Get a list of all inclusions
                all_inclusions = []
                new_inclusions = list(layer.pars.contains)
                while len(new_inclusions)>0:
                    new_incl = new_inclusions.pop(0)
                    all_inclusions.append(new_incl)
                    new_inclusions.extend(list(self.layers[new_incl].pars.contains))
                print('List of all inclusions is: ',all_inclusions)
                for i in all_inclusions:
                    immersed_in = self.layers[i].pars['immersed_in']
                    density = self.layers[i].pars['density']
                    density_immersed_in = self.layers[immersed_in].pars['density']
                    density_diff = density - density_immersed_in
                    layer.pars.F_gravity -= self.g * density_diff * self.layers[i].pars['total_volume']
                    layer.pars.C_gravity += self.g * density_diff * self.layers[i].pars['total_volume'] * \
                                                                self.layers[i].pars['volume_center']
                layer.pars.C_gravity /= -layer.pars.F_gravity
                print('F_gravity = ',layer.pars.F_gravity)
                print('C_gravity = ',layer.pars.C_gravity)
                layer.pars.F_gravity_vec = np.asarray([0.,0.,layer.pars.F_gravity]).reshape([3,1])
                layer.pars.F_buoyancy_vec = np.asarray([0.,0.,layer.pars.F_buoyancy]).reshape([3,1])
            elif layer.pars.layer_type == 'inclusion':
                pass  # inclusions are accounted for in calculations for their enclosing surface
            else:
                msg = 'Unknown layer type in body_calcs in layer {}'.format(i)
                raise ValueError(msg)

    def flow_calcs(self,surface_layer=1,clear_big_arrays=True):
        """A method to calculate force and moment distributions for hydrodynamic simulations.
           surface_layer is the index of the layer to be used as the ciliated surface.
        """
        # Extract properties of ambient fluid
        immersed_in = self.layers[surface_layer].pars['immersed_in']
        mu = self.layers[immersed_in].pars['mu']
        #nu = self.layers[immersed_in].pars['nu']
        density = self.layers[immersed_in].pars['density']
        
        # Construct influence matrix
        print('Assembling influence matrix')
        nfaces = self.layers[surface_layer].areas.shape[0] #size(VRS_morph(1).faces,1)

        Q11 = np.zeros([nfaces,nfaces])
        Q12 = np.zeros([nfaces,nfaces])
        Q13 = np.zeros([nfaces,nfaces])
        Q21 = np.zeros([nfaces,nfaces])
        Q22 = np.zeros([nfaces,nfaces])
        Q23 = np.zeros([nfaces,nfaces])
        Q31 = np.zeros([nfaces,nfaces])
        Q32 = np.zeros([nfaces,nfaces])
        Q33 = np.zeros([nfaces,nfaces])

        for iface in range(nfaces):
            U1,U2,U3 = Stokeslet_shape(self.layers[surface_layer].ctrlpts,
                                       self.layers[surface_layer].singpts[iface,:].reshape([1,3]),
                                       np.ones([1,3]),mu)
            #  The function [U1,U2,U3] = Stokeslet_shape(X,P,alpha) returns the
            #  velocities at points X induced by Stokeslets at points P with
            #  strengths alpha.
            #
            #  Note that U1, U2 and U3 represent the separated contributions of
            #  alpha1, alpha2 and alpha3.
            #
            #  Also, note that alpha represents forces exerted on the fluid. The
            #  forces exerted on an immersed object are equal and opposite.
            
            #	Qij(k,n) is the influence of i-direction force at the nth sing point upon the j-direction velocity
	    #	at the kth skin point
            
            Q11[:,iface] = U1[:,0]
            Q21[:,iface] = U1[:,1]
            Q31[:,iface] = U1[:,2]
            
            Q12[:,iface] = U2[:,0]
            Q22[:,iface] = U2[:,1]
            Q32[:,iface] = U2[:,2]
            
            Q13[:,iface] = U3[:,0]
            Q23[:,iface] = U3[:,1]
            Q33[:,iface] = U3[:,2]

        #  Now, Q * F = U, where F is the set of forces at singularity points P and
        #  U is the set of velocities at control points X (in the call to
        #  Stokeslet_shape).
        self.layers[surface_layer].Q = np.concatenate((np.concatenate((Q11,Q12,Q13),axis=1),
                            np.concatenate((Q21,Q22,Q23),axis=1),
                            np.concatenate((Q31,Q32,Q33),axis=1)),
                           axis=0)
        # clean up
        del Q11, Q12, Q13, Q21, Q22, Q23, Q31, Q32, Q33
        #	Calculate its inverse...
        print('Calculating inverse...')
        #  Hence, F = Q_inv * U is the set of forces at singularity points P
        #  required to induce velocities U at control points X.
        self.layers[surface_layer].Q_inv = np.linalg.inv(self.layers[surface_layer].Q)
        print('Done calculating inverse.')

        #==========================================================================
        #==========================================================================
        # Set up zero external flow and motion; then calculate indirect forces on larva due to 
        # cilia and add it to direct forces.
        U_const = np.zeros([1,3])
        S = np.zeros(9)      # Vector of shear velocities
        V_L = np.asarray([0.,0.,0.])
        Omega_L = np.asarray([0.,0.,0.])
        cil_speed = 1.
        F_cilia_indirect,M_cilia_indirect = solve_flowVRS(self.layers[surface_layer],
                                                          V_L,Omega_L,
                                                          cil_speed,U_const,S)
        F_cilia = F_cilia_indirect
        M_cilia = M_cilia_indirect
        self.layers[surface_layer].F_total_cilia = np.sum(F_cilia,axis=0)
        self.layers[surface_layer].M_total_cilia = np.sum(M_cilia,axis=0)
        #-------------------------------------------
        # Calculate the matrix K_CF, which is the matrix of forces resulting from 
        # external constant velocities.
        V_L = np.asarray([0.,0.,0.])
        Omega_L = np.asarray([0.,0.,0.])
        cil_speed = 0.
        S = np.zeros(9)   # Vector of shear velocities
      
        U_const = np.asarray([1.,0.,0.])
        F_C1,M_C1 = solve_flowVRS(self.layers[surface_layer],
                                  V_L,Omega_L,
                                  cil_speed,U_const,S)
        self.layers[surface_layer].F_total_C1 = np.sum(F_C1,axis=0,keepdims=True) 
        self.layers[surface_layer].M_total_C1 = np.sum(M_C1,axis=0,keepdims=True) 
        
        U_const = np.asarray([0.,1.,0.])
        F_C2,M_C2 = solve_flowVRS(self.layers[surface_layer],
                                  V_L,Omega_L,cil_speed,U_const,S)
        self.layers[surface_layer].F_total_C2 = np.sum(F_C2,axis=0,keepdims=True) 
        self.layers[surface_layer].M_total_C2 = np.sum(M_C2,axis=0,keepdims=True) 
        
        U_const = np.asarray([0.,0.,1.])
        F_C3,M_C3 = solve_flowVRS(self.layers[surface_layer],
                                  V_L,Omega_L,cil_speed,U_const,S)
        self.layers[surface_layer].F_total_C3 = np.sum(F_C3,axis=0,keepdims=True) 
        self.layers[surface_layer].M_total_C3 = np.sum(M_C3,axis=0,keepdims=True) 
        
        #  This will be 3x6 when moments are added for forces
        self.layers[surface_layer].K_FC = np.concatenate((self.layers[surface_layer].F_total_C1.T,
                                                          self.layers[surface_layer].F_total_C2.T,
                                                          self.layers[surface_layer].F_total_C3.T),axis=1)	
        self.layers[surface_layer].K_MC = np.concatenate((self.layers[surface_layer].M_total_C1.T,
                                                          self.layers[surface_layer].M_total_C2.T,
                                                          self.layers[surface_layer].M_total_C3.T),axis=1)
        self.layers[surface_layer].K_C = np.concatenate((self.layers[surface_layer].K_FC,
                                                         self.layers[surface_layer].K_MC),axis=0)

        #-------------------------------------------
        #	Calculate the matrix K_S, which is the matrix of forces and moments resulting from 
        #	external shear velocities.
        #	There are nine cases: du1/dx1,du1/dx2,du1/dx3,
        #                         du2/dx1,du2/dx2,du2/dx3,
        #                         du3/dx1,du3/dx2,du3/dx3
        U_const = np.zeros([1,3])
        V_L = np.asarray([0.,0.,0.])
        Omega_L = np.asarray([0.,0.,0.])
        cil_speed = 0.
        self.layers[surface_layer].K_FS = np.zeros([3,9])	
        self.layers[surface_layer].K_MS = np.zeros([3,9])	

        for i_S in range(9):
            S = np.zeros([9,1])      # Vector of shear velocities
            S[i_S] = 1		#	Vector of shear velocities	
            F_S1,M_S1 = solve_flowVRS(self.layers[surface_layer],
                                      V_L,Omega_L,cil_speed,U_const,S)
            self.layers[surface_layer].F_total_S1 = np.sum(F_S1,axis=0)
            self.layers[surface_layer].M_total_S1 = np.sum(M_S1,axis=0)
            
            self.layers[surface_layer].K_FS[:,i_S] = self.layers[surface_layer].F_total_S1.T
            self.layers[surface_layer].K_MS[:,i_S] = self.layers[surface_layer].M_total_S1.T
            
        self.layers[surface_layer].K_S = np.concatenate((self.layers[surface_layer].K_FS,
                                                         self.layers[surface_layer].K_MS),axis=0)

        #-------------------------------------------
        #  Calculate the matrix K_FV, which is the matrix of forces resulting from translational velocities
        #  Zero external flow; unit larval translation in the x direction; no rotation; zero ciliary action
        U_const = np.zeros([1,3])
        Omega_L = np.asarray([0.,0.,0.])
        cil_speed = 0.
        S = np.zeros([9,1])      # Vector of shear velocities

        V_L = np.asarray([1.,0.,0.])
        F_trans1,M_trans1 = solve_flowVRS(self.layers[surface_layer],
                                          V_L,Omega_L,cil_speed,U_const,S)
        self.layers[surface_layer].F_total_trans1 = np.sum(F_trans1,axis=0,keepdims=True) 
        self.layers[surface_layer].M_total_trans1 = np.sum(M_trans1,axis=0,keepdims=True) 
        # Zero external flow; unit larval translation in the y direction; no rotation; zero ciliary action
        V_L = np.asarray([0.,1.,0.])
        F_trans2,M_trans2 = solve_flowVRS(self.layers[surface_layer],
                                          V_L,Omega_L,cil_speed,U_const,S)
        self.layers[surface_layer].F_total_trans2 = np.sum(F_trans2,axis=0,keepdims=True) 
        self.layers[surface_layer].M_total_trans2 = np.sum(M_trans2,axis=0,keepdims=True) 
        # Zero external flow; unit larval translation in the z direction; no rotation; zero ciliary action
        V_L = np.asarray([0.,0.,1.])
        F_trans3,M_trans3 = solve_flowVRS(self.layers[surface_layer],
                                          V_L,Omega_L,cil_speed,U_const,S)
        self.layers[surface_layer].F_total_trans3 = np.sum(F_trans3,axis=0,keepdims=True) 
        self.layers[surface_layer].M_total_trans3 = np.sum(M_trans3,axis=0,keepdims=True) 
        # Zero external flow; unit larval rotation in the x direction; no translation; zero ciliary action
        V_L = np.asarray([0.,0.,0.])
        Omega_L = np.asarray([1.,0.,0.])
        F_rot1,M_rot1 = solve_flowVRS(self.layers[surface_layer],
                                      V_L,Omega_L,cil_speed,U_const,S)
        self.layers[surface_layer].F_total_rot1 = np.sum(F_rot1,axis=0,keepdims=True)
        self.layers[surface_layer].M_total_rot1 = np.sum(M_rot1,axis=0,keepdims=True)
        # Zero external flow; unit larval rotation in the y direction; no translation; zero ciliary action
        Omega_L = np.asarray([0.,1.,0.])
        F_rot2,M_rot2 = solve_flowVRS(self.layers[surface_layer],
                                      V_L,Omega_L,cil_speed,U_const,S)
        self.layers[surface_layer].F_total_rot2 = np.sum(F_rot2,axis=0,keepdims=True)
        self.layers[surface_layer].M_total_rot2 = np.sum(M_rot2,axis=0,keepdims=True)
        # Zero external flow; unit larval rotation in the z direction; no translation; zero ciliary action
        Omega_L = np.asarray([0.,0.,1.])
        F_rot3,M_rot3 = solve_flowVRS(self.layers[surface_layer],
                                      V_L,Omega_L,cil_speed,U_const,S)
        self.layers[surface_layer].F_total_rot3 = np.sum(F_rot3,axis=0,keepdims=True)
        self.layers[surface_layer].M_total_rot3 = np.sum(M_rot3,axis=0,keepdims=True)

        self.layers[surface_layer].K_FV = np.concatenate((self.layers[surface_layer].F_total_trans1.T,
                                                          self.layers[surface_layer].F_total_trans2.T,
                                                          self.layers[surface_layer].F_total_trans3.T),
                                                         axis=1)
        self.layers[surface_layer].K_MV = np.concatenate((self.layers[surface_layer].M_total_trans1.T,
                                                          self.layers[surface_layer].M_total_trans2.T,
                                                          self.layers[surface_layer].M_total_trans3.T),
                                                         axis=1)
        
        self.layers[surface_layer].K_FW = np.concatenate((self.layers[surface_layer].F_total_rot1.T,
                                                          self.layers[surface_layer].F_total_rot2.T,
                                                          self.layers[surface_layer].F_total_rot3.T),
                                                         axis=1)
        self.layers[surface_layer].K_MW = np.concatenate((self.layers[surface_layer].M_total_rot1.T,
                                                          self.layers[surface_layer].M_total_rot2.T,
                                                          self.layers[surface_layer].M_total_rot3.T),
                                                         axis=1)

        self.layers[surface_layer].K_VW = np.concatenate((np.concatenate((self.layers[surface_layer].K_FV,
                                                                          self.layers[surface_layer].K_FW),
                                                                         axis=1),
                                                          np.concatenate((self.layers[surface_layer].K_MV,
                                                                          self.layers[surface_layer].K_MW),
                                                                         axis=1))
                                                         ,axis=0)

        if clear_big_arrays:
            self.clear_big_arrays()
        
    def clear_big_arrays(self,clearQ=True,clearQ_inv=True,surface_layer=1):
        """A method to clear big arrays, so that the saved object will be smaller
        """
        if clearQ:
            try:
                del self.layers[surface_layer].Q
                print('Q deleted')
            except:
                print('Q not found')
        if clearQ_inv:
            try:
                del self.layers[surface_layer].Q_inv
                print('Q_inv deleted')
            except:
                print('Q_inv not found')

