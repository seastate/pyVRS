#
#   Submodule containing class definitions and methods to create and modify
#   morphologies for Volume Rendered Swimmer hydrodynamic calculations.
#

from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from matplotlib.colors import LightSource
import numpy as np
import math

from attrdict import AttrDict


class Layer():
    """ A base class to facilitate creating and modifying layers (surfaces enclosing
        or excluding morphological features of a swimming organism) for
        hydrodynamic modeling. 
    """
    def __init__(self,stlfile=None,mesh=None,pars={},**kwargs):
        """ Create a layer instance, using an AttrDict object.
        """
        super().__init__(**kwargs)
        # Default base parameters
        base_pars={'density':None,
                   'immersed_in':None,
                   'scale_factor':1,
                   'offset':np.array([0,0,0]),
                   'rotate':np.array([0,0,0,0])}
        # Update with passed parameters
        self.pars=AttrDict(base_pars)
        self.pars.update(pars)
        #print(self.pars)
        self.mesh = mesh
        # If provided, load the specified stl file
        self.stlfile = stlfile
        if self.stlfile is not None:
            self.loadSTL()

    def loadSTL(self,update=True):
        """ A convenience method to execute a commit
        """
        self.mesh = mesh.Mesh.from_file(self.stlfile)
        if update:
            self.update()

    def translate_mesh(self,translation,update=True):
        """ A convenience method to translate the current mesh
        """
        self.mesh.translate(translation)
        if update:
            self.update()

    def rotate_mesh(self,axis,theta,point=None,update=True):
        """ A convenience method to rotate the current mesh
        """
        self.mesh.rotate(axis,theta=theta,point=point)
        if update:
            self.update()

    def update(self,centroids=True,areas=True,units=True,normals=True,
               unitnormals=True,minmax=True,mass_props=True):
        """ A convenience method to initialize or update mesh properties
        """
        if units:
            self.mesh.update_units()
        if areas:
            self.mesh.update_areas()
            self.areas = self.mesh.areas
        if centroids:
            self.mesh.update_centroids()
        if normals and not areas: # areas updates normals automatically
            self.mesh.update_normals()
        if unitnormals: # areas updates normals automatically
            self.unitnormals()
        if minmax:
            self.mesh.update_min()
            self.mesh.update_max()
        if mass_props:
            self.volume, self.cog, self.inertia = self.mesh.get_mass_properties()

  
    def unitnormals(self,outwards=True,ref_point=None):
        """ A method to calculate unit normals for mesh faces using
            numpy-stl methods. If outwards is True, the normals are
            checked to insure they are outwards-pointing. This method
            works only for simple shapes; it needs to be upgraded using
            interior/exterior tracking e.g. with the intersect_line_triangle
            code below.

            For centered ellipsoids, use the temperary code below. 
            TODO:
            ref_point is a point guaranteed to be outside the layer. If not
            provided, it is assigned using the builtin max_ and min_ mesh
            attributes. Intersections with faces are counted to determine 
            whether a point projected from each face along the unit normal
            is interior or exterior.
        """
        self.unormals = self.mesh.get_unit_normals()
        if outwards:
            if ref_point is None:
                #ref_point = self.mesh.max_ + 0.5*(self.mesh.max_-self.mesh.min_)
                print('Using temporary algorithm to correct unit normal directions...\n')
                ref_point = [0.,0.,0.] 
            self.pars.ref_point = ref_point
            s = np.sign(np.sum(self.mesh.centroids * self.unormals,axis=1))
            self.unormals *= s.reshape([s.shape[0],1]).repeat(3,axis=1)

class Surface(Layer):
    """ A derived class to contain the surface Layer, which additionally 
        includes singularities associated with boundary conditions and ciliary
        forces, control points on the skin, etc.
    """
    def __init__(self,stlfile=None,mesh=None,pars={},
                 density=1070.,sing=True,control=True,
                 tetra_project=0.03,tetra_project_min=0.01e-6,**kwargs):
        super().__init__(stlfile,mesh,pars,**kwargs)
        #print(self.pars)
        self.pars.density = density
        self.pars.tetra_project = tetra_project
        self.pars.tetra_project_min = tetra_project_min
        print('Created Surface object with parameters:\n{}'.format(self.pars))
        
    def get_points(self,sing=True,control=True):
        """ A method to generate control points and singularity (Stokeslet)
            locations.
        """
        self.ctrlpts = self.mesh.centroids
        scl = np.maximum(self.pars.tetra_project * np.sqrt(self.mesh.areas),
                     self.pars.tetra_project_min*np.ones(self.mesh.areas.shape)).repeat(3,axis=1)
        self.singpts = self.mesh.centroids - scl*self.unormals

        nfaces = self.mesh.areas.shape[0]
        self.normal_z_project = self.unormals[:,2]
        #print('shape1 = ',np.asarray([0.,0.,-1.]).reshape([1,3]).repeat(nfaces,axis=0).shape)
        #print('shape2 = ',self.unormals[:,2].reshape([nfaces,1]).repeat(3,axis=1).shape)
        #print('shape3 = ',self.unormals.shape)
        self.rel_Ucilia = np.asarray([0.,0.,-1.]).reshape([1,3]).repeat(nfaces,axis=0) + \
            self.unormals[:,2].reshape([nfaces,1]).repeat(3,axis=1)*self.unormals
        self.rel_speed = np.linalg.norm(self.rel_Ucilia,ord=2,axis=1,keepdims=True)


class Morphology():
    """ A class to faciliate definition and calculations with organismal morphologies, including 
        ciliated and unciliated surfaces, inclusions and internal gaps, and various material
        densities.
    """
    def __init__(self,mcon,pars={},table=None,required=[],required_unique=[],**kwargs):
        """ Create a morphology instance, using an AttrDict object.
 
        """
        super().__init__(**kwargs)
        pass








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


