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

from pyVRSutils import n2s_fmt
from pyVRSflow import Stokeslet_shape, External_vel3, larval_V, solve_flowVRS, R_Euler

#==============================================================================
# Set up defaults for densities of named materials
base_densities={'freshwater':1000.,
                'seawater':1030.,
                'tissue':1070.,
                'lipid':900.,
                'calcite':2669.}

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
        hydrodynamic modeling. 
    """
    def __init__(self,stlfile=None,vectors=None,pars={},layer_type=None,
                 material=None,density=None,immersed_in=None,
                 check_normals=True,**kwargs):
        """ Create a layer instance, using an AttrDict object.
        """
        # Default base parameters
        base_pars={'density':density,
                   'material':material,
                   'immersed_in':immersed_in,
                   'contains':[],
                   'scale_factor':1,
                   'offset':np.array([0,0,0]),
                   'rotate':np.array([0,0,0,0])}
        # Update with passed parameters
        self.pars=AttrDict(base_pars)
        self.pars.update(pars)
        self.pars.layer_type = layer_type
        self.pars.transformations = []
        self.vectors = vectors
        self.check_normals = check_normals
        # If provided, load the specified stl file
        self.pars.stlfile = stlfile
        if vectors is not None:
            self.vectors=vectors
            self.update()
        if self.pars.stlfile is not None:
            self.loadSTL()

    def loadSTL(self,stlfile=None,update=True):
        """ A wrapper method to load an stl file as a numpy-stl mesh.
            numpy-stl creates float32 arrays for vectors and normals, 
            which are subject to errors during calculations. Here they
            are replaced by float64 equivalents.
        """
        if stlfile is not None:
            self.pars.stlfile = stlfile
        # get float64 copy of vectors in the stl file
        self.vectors = loadSTL(self.pars.stlfile)
        if update:
            self.update()

    # Disable numpy-stl based transformations until they can be
    # replaced by vector-based ones
    #def translate_mesh(self,translation,update=True):
    #    """ A convenience method to translate the current mesh
    #    """
    #    self.mesh.translate(translation)
    #    self.pars.transformations.append(['trans',translation])
    #    self.vectors = self.mesh.vectors.copy().astype('float64')
    #    if update:
    #        self.update()

    #def rotate_mesh(self,axis,theta,point=None,update=True):
    #    """ A convenience method to rotate the current mesh
    #    """
    #    self.mesh.rotate(axis,theta=theta,point=point)
    #    self.pars.transformations.append(['rot',axis,theta])
    #    self.vectors = self.mesh.vectors.copy().astype('float64')
    #    if update:
    #        self.update()

    def update(self):
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
        self.unitnormals()
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
            
    def unitnormals(self,outwards=True,ref_point=None):
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

class Surface(Layer):
    """ A derived class to contain a surface Layer, which additionally 
        includes singularities associated with boundary conditions and ciliary
        forces, control points on the skin, etc.

        Surface layers are always immersed in the medium, which is (pseudo)layer 0.
    """
    def __init__(self,stlfile=None,vectors=None,pars={},layer_type='surface',
                 density=1070.,material='tissue',immersed_in=0,
                 get_points=True,check_normals=True,
                 tetra_project=0.03,tetra_project_min=0.01e-6,**kwargs):
        super().__init__(stlfile,vectors,pars,layer_type,material,density,immersed_in,
                         check_normals,**kwargs)
        self.pars.tetra_project = tetra_project
        self.pars.tetra_project_min = tetra_project_min
        print('Created Surface object with parameters:\n{}'.format(self.pars))
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
    def __init__(self,stlfile=None,vectors=None,pars={},layer_type='inclusion',
                 density=1070.,material='seawater',immersed_in=None,
                 check_normals=True,**kwargs):
        super().__init__(stlfile,vectors,pars,layer_type,material,density,immersed_in,
                         check_normals,**kwargs)
        print('Created Inclusion object with parameters:\n{}'.format(self.pars))

#==============================================================================
class Medium(Layer):
    """ A derived class to contain the properties of the medium (ambient seawater,
        typically) in the form of a pseudo-layer (which is always the 0th layer).
    """
    def __init__(self,stlfile=None,vectors=None,pars={},layer_type='medium',
                 density=1030.,material='seawater',nu = 1.17e-6,
                 check_normals=False,**kwargs):
        super().__init__(stlfile,vectors,pars,layer_type,material,density,
                         check_normals,**kwargs)
        mu = nu * density
        self.pars.nu = nu
        self.pars.mu = mu
        print('Created Medium object with parameters:\n{}'.format(self.pars))

#==============================================================================
class Morphology():
    """ A class to faciliate specifications and calculations with organismal morphologies, including 
        ciliated and unciliated surfaces, inclusions and internal gaps, and various material
        densities.
    """
    def __init__(self,densities={},g=9.81,**kwargs):
        """ Create a morphology instance, using an AttrDict object.
 
        """
        #super().__init__(**kwargs) # currently is not a derived class
        base_densities={'freshwater':1000.,
                    'seawater':1030.,
                    'tissue':1070.,
                    'lipid':900.,
                    'calcite':2669.}
        # Update with passed parameters
        self.densities=AttrDict(base_densities)
        self.densities.update(densities)
        self.g = g # Include as an argument for units flexibility
        # Add an attribute to store Layers. The medium (typically
        # ambient seawater) is always the 0th layer
        self.layers = [Medium(density=self.densities['seawater'])]

    def print_layer(self,layer_list=[],print_pars=True):
        """A method to display a summary of layer properties.
        """
        if len(layer_list) == 0:
            layer_list = range(len(self.layers))
        for l in layer_list:
            print('Layer {} of type {}'.format(l,type(self.layers[l])))
            if print_pars:
                print(self.layers[l].pars)
                    
    def gen_surface(self,stlfile=None,vectors=None,pars={},
                        layer_type='surface',get_points=True,
                        material='tissue',immersed_in=0):
        """A method to facilitate generating Surface objects to iniate
           a morphology. The parameter immersed_in specifies the layer
           in which the surface is immersed, almost always the medium with
           layer index 0.
        """
        try:
            nlayers = len(self.layers)
            surface = Surface(stlfile=stlfile,vectors=vectors,pars=pars,
                              density=self.densities[material],
                              layer_type=layer_type,get_points=get_points,
                              material=material,immersed_in=immersed_in,
                              check_normals=self.check_normals)
            self.layers.append(surface)
            # Add new layer to the layer which contains it
            self.layers[surface.pars.immersed_in].pars['contains'].append(nlayers)
            print('Added surface to layers list:')
            self.print_layer(layer_list=[-1]) #[len(self.layers)-1])
        except:
            print('Failed to load file or generate a Surface object...')
        
    def gen_inclusion(self,stlfile=None,vectors=None,pars={},
                        layer_type='inclusion',
                        material='seawater',immersed_in=None):
        """A method to facilitate generating Inclusion objects within a surface
           or another inclusion. The parameter immersed_in specifies the index of 
           the layer in which the inclusion is immersed, almost always a surface 
           layer of layer_type tissue. Common inclusions include seawater, lipid 
           and calcite.
        """
        if immersed_in is None:
            print('Please specify immersed_in, the index of the layer \nsurrounding this inclusion.')
        try:
            nlayers = len(self.layers)
            inclusion = Inclusion(stlfile=stlfile,vectors=vectors,pars=pars,
                              density=self.densities[material],layer_type=layer_type,
                              material=material,immersed_in=immersed_in,
                              check_normals=self.check_normals)
            self.layers.append(inclusion)
            # Add new layer to the layer which contains it
            print('Adding new layer to container...')
            self.layers[immersed_in].pars['contains'].append(nlayers)
            print('Added inclusion {} to layers list...'.format(len(self.layers)-1))
        except:
            print('Failed to load file to generate a Inclusion object...')

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
            elif layer.pars.material == 'lipid':
                colors = np.asarray([0.,1.,1.])
            elif layer.pars.material == 'calcite':
                colors = 'gray'
            elif layer.pars.material == 'seawater':
                colors = np.asarray([0.3,0.3,0.3])
            elif layer.pars.material == 'freshwater':
                colors = np.asarray([0.1,0.3,0.3])
            else:
                print('Unknown layer material in plot_layers; skipping layer {}'.format(i))
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
                    layer.pars.C_gravity += self.g*density_diff*self.layers[i].pars['total_volume'] * \
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

    def flow_calcs(self,surface_layer=1):
        """A method to calculate force and moment distributions for hydrodynamic simulations.
           surface_layer is the index of the layer to be used as the ciliated surface.
        """
        # Extract properties of ambient fluid
        immersed_in = self.layers[surface_layer].pars['immersed_in']
        mu = self.layers[immersed_in].pars['mu']
        nu = self.layers[immersed_in].pars['nu']
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
                                                          self.layers[surface_layer].F_total_trans3.T),axis=1)
        self.layers[surface_layer].K_MV = np.concatenate((self.layers[surface_layer].M_total_trans1.T,
                                                          self.layers[surface_layer].M_total_trans2.T,
                                                          self.layers[surface_layer].M_total_trans3.T),axis=1)
        
        self.layers[surface_layer].K_FW = np.concatenate((self.layers[surface_layer].F_total_rot1.T,
                                                          self.layers[surface_layer].F_total_rot2.T,
                                                          self.layers[surface_layer].F_total_rot3.T),axis=1)
        self.layers[surface_layer].K_MW = np.concatenate((self.layers[surface_layer].M_total_rot1.T,
                                                          self.layers[surface_layer].M_total_rot2.T,
                                                          self.layers[surface_layer].M_total_rot3.T),axis=1)

        self.layers[surface_layer].K_VW = np.concatenate((np.concatenate((self.layers[surface_layer].K_FV,
                                                                          self.layers[surface_layer].K_FW),axis=1),
                                                          np.concatenate((self.layers[surface_layer].K_MV,
                                                                          self.layers[surface_layer].K_MW),axis=1))
                                                         ,axis=0)

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


#==============================================================================
class MorphologyND(Morphology):
    """ A derived class to faciliate specifications and calculations with nondimensionalized
        organismal morphologies. This class is structurally equivalent to a dimensional
        morphology. The purpose of this classes are: (a) to clarify which of a large set
        of morphologies are nondimensional, using the object type as a filter; and (b) to
        streamline conversion of dimensional scenarios to nondimensional ones, and vice
        versa.
    """
    def __init__(self,densities={},g=9.81,**kwargs):
        """ Create a morphology instance, using an AttrDict object.
 
        """
        super().__init__(densities,g,**kwargs)
        base_densities={'freshwater':1000.,
                    'seawater':1030.,
                    'tissue':1070.,
                    'lipid':900.,
                    'calcite':2669.}
        # Update with passed parameters
        self.densities=AttrDict(base_densities)
        self.densities.update(densities)
        # nondimensionalize densities by medium (seawater) density
        reference_density = self.densities['seawater']
        for mtrl,dens in self.densities.items():
            self.densities.update({mtrl:dens/reference_density})
        print('Updated densities: ',self.densities)
        self.g = 1 # scaled to 1 in nondimensionalization
        # Add an attribute to store Layers. The medium (typically
        # ambient seawater) is always the 0th layer
        self.layers = [Medium(density=self.densities['seawater'],nu = 1)]

#==============================================================================
class MorphPars():
    """ A class to faciliate specifications and calculations with parameter sets 
        for early embryo morphologies. There are three distinct forms of these 
        parameter sets:
            1. dimensional geometrical & simulation parameters
            2. nondimensional geometrical & simulation parameters
            3. Shape and scaling parameters
        Together, shape and scaling parameters define either dimensional or non-
        dimensional geometrical & simulation parameters. Likewise, dimensional
        geometrical & simulation parameters define a set of shape and scaling
        parameters. Methods are provided to facilitate conversion of one form to
        the others. 

        Parameters are stored in AttrDict objects (enhanced dictionaries):
            - geom_pars contains geometrical information, as required by meshSpheroid,
              and additional information needed to calculate Morphology objects.
            - sim_pars contains simulation parameters (shears, ciliary velocity, t_end)
              as required by VRSsim
            - ___ND versions are nondimensional versions of the above.
            - shape_pars contains shape parameters (alpha, beta, eta, xi)
            - scale_pars contains parameters (tissue volume, viscosity, density, etc.)
              needed to reconstruct a dimensional scenario from the shape parameters

        SI units are used throughout.
    """
    def __init__(self,densities={},densitiesND={},**kwargs):
        """ Create AttrDict objects to contain parameters of different types:
        """
        print('Creating new MorphPars objects...')
        # Create AttrDict objects for the dimensional and nondimensional scenarios
        self.geom_pars = AttrDict()
        self.sim_pars = AttrDict()
        self.geom_parsND = AttrDict()
        self.sim_parsND = AttrDict()
        # shape_pars contains shape parameters 
        self.shape_pars = AttrDict()
        self.scale_pars = AttrDict()
        
    def new_sim_dim(self,dudz=0.,dvdz=0.,dwdx=0.,Tmax=10.,cil_speed=500e-6,
                    phi=pi/3,theta=pi/4,psi=pi,x0=0,y0=0,z0=0,pars={}):
        """ A method to parameterize a dimensional embryo swimming simulation.

            A default parameter set is provided, as a functional example. Alternative parameters
            can be provided directly as arguments or as elements of the sim_pars dictionary argument.
            The pars dictionary argument is considered to be the standard method, and in
            case of duplication an entry in the pars dictionary supercedes the direct argument.
        """
        print('Creating or replacing geo_pars dictionary...')
        new_pars = {'dudz':dudz,'dvdz':dvdz,'dwdx':dwdx,'Tmax':Tmax,
                    'cil_speed':cil_speed,'phi':phi,'theta':theta,'psi':psi,
                     'x0':x0,'y0':y0,'z0':z0}
        self.sim_pars.update(new_pars)
        self.sim_pars.update(pars)
        return self.sim_pars
        
    def calc_sim_nondim(self,pars={}):
        """ A method to calculate nondimensional embryo swimming simulation parameters
            from the corresponding dimensional parameters in sim_pars, and characteristic
            length and time scales in geom_pars.

            The nondimensional parameters are stored in sim_parsND.
        """
        print('Creating or replacing the sim_parsND dictionary...')
        self.sim_pars.update(pars)  # incorporate any updates
        # Create a shortcut
        sm = self.sim_pars
        print('sm = ',sm)
        p = self.geom_pars
        smn = self.sim_parsND
        #
        smn.cil_speed = p.tau_rot/p.l * sm.cil_speed
        smn.dudz = p.tau_rot * sm.dudz
        smn.dwdx = p.tau_rot * sm.dwdx
        smn.dvdz = p.tau_rot * sm.dvdz
        smn.Tmax = sm.Tmax/p.tau_rot
        #
        smn.theta = sm.theta
        smn.phi = sm.phi
        smn.psi = sm.psi
        smn.x0 = sm.x0/p.l
        smn.y0 = sm.y0/p.l
        smn.z0 = sm.z0/p.l
        
    def calc_sim_dim(self,pars={}):
        """ A method to calculate dimensional embryo swimming simulation parameters
            from the corresponding nondimensional parameters in sim_parsND, and characteristic
            length and time scales in geom_pars.

            The nondimensional parameters are stored in sim_parsND.
        """
        print('Creating or replacing the sim_pars dictionary...')
        self.sim_pars.update(pars)  # incorporate any updates
        # Create a shortcut
        sm = self.sim_pars
        p = self.geom_pars
        smn = self.sim_parsND
        #
        sm.cil_speed = p.l/p.tau_rot * smn.cil_speed
        sm.dudz = 1/p.tau_rot * smn.dudz
        sm.dwdx = 1/p.tau_rot * smn.dwdx
        sm.dvdz = 1/p.tau_rot * smn.dvdz
        sm.Tmax = p.tau_rot * smn.Tmax
        #
        sm.theta = smn.theta
        sm.phi = smn.phi
        sm.psi = smn.psi
        sm.x0 = p.l * smn.x0
        sm.y0 = p.l * smn.y0
        sm.z0 = p.l * smn.z0
        
    def new_geom_dim(self,D_s=50e-6,L1_s=100e-6,L2_s=40e-6,D_i=30e-6,L1_i=50e-6,L2_i=20e-6,
                     h_i=40e-6,d_s=6e-6,nlevels_s=[16,12],d_i=5e-6,nlevels_i=[12,8],
                     pars={},calc=True,g=9.81,mu=1.17e-6 * 1030.,
                     rho_med=base_densities['seawater'],gamma=0.1,
                     rho_t=base_densities['tissue'],
                     rho_i=base_densities['freshwater']):
        """ A method to parameterize a dimensional embryo geometry.

            A default parameter set is provided, as a functional example. Alternative parameters
            can be provided directly as arguments or as elements of the geom_pars dictionary argument.
            The pars dictionary argument is considered to be the standard method, and in
            case of duplication an entry in the pars dictionary supercedes the direct argument.

            g is the gravitational constant, provided as a way to use non-SI units (but don't).
        """
        print('Creating or replacing geo_pars dictionary...')
        new_pars = {'D_s':D_s,'L1_s':L1_s,'L2_s':L2_s,'D_i':D_i,'L1_i':L1_i,'L2_i':L2_i,'h_i':h_i, \
                    'd_s':d_s,'nlevels_s':nlevels_s,'d_i':d_i,'nlevels_i':nlevels_i, \
                    'mu':mu,'rho_med':rho_med,'gamma':gamma,'rho_t':rho_t,'rho_i':rho_i,'g':g}
        self.geom_pars.update(new_pars)
        self.geom_pars.update(pars)
        
        # If directed (and by default), calculate corresponding nondimensional parameters
        if calc:
            self.calc_geom_dim()

    def calc_geom_dim(self):
        """ A method to calculate the geometrical properties derived from the current dimensional
            geom_pars parameter set.

            This should be re-executed after each change of the basic geometrical parameters, to
            update the derived geom_pars dictionary entries.
        """
        print('Calculating derived geometrical parameters from geom_pars...')
        # make a shortcut
        p = self.geom_pars
        # Calculate total lengths
        p.L0_s = p.L1_s + p.L2_s
        p.L0_i = p.L1_i + p.L2_i
        # Calculate volumes
        p.V_s = pi/6 * p.L0_s * p.D_s**2
        p.V_i = pi/6 * p.L0_i * p.D_i**2
        p.V_t = p.V_s - p.V_i
        # Calculate length and time scales
        p.l = p.V_t**(1/3)
        p.tau_rot = (36*pi)**(1/3) * p.mu / (p.gamma * p.rho_med * p.g * p.V_t**(1/3))
        # save scales as attributes, because they are common to many calculations
        self.l = p.l
        self.tau_rot = p.tau_rot
        # return the new AttrDict, in case it needs to be used directly
        return self.geom_pars

    def new_shape2geom(self):
        """ A method to calculate dimensional geometry parameters from the current shape
            and scale parameters.

            Shape parameters are stored in the shape_pars dictionary. Scale parameters
            are stored in the scale_pars dictionary. The new geometrical parameters are
            stored in the pars_geom dictionary.
        """
        print('Calculating geom_pars from shape_pars and scale_pars...')
        # Create shortcuts
        sh = self.shape_pars
        sc = self.scale_pars
        p = self.geom_pars
        # Calculate geometry from the shape and scale parameters
        print(sc.g)
        print(p)
        p.g = sc.g
        p.mu = sc.mu
        p.rho_med = sc.rho_med
        p.gamma = sc.gamma
        #
        p.rho_t = sc.rho_med * sh.rho_t
        p.rho_i = sc.rho_med * sh.rho_i
        p.V_t = sc.V_t
        p.V_s = sh.beta * sc.V_t
        p.V_i = (sh.beta-1) * sc.V_t
        p.l = p.V_t**(1/3)
        p.tau_rot = (36*pi)**(1/3) * p.mu / (p.rho_med * p.g * p.V_t**(1/3))
        #
        p.D_s = (6*sh.beta/(pi*sh.alpha_s))**(1/3) * p.l
        p.D_i = (6*(sh.beta-1)/(pi*sh.alpha_i))**(1/3) * p.l
        p.L0_s = sh.alpha_s * p.D_s
        p.L2_s = sh.eta_s * p.L0_s
        p.L1_s = (1-sh.eta_s) * p.L0_s
        p.L0_i = sh.alpha_i * p.D_i
        p.L2_i = sh.eta_i * p.L0_i
        p.L1_i = (1-sh.eta_i) * p.L0_i
        p.h_i = sh.xi * p.L0_s
        # Pass the tiling parameters
        p.d_s = sh.d_s
        p.nlevels_s = sh.nlevels_s
        p.d_i = sh.d_i
        p.nlevels_i = sh.nlevels_i
        # return the new AttrDict, in case it needs to be used directly
        return self.geom_pars

    def new_geom2scale(self,pars={}):
        """ A method to calculate scale parameters from a set of dimensional geometry parameters.
            Scale parameters are stored in the scale_pars dictionary.

            Together, the scale_pars and shape_pars dictionaries contain the information required
            to reconstruct a dimensional embryo swimming scenario.

            By default, parameters are taken from the geom_pars dictionary. The pars dictionary 
            can be used to update or augment values in the scale_pars dictionary.
        """
        print('Calculating scale_pars from geom_pars...')
        # Create shortcuts
        p = self.geom_pars
        sc = self.scale_pars
        # Calculate shape indices
        sc.V_t= p.V_t
        sc.mu = p.mu
        sc.rho_med = p.rho_med
        sc.gamma = p.gamma
        sc.g = p.g
        sc.update(pars)
        # return the new AttrDict, in case it needs to be used directly
        return self.scale_pars

    def calc_geom2shape(self):
        """ A method to calculate shape parameters from a set of dimensional geometry parameters.

            In addition to the proper shape parameters (alpha, eta, beta, xi), the nondimensional
            tissue and inclusion densities are calculated, because they are required for the flow
            precalculations in Morphology methods. Currently, the tiling parameters d_ and nlevels
            are also carried with the shape parameters.
        """
        print('Calculating shape_pars from geom_pars...')
        # Create shortcuts
        p = self.geom_pars
        sh = self.shape_pars
        # Calculate shape indices
        sh.alpha_s = p.L0_s/p.D_s
        sh.eta_s = p.L2_s/p.L0_s
        sh.alpha_i = p.L0_i/p.D_i
        sh.eta_i = p.L2_i/p.L0_i
        sh.xi = p.h_i/p.L0_s
        sh.beta = p.V_s/p.V_t
        # Calculate nondimensional densities
        sh.rho_t = p.rho_t/p.rho_med
        sh.rho_i = p.rho_i/p.rho_med
        # Store the tiling parameters, because they are needed for reconstructing geom_pars
        sh.d_s = p.d_s
        sh.nlevels_s = p.nlevels_s
        sh.d_i = p.d_i
        sh.nlevels_i = p.nlevels_i
        # return the new AttrDict, in case it needs to be used directly
        return self.shape_pars

    def calc_geom_nondim(self):
        """ A method to calculate the nondimensional parameter set from 
            a dimensional embryo swimming scenario, as defined by the current 
            parameter sets in the geom_pars, shape_pars and scale_pars dictionaries.
        """
        # Calculate shape indices
        pn = self.geom_parsND
        p = self.geom_pars
        sh = self.shape_pars
        sc = self.scale_pars
        #
        pn.gamma = p.gamma
        pn.rho_med = 1.
        pn.rho_t = p.rho_t/p.rho_med
        pn.rho_i = p.rho_i/p.rho_med
        #
        pn.V_t = 1.
        pn.V_s = sh.beta
        pn.V_i = 1 - sh.beta
        #
        pn.D_t = (6/pi)**(1/3)
        pn.D_s = (6*sh.beta/pi)**(1/3)
        pn.L0_s = sh.alpha_s * pn.D_s
        pn.L2_s = sh.eta_s * pn.L0_s
        pn.L1_s = (1-sh.eta_s) * pn.L0_s
        pn.D_i = (6*(sh.beta-1)/pi)**(1/3)
        pn.L0_i = sh.alpha_i * pn.D_i
        pn.L2_i = sh.eta_i * pn.L0_i
        pn.L1_i = (1-sh.eta_i) * pn.L0_i
        pn.h_i = sh.xi * pn.L0_s
        #
        return pn
