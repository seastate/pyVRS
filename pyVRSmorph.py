#
#   Submodule containing class definitions and methods to create and modify
#   morphologies for Volume Rendered Swimmer hydrodynamic calculations.
#

from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
import math

from attrdict import AttrDict

from pyVRSflow import Stokeslet_shape, External_vel3, larval_V, solve_flowVRS

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
    def __init__(self,stlfile=None,mesh=None,pars={},layer_type=None,
                 material=None,density=None,immersed_in=None,**kwargs):
        """ Create a layer instance, using an AttrDict object.
        """
        super().__init__(**kwargs)
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
        #print(self.pars)
        self.pars.layer_type = layer_type
        self.pars.transformations = []
        self.mesh = mesh
        # If provided, load the specified stl file
        self.pars.stlfile = stlfile
        if self.pars.stlfile is not None:
            self.loadSTL()

    def loadSTL(self,stlfile=None,update=True):
        """ A convenience method to execute a commit
        """
        if stlfile is not None:
            self.pars.stlfile = stlfile
        self.mesh = mesh.Mesh.from_file(self.pars.stlfile)
        if update:
            self.update()

    def translate_mesh(self,translation,update=True):
        """ A convenience method to translate the current mesh
        """
        self.mesh.translate(translation)
        self.pars.transformations.append(['trans',translation])
        if update:
            self.update()

    def rotate_mesh(self,axis,theta,point=None,update=True):
        """ A convenience method to rotate the current mesh
        """
        self.mesh.rotate(axis,theta=theta,point=point)
        self.pars.transformations.append(['rot',axis,theta])
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
            #self.areas = self.mesh.areas
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
            self.pars.volume, self.pars.cog, self.pars.inertia = self.mesh.get_mass_properties()

  
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
    """ A derived class to contain an surface Layer, which additionally 
        includes singularities associated with boundary conditions and ciliary
        forces, control points on the skin, etc.

        Surface layers are always immersed in the medium, which is (pseudo)layer 0.
    """
    def __init__(self,stlfile=None,mesh=None,pars={},layer_type='surface',
                 density=1070.,material='tissue',immersed_in=0,
                 get_points=True,
                 tetra_project=0.03,tetra_project_min=0.01e-6,**kwargs):
        super().__init__(stlfile,mesh,pars,layer_type,material,density,immersed_in,**kwargs)
        #print(self.pars)
        #self.pars.density = density
        #self.pars.material = material
        #self.pars.immersed_in = immersed_in
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

#==============================================================================
class Inclusion(Layer):
    """ A derived class to contain an inclusion Layer, which displaces
        volume from the Layer in which it is immersed. It is assumed, but
        not currently verified, that the inclusion lies entirely within 
        the specified surrounding Layer. This assumption arises in calculations
        of gravity and buoyancy centers and forces.
    """
    def __init__(self,stlfile=None,mesh=None,pars={},layer_type='inclusion',
                 density=1070.,material='seawater',immersed_in=None,**kwargs):
        super().__init__(stlfile,mesh,pars,layer_type,material,density,immersed_in,**kwargs)
        #print(self.pars)
        print('Created Inclusion object with parameters:\n{}'.format(self.pars))

#==============================================================================
class Medium(Layer):
    """ A derived class to contain the properties of the medium (ambient seawater,
        typically) in the form of a pseudo-layer (which is always the 0th layer).
    """
    def __init__(self,stlfile=None,mesh=None,pars={},layer_type='medium',
                 density=1070.,material='seawater',nu = 1.17e-6,**kwargs):
        super().__init__(stlfile,mesh,pars,layer_type,material,density,**kwargs)
        #print(self.pars)
        #nu = 1.17e-6    #   Kinematic viscosity, meters^2/second
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
        super().__init__(**kwargs)
        base_densities={'seawater':1030,
                   'tissue':1070,
                   'lipid':900,
                   'calcite':2669}
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
                    
    def gen_surface(self,stlfile=None,mesh=None,pars={},
                        layer_type='surface',get_points=True,
                        material='tissue',immersed_in=0):
        """A method to facilitate generating Surface objects to iniate
           a morphology. The parameter immersed_in specifies the layer
           in which the surface is immersed, almost always the medium with
           layer index 0.
        """
        try:
            nlayers = len(self.layers)
            surface = Surface(stlfile=stlfile,mesh=mesh,pars=pars,
                              density=self.densities[material],
                              layer_type=layer_type,get_points=get_points,
                              material=material,immersed_in=immersed_in)
            self.layers.append(surface)
            # Add new layer to the layer which contains it
            self.layers[surface.pars.immersed_in].pars['contains'].append(nlayers)
            print('Added surface to layers list:')
            self.print_layer(layer_list=[-1]) #[len(self.layers)-1])
        except:
            print('Failed to load file or generate a Surface object...')
        
    def gen_inclusion(self,stlfile=None,mesh=None,pars={},
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
            inclusion = Inclusion(stlfile=stlfile,mesh=mesh,pars=pars,
                              density=self.densities[material],layer_type=layer_type,
                              material=material,immersed_in=immersed_in)
            self.layers.append(inclusion)
            # Add new layer to the layer which contains it
            print('Adding new layer to container...')
            self.layers[immersed_in].pars['contains'].append(nlayers)
            print('Added inclusion {} to layers list...'.format(len(self.layers)-1))
        except:
            print('Failed to load file to generate a Inclusion object...')

    def plot_layers(self,axes,alpha=0.5,autoscale=True):
        """A method to simplify basic 3D visualizatioin of larval morphologies.
        """
        for i,layer in enumerate(self.layers):
            print('Layer {} of type {}'.format(i,type(layer)))
            # layer type "Medium" is invisible
            if isinstance(layer,Medium):
                continue
            elif layer.pars.layer_type == 'surface':
                nfaces = layer.mesh.areas.shape[0]
                colors = np.zeros([nfaces,3])
                colors[:,0] = layer.rel_speed.flatten()
                colors[:,2] = np.ones([nfaces])-layer.rel_speed.flatten()
                scale = layer.mesh.points.flatten()
            elif layer.pars.material == 'lipid':
                colors = np.asarray([0.,1.,1.])
            elif layer.pars.material == 'calcite':
                colors = 'gray'
            elif layer.pars.material == 'seawater':
                colors = np.asarray([0.3,0.3,0.3])
            else:
                print('Unknown layer material in plot_layers; skipping layer {}'.format(i))
            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(layer.mesh.vectors,
                                                                 shade=False,
                                                                 facecolors=colors,
                                                                 alpha=alpha))
            if autoscale:
                axes.auto_scale_xyz(scale,scale,scale)


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
                #density_diff = density - density_immersed
                # Buoyancy forces are due to displacement by the surface
                layer.pars.F_buoyancy = self.g * density_immersed_in * layer.pars['volume']
                layer.pars.C_buoyancy = layer.pars['cog']
                print('F_buoyancy = ',layer.pars.F_buoyancy)
                print('C_buoyancy = ',layer.pars.C_buoyancy)
                # begin calculation of gravity forces; CoG's of included layers are weighted by mass
                layer.pars.F_gravity = -self.g * density * layer.pars['volume']
                layer.pars.C_gravity = self.g*density*layer.pars['volume'] * layer.pars['cog']
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
                    layer.pars.F_gravity -= self.g * density_diff * self.layers[i].pars['volume']
                    layer.pars.C_gravity = self.g*density_diff*layer.pars['volume'] * layer.pars['cog']
                print('F_gravity = ',layer.pars.F_gravity)
                print('C_gravity = ',layer.pars.C_gravity)
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
        #n_vert = size(VRS_morph(1).vertices,1)
        nfaces = self.layers[surface_layer].mesh.areas.shape[0] #size(VRS_morph(1).faces,1)

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
            #[U1,U2,U3] = Stokeslet_shape(VRS_morph(1).vertices_skin,VRS_morph(1).vertices_sing(iface,:),[1 1 1])
    
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
        #S = np.zeros([1,9])      # Vector of shear velocities
        #S = np.zeros([9,1])      # Vector of shear velocities
        V_L = np.asarray([0,0,0])
        Omega_L = np.asarray([0,0,0])
        #V_larva = larval_V(P_center,V_L,Omega_L)
        cil_speed = 1
        F_cilia_indirect,M_cilia_indirect = solve_flowVRS(self.layers[surface_layer],
                                                          V_L,Omega_L,
                                                          cil_speed,U_const,S)
        F_cilia = F_cilia_indirect
        M_cilia = M_cilia_indirect
        self.layers[surface_layer].F_total_cilia = np.sum(F_cilia,1)
        self.layers[surface_layer].M_total_cilia = np.sum(M_cilia,1)
        #-------------------------------------------
        # Calculate the matrix K_CF, which is the matrix of forces resulting from 
        # external constant velocities.
        V_L = np.asarray([0,0,0])
        Omega_L = np.asarray([0,0,0])
        #V_larva = larval_V(P_center,V_L,Omega_L)
        cil_speed = 0
        S = np.zeros(9)   # Vector of shear velocities
      
        U_const = np.asarray([1,0,0])
        #[F_C1,M_C1] = solve_flowVRS
        F_C1,M_C1 = solve_flowVRS(self.layers[surface_layer],
                                  V_L,Omega_L,
                                  cil_speed,U_const,S)
        self.layers[surface_layer].F_total_C1 = np.sum(F_C1,axis=0,keepdims=True) 
        self.layers[surface_layer].M_total_C1 = np.sum(M_C1,axis=0,keepdims=True) 
        
        U_const = np.asarray([0,1,0])
        #[F_C2,M_C2] = solve_flowVRS
        F_C2,M_C2 = solve_flowVRS(self.layers[surface_layer],
                                  V_L,Omega_L,cil_speed,U_const,S)
        self.layers[surface_layer].F_total_C2 = np.sum(F_C2,axis=0,keepdims=True) 
        self.layers[surface_layer].M_total_C2 = np.sum(M_C2,axis=0,keepdims=True) 
        
        U_const = np.asarray([0,0,1])
        #[F_C3,M_C3] = solve_flowVRS
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
        self.layers[surface_layer].K_C = np.concatenate((self.layers[surface_layer].K_FC,self.layers[surface_layer].K_MC),axis=0)

        #-------------------------------------------
        #	Calculate the matrix K_S, which is the matrix of forces and moments resulting from 
        #	external shear velocities.
        #	There are nine cases: du1/dx1,du1/dx2,du1/dx3,
        #                         du2/dx1,du2/dx2,du2/dx3,
        #                         du3/dx1,du3/dx2,du3/dx3
        U_const = np.zeros([1,3])
        V_L = np.asarray([0,0,0])
        Omega_L = np.asarray([0,0,0])
        cil_speed = 0
        self.layers[surface_layer].K_FS = np.zeros([3,9])	
        self.layers[surface_layer].K_MS = np.zeros([3,9])	

        #F_S = np.zeros([nfaces,3,9])
        #M_S = np.zeros([nfaces,3,9])
        ##F_S = repmat(NaN,[size(P_center,1),3,9])
        ##M_S = repmat(NaN,[size(P_center,1),3,9])

        for i_S in range(9):
            S = np.zeros([9,1])      # Vector of shear velocities
            S[i_S] = 1		#	Vector of shear velocities	
            #[F_S1,M_S1] = solve_flowVRS
            F_S1,M_S1 = solve_flowVRS(self.layers[surface_layer],
                                      V_L,Omega_L,cil_speed,U_const,S)
            #F_S[:,:,i_S] = F_S1
            #M_S[:,:,i_S] = M_S1
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
        Omega_L = np.asarray([0,0,0])
        cil_speed = 0
        S = np.zeros([9,1])      # Vector of shear velocities

        V_L = np.asarray([1,0,0])
        F_trans1,M_trans1 = solve_flowVRS(self.layers[surface_layer],
                                          V_L,Omega_L,cil_speed,U_const,S)
        self.layers[surface_layer].F_total_trans1 = np.sum(F_trans1,axis=0,keepdims=True) 
        self.layers[surface_layer].M_total_trans1 = np.sum(M_trans1,axis=0,keepdims=True) 
#	Zero external flow; unit larval translation in the y direction; no rotation; zero ciliary action
        V_L = np.asarray([0,1,0])
        F_trans2,M_trans2 = solve_flowVRS(self.layers[surface_layer],
                                          V_L,Omega_L,cil_speed,U_const,S)
        self.layers[surface_layer].F_total_trans2 = np.sum(F_trans1,axis=0,keepdims=True) 
        self.layers[surface_layer].M_total_trans2 = np.sum(M_trans1,axis=0,keepdims=True) 
#	Zero external flow; unit larval translation in the z direction; no rotation; zero ciliary action
        V_L = np.asarray([0,0,1])
        F_trans3,M_trans3 = solve_flowVRS(self.layers[surface_layer],
                                          V_L,Omega_L,cil_speed,U_const,S)
        self.layers[surface_layer].F_total_trans3 = np.sum(F_trans1,axis=0,keepdims=True) 
        self.layers[surface_layer].M_total_trans3 = np.sum(M_trans1,axis=0,keepdims=True) 
#	Zero external flow; unit larval rotation in the x direction; no translation; zero ciliary action
        V_L = np.asarray([0,0,0])
        Omega_L = np.asarray([1,0,0])
        F_rot1,M_rot1 = solve_flowVRS(self.layers[surface_layer],
                                      V_L,Omega_L,cil_speed,U_const,S)
        self.layers[surface_layer].F_total_rot1 = np.sum(F_rot1,axis=0,keepdims=True)
        self.layers[surface_layer].M_total_rot1 = np.sum(M_rot1,axis=0,keepdims=True)
#	Zero external flow; unit larval rotation in the y direction; no translation; zero ciliary action
        Omega_L = np.asarray([0,1,0])
        F_rot2,M_rot2 = solve_flowVRS(self.layers[surface_layer],
                                      V_L,Omega_L,cil_speed,U_const,S)
        self.layers[surface_layer].F_total_rot2 = np.sum(F_rot2,axis=0,keepdims=True)
        self.layers[surface_layer].M_total_rot2 = np.sum(M_rot2,axis=0,keepdims=True)
#	Zero external flow; unit larval rotation in the z direction; no translation; zero ciliary action
        Omega_L = np.asarray([0,0,1])
        F_rot3,M_rot3 = solve_flowVRS(self.layers[surface_layer],
                                      V_L,Omega_L,cil_speed,U_const,S)
        self.layers[surface_layer].F_total_rot3 = np.sum(F_rot2,axis=0,keepdims=True)
        self.layers[surface_layer].M_total_rot3 = np.sum(M_rot2,axis=0,keepdims=True)

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
        # K_VW = [[K_FV,K_MV];[K_FW,K_MW]]

        #K_FV = [F_total_trans1', F_total_trans2', F_total_trans3']
        #K_MV = [M_total_trans1', M_total_trans2', M_total_trans3']
        # 
        #K_FW = [F_total_rot1', F_total_rot2', F_total_rot3']
        #K_MW = [M_total_rot1', M_total_rot2', M_total_rot3']
        #
        #K_VW = [[K_FV,K_FW];[K_MV,K_MW]]
        ## K_VW = [[K_FV,K_MV];[K_FW,K_MW]]

#-------------------------------------------
'''
VRS_morph(1).F_total_cilia = F_total_cilia
VRS_morph(1).M_total_cilia = M_total_cilia

VRS_morph(1).F_cilia_indirect = F_cilia_indirect
VRS_morph(1).M_cilia_indirect = M_cilia_indirect

VRS_morph(1).F_C1 = F_C1
VRS_morph(1).M_C1 = M_C1

VRS_morph(1).F_C2 = F_C2
VRS_morph(1).M_C2 = M_C2

VRS_morph(1).F_C3 = F_C3
VRS_morph(1).M_C3 = M_C3

VRS_morph(1).F_trans1 = F_trans1
VRS_morph(1).M_trans1 = M_trans1

VRS_morph(1).F_trans2 = F_trans2
VRS_morph(1).M_trans2 = M_trans2

VRS_morph(1).F_trans3 = F_trans3
VRS_morph(1).M_trans3 = M_trans3

VRS_morph(1).F_rot1 = F_rot1
VRS_morph(1).M_rot1 = M_rot1

VRS_morph(1).F_rot2 = F_rot2
VRS_morph(1).M_rot2 = M_rot2

VRS_morph(1).F_rot3 = F_rot3
VRS_morph(1).M_rot3 = M_rot3


VRS_morph(1).F_S = F_S
VRS_morph(1).M_S = M_S




VRS_morph(1).K_C = K_C
VRS_morph(1).K_S = K_S
VRS_morph(1).K_VW = K_VW


#====================================================
#====================================================


VRS_morph_sav = VRS_morph
tmp = ['save ',directory,filesep,prefix,suffix,' VRS_morph_sav -mat']
eval(tmp)
'''



