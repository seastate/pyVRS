#from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
from math import *

plt.ion()

Exy = lambda t_,a_,b_: (a_*cos(t_),b_*sin(t_))

class chimeraEllipsoid():
    """ A class to facilitate creating and modifying meshes forming chimeras 
        of two semi-ellipsoids as approximations of microorganism forms for 
        hydrodynamic modeling.
    """
    def __init__(self,a=None,bs=[],d=None,nlevels=[16,12],levels=True,
                 translate=None,**kwargs):
        # Generate upper semiellipsoid
        SE = semiEllipsoid(a=a,b = bs[0],d = d,nlevel=nlevels[0])
        SE.tile_quadrant()   # Create tiles for 1/8 of the ellipsoid
        SE.reflect_tiles()   # Reflect and mirror to complete upper semiellipsoid
        SE.mirror_tiles(directions=['x','y'])
        SE.get_normals()
        # Generate lower semiellipsoid
        SE2 = semiEllipsoid(a=a,b = bs[1],d = d,nlevel=nlevels[1])
        SE2.tile_quadrant()   # Create tiles for 1/8 of the ellipsoid
        SE2.reflect_tiles()   # Reflect and mirror to complete upper semiellipsoid
        SE2.mirror_tiles(directions=['x','y'])
        SE2.get_normals()
        # Combine to form a complete closed shape
        self.vectors = np.append(SE.vectors,SE2.vectors,axis=0)
        # Translate in xyz, if requested
        if translate is not None:
            # trigger an error if translate does not have 3 entries
            t0 = translate[0]
            t1 = translate[1]
            t2 = translate[2]
            m = self.vectors.shape[0]
            self.vectors.reshape([3*m,3])[:,0] += t0
            self.vectors.reshape([3*m,3])[:,1] += t1
            self.vectors.reshape([3*m,3])[:,2] += t2
         
    def plot_tiles(self,cla=True,axes=None):
        if axes is None:
            figure = plt.figure()
            axes = figure.add_subplot(projection='3d')
        if cla:
            axes.cla()
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.vectors,edgecolors='blue',alpha=0.5))
        scale = self.vectors.flatten()
        axes.auto_scale_xyz(scale, scale, scale)
        axes.set_aspect('equal')
        
        axes.set_xlabel('$X$ position')
        axes.set_ylabel('$Y$ position')
        axes.set_zlabel('$Z$ position')


class semiEllipsoid():
    """ A class to facilitate creating and modifying meshes forming semi-ellipsoids
        to be combined to contruct approximations of microorganism forms for
        hydrodynamic modeling. A constraint is observed that constructed meshes
        must have 4-fold symmetry, to prevent artifacts such as twisting in 
        the swimming of modeled morphologies. Another constraint is that the
        order of vertices in faces conform to the normal formed by the cross
        product of the first two vertex pairs be outward-pointing.
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
           spacing in t, but this atribute can be modified directly during
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
            print('i = ',i)

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
                print('Added peak tile...')
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
            figure = plt.figure()
            axes = figure.add_subplot(projection='3d')
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
