#from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
from math import *

from attrdict import AttrDict

plt.ion()

Exy = lambda t_,a_,b_: (a_*cos(t_),b_*sin(t_))

class SemiEllipsoid():
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
    
            s0 = (np.ones(self.ns[i]).cumsum() - (self.ns[i]+1.)/2.) * self.Ss[i]/(self.ns[i]-1.)
            z0 = self.zs[i] * np.ones(s0.shape)
            self.row0 = np.zeros([self.ns[i],3])
            self.row0[:,0] = self.rs[i] * np.cos(s0/(self.rs[i]))
            self.row0[:,1] = self.rs[i] * np.sin(s0/(self.rs[i]))
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
                 
            s1 = (np.ones(self.ns[i+1]).cumsum() - (self.ns[i+1]+1.)/2.) * self.Ss[i+1]/(self.ns[i+1]-1.)
            z1 = self.zs[i+1] * np.ones(s1.shape)
            self.row1 = np.zeros([self.ns[i+1],3])
            self.row1[:,0] = self.rs[i+1] * np.cos(s1/(self.rs[i+1]))
            self.row1[:,1] = self.rs[i+1] * np.sin(s1/(self.rs[i+1]))
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







        ## The original kludge, for centered ellisoids only...
        #if outwards:
        #    if ref_point is None:
        #        #ref_point = self.mesh.max_ + 0.5*(self.mesh.max_-self.mesh.min_)
        #        print('Using temporary algorithm to correct unit normal directions...\n')
        #        ref_point = [0.,0.,0.] 
        #    self.pars.ref_point = ref_point
        #    s = np.sign(np.sum(self.mesh.centroids * self.unormals,axis=1))
        #    self.unormals *= s.reshape([s.shape[0],1]).repeat(3,axis=1)
