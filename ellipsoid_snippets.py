

from meshEllipsoid import chimeraEllipsoid

CE = chimeraEllipsoid(a=50.e-6,bs=[100.e-6,-40.e-6],d=6e-6,nlevels=[16,12])
CE.plot_tiles()



====================================================================

import numpy as np
from semiEllipsoid import SemiEllipsoid

SE = SemiEllipsoid(a=50.e-6,b = 100.e-6,d = 6e-6,nlevel=16)
SE.tile_quadrant()
SE.reflect_tiles()
SE.mirror_tiles(directions=['x','y'])

SE2 = SemiEllipsoid(a=50.e-6,b = ],d = 6e-6,nlevel=12)
SE2.tile_quadrant()
SE2.reflect_tiles()
SE2.mirror_tiles(directions=['x','y'])

SE.vectors = np.append(SE.vectors,SE2.vectors,axis=0)
SE.get_normals()

SE.plot_tiles()

====================================================================

import numpy as np
from semiEllipsoid import SemiEllipsoid

SE = SemiEllipsoid(a=6.,b = 10.,d = 0.6,nlevel=16)
SE.tile_quadrant()
SE.reflect_tiles()
SE.mirror_tiles(directions=['x','y'])

SE2 = SemiEllipsoid(a=6.,b = -4.,d = 0.6,nlevel=12)
SE2.tile_quadrant()
SE2.reflect_tiles()
SE2.mirror_tiles(directions=['x','y'])

SE.vectors = np.append(SE.vectors,SE2.vectors,axis=0)
SE.get_normals()

SE.plot_tiles()

====================================================================

from semiEllipsoid import SemiEllipsoid
SE = SemiEllipsoid(a=6.,b = 10.,d = 0.3)
SE.tile_quadrant()
SE.reflect_tiles()
SE.plot_tiles()

SE.mirror_tiles(directions=['x','y','z'])


for j in range(35):
    if np.any(SE.vectors[j,:,:] != tri3[j,:,:]):
        print(j)

SE.plot_tiles(axes=axes)

SE.tile_quadrant(trange=[28,30])

====================================================================
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
from math import *
from attrdict import AttrDict

plt.ion()
figure = plt.figure()
axes = figure.add_subplot(projection='3d')

Exy = lambda t_,a_,b_: (a_*cos(t_),b_*sin(t_))

ts = np.linspace(0.,pi/2,num=32)

a = 6.
b = 10.
d = 0.3
ds = sqrt(3.)/2. * d

tri3 = np.zeros([0,3,3])

#for i in range(1,ts.shape[0]-1):
for i in range(0,1): #range(1,ts.shape[0]-7):
    print('i = ',i)
    r0,z0_ = Exy(ts[i],a,b)
    r1,z1_ = Exy(ts[i+1],a,b)
    print(r0,z0_,r1,z1_)
    
    S0 = 2. * pi * r0/8.
    S1 = 2. * pi * r1/8.
    
    n0 = ceil(S0/ds)
    n1 = ceil(S1/ds)
    print('n0, n1 = ',n0,n1)
    
    s0 = (np.ones(n0).cumsum() - (n0+1.)/2.) * S0/(n0-1.)
    s1 = (np.ones(n1).cumsum() - (n1+1.)/2.) * S1/(n1-1.)
    print('s0 = ',s0)
    print('s1 = ',s1)
    
    z0 = z0_ * np.ones(s0.shape)
    z1 = z1_ * np.ones(s1.shape)
    
    row0 = np.zeros([n0,3])
    row1 = np.zeros([n1,3])
    
    row0[:,0] = r0 * np.cos(s0/(r0))
    row0[:,1] = r0 * np.sin(s0/(r0))
    row0[:,2] = z0
    row1[:,0] = r1 * np.cos(s1/(r1))
    row1[:,1] = r1 * np.sin(s1/(r1))
    row1[:,2] = z1
    
    if n0 == n1 or n0 == n1+1:
        offset = 1
    elif n0 == n1-1:
        offset = -1
    else:
        print('row length difference > 1')
    
    for i in range(n0):
        try:
            if i + offset <0:
                continue
            tri1 = np.zeros([1,3,3])
            tri1[0,0,:] = row0[i]
            tri1[0,1,:] = row1[i]
            tri1[0,2,:] = row0[i+offset]
            tri3 = np.append(tri3,tri1,axis=0)
        except:
           pass
    
    for i in range(n1):
        try:
            if i + offset <0:
                continue
            tri1 = np.zeros([1,3,3])
            tri1[0,0,:] = row1[i]
            tri1[0,1,:] = row1[i+offset]
            tri1[0,2,:] = row0[i+offset]
            tri3 = np.append(tri3,tri1,axis=0)
        except:
            pass


axes.cla()

axes.add_collection3d(mplot3d.art3d.Poly3DCollection(tri3,edgecolors='red',alpha=0.5))

scale = tri3.flatten()
axes.auto_scale_xyz(scale, scale, scale)
axes.set_aspect('equal')

axes.set_xlabel('$X$ position')
axes.set_ylabel('$Y$ position')
axes.set_zlabel('$Z$ position')












L0 = 9. #10.
L1 = 10. #9.
dy = 1.
dx = sqrt(3)/2

n0 = ceil(L0/dx)
n1 = ceil(L1/dx)

x0 = (np.ones(n0).cumsum() - (n0+1.)/2.) * L0/(n0-1)
x1 = (np.ones(n1).cumsum() - (n1+1.)/2.) * L1/(n1-1)

y0 = np.zeros(x0.shape)
y1 = dy * np.ones(x1.shape)


row0 = np.zeros([n0,3])
row1 = np.zeros([n1,3])

row0[:,0] = x0
row0[:,1] = y0
row1[:,0] = x1
row1[:,1] = y1

tri0 = np.zeros([1,3,3])
tri0[0,0,:] = row0[0]
tri0[0,1,:] = row1[0]
tri0[0,2,:] = row0[1]

tri2 = tri0.copy()

tri3 = np.zeros([0,3,3])

if n0 == n1 or n0 == n1+1:
    offset = 1
elif n0 == n1-1:
    offset = -1
else:
    print('row length difference > 1')

for i in range(n0):
    try:
        if i + offset <0:
            continue
        tri1 = np.zeros([1,3,3])
        tri1[0,0,:] = row0[i]
        tri1[0,1,:] = row1[i]
        tri1[0,2,:] = row0[i+offset]
        tri3 = np.append(tri3,tri1,axis=0)
    except:
        pass

for i in range(n1):
    try:
        if i + offset <0:
            continue
        tri1 = np.zeros([1,3,3])
        tri1[0,0,:] = row1[i]
        tri1[0,1,:] = row1[i+offset]
        tri1[0,2,:] = row0[i+offset]
        tri3 = np.append(tri3,tri1,axis=0)
    except:
        pass


axes.cla()

axes.add_collection3d(mplot3d.art3d.Poly3DCollection(tri3,edgecolors='red'))

scale = tri3.flatten()
axes.auto_scale_xyz(scale, scale, scale)
axes.set_aspect('equal')

axes.set_xlabel('$X$ position')
axes.set_ylabel('$Y$ position')
axes.set_zlabel('$Z$ position')







for i in range(n0-1):
    tri1 = np.zeros([1,3,3])
    tri1[0,0,:] = row0[i]
    tri1[0,1,:] = row1[i]
    tri1[0,2,:] = row0[i+1]
    #print(tri2.shape,tri1.shape)
    #tri2 = np.concatenate((tri2,tri1),axis=0)
    tri3 = np.append(tri3,tri1,axis=0)

for i in range(n1-1):
    tri1 = np.zeros([1,3,3])
    tri1[0,0,:] = row1[i]
    tri1[0,1,:] = row1[i+1]
    tri1[0,2,:] = row0[i+1]
    #print(tri2.shape,tri1.shape)
    #tri2 = np.concatenate((tri2,tri1),axis=0)
    tri3 = np.append(tri3,tri1,axis=0)


