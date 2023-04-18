

#from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from matplotlib.colors import LightSource
import numpy as np
from math import *

pyplot.ion()
figure = pyplot.figure()
axes = figure.add_subplot(projection='3d')


from pyVRSmorph import *
from pyVRSflow import *


M = Morphology()

estl = '/home/dg/VRS/pyFormex/STLfiles/beta_series_1.000000e-12_1.000000e+00_1.250000e-01_7.500000e-01_1.562500e-02_7.599206e-01_ext.stl'
istl = '/home/dg/VRS/pyFormex/STLfiles/beta_series_1.000000e-12_1.000000e+00_1.250000e-01_7.500000e-01_1.562500e-02_7.599206e-01_int.stl'

M.gen_surface(stlfile=estl)
M.gen_inclusion(stlfile=istl,material='lipid',immersed_in=1)
M.plot_layers(axes=axes)

scale = M.layers[1].mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)
M.flow_calcs(surface_layer=1)


from stl_utils import loadSTL
estl = '/home/dg/VRS/pyFormex/STLfiles/beta_series_1.000000e-12_1.000000e+00_1.250000e-01_7.500000e-01_1.562500e-02_7.599206e-01_ext.stl'
vs = loadSTL(stlfile=estl)

#================================================================

#from stl import mesh
#import numpy as np
#import math
#from pyVRSmorph import *
#from pyVRSflow import *


#================================================================

import pyVRSmorph as mrph
import pyVRSflow as flw
import numpy as np
from math import pi

S_fixed = -10.*np.asarray([0.,0.,1.,0.,0.,0.,0.,0.,0.])
U_const_fixed = 0.*np.asarray([1.,0.,0.])

M = mrph.Morphology()
estl = '/home/dg/VRS/pyFormex/STLfiles/beta_series_1.000000e-12_1.000000e+00_1.250000e-01_7.500000e-01_1.562500e-02_7.599206e-01_ext.stl'
istl = '/home/dg/VRS/pyFormex/STLfiles/beta_series_1.000000e-12_1.000000e+00_1.250000e-01_7.500000e-01_1.562500e-02_7.599206e-01_int.stl'
M.gen_surface(stlfile=estl)
M.gen_inclusion(stlfile=istl,material='lipid',immersed_in=1)
M.body_calcs()
M.flow_calcs(surface_layer=1)

sim = flw.VRSsim(morph=M)
sim.run(XEinit=0.001*np.asarray([0.,0.,0.,pi/3,-pi/4,pi]),Tmax=100.,cil_speed=0.5*1000*1e-6,U_const_fixed=U_const_fixed,S_fixed=S_fixed)



#flw.flowfield3(np.zeros([1,3]))

#================================================================

import pyVRSmorph as mrph
import pyVRSflow as flw
import numpy as np
from math import *

S_fixed = 1.*np.asarray([0.,0.,1.,0.,0.,0.,0.,0.,0.])
U_const_fixed = 0.*np.asarray([1.,0.,0.])

M = mrph.Morphology()
estl = '/home/dg/VRS/pyFormex/STLfiles/beta_series_1.000000e-12_1.000000e+00_1.250000e-01_7.500000e-01_1.562500e-02_7.599206e-01_ext.stl'
istl = '/home/dg/VRS/pyFormex/STLfiles/beta_series_1.000000e-12_1.000000e+00_1.250000e-01_7.500000e-01_1.562500e-02_7.599206e-01_int.stl'
M.gen_surface(stlfile=estl)
M.gen_inclusion(stlfile=istl,material='lipid',immersed_in=1)
M.body_calcs()
M.flow_calcs(surface_layer=1)

sim = flw.VRSsim(morph=M)
sim.run(XEinit=1.*np.asarray([0.,0.,0.,pi/3,-pi/4,pi]),Tmax=10.,cil_speed=-0.5*1000.*1.e-6,U_const_fixed=U_const_fixed,S_fixed=S_fixed)



#================================================================

import pyVRSmorph as mrph
import pyVRSflow as flw
import numpy as np
from math import *

S_fixed = 0.*np.asarray([0.,0.,1.,0.,0.,0.,0.,0.,0.])
U_const_fixed = 0.*np.asarray([1.,0.,0.])

M = mrph.Morphology()
estl = '/home/dg/VRS/pyFormex/plankter.stl'
M.gen_surface(stlfile=estl)
M.body_calcs()
M.flow_calcs(surface_layer=1)

sim = flw.VRSsim(morph=M)
sim.run(XEinit=0.001*np.asarray([0.,0.,0.,pi/3,-pi/4,pi]),Tmax=10.,cil_speed=0.e-6,U_const_fixed=U_const_fixed,S_fixed=S_fixed)


sim.run(XEinit=1.*np.asarray([0.,0.,0.,pi/3,-pi/4,pi]),Tmax=10.,cil_speed=-0.5*1000.*1.e-6,U_const_fixed=U_const_fixed,S_fixed=S_fixed)
#================================================================

import pyVRSmorph as mrph
import pyVRSflow as flw
import numpy as np
from math import *

S_fixed = 1.*np.asarray([0.,0.,1.,0.,0.,0.,0.,0.,0.])
U_const_fixed = 0.*np.asarray([1.,0.,0.])
flw.flowfield3(np.zeros([1,3]),U_const_fixed=U_const_fixed,S_fixed=S_fixed)

M = mrph.Morphology()
estl = '/home/dg/VRS/pyFormex/plankter.stl'
M.gen_surface(stlfile=estl)
M.body_calcs()
M.flow_calcs(surface_layer=1)

sim = flw.VRSsim(morph=M,XEinit=0.001*np.asarray([0.,0.,0.,pi/3,-pi/4,pi]),Tmax=100.,cil_speed=0.5e-6)
sim.run()

sim.Rotated_CoordsVRS(0.,sim.XEinit)

#================================================================


import pyVRSmorph as mrph
import pyVRSflow as flw
import numpy as np
from math import *
from matplotlib import pyplot as plt

plt.ion()
figure = plt.figure()
axes = figure.add_subplot(projection='3d')

M = mrph.Morphology()
estl = '/home/dg/VRS/pyFormex/plankter.stl'
M.gen_surface(stlfile=estl)

M.plot_layers(axes=axes,XE=np.asarray([0.,0.,0.,0.,0.,0.]))

M.layers[1].mesh.max_
M.layers[1].mesh.update_max()

#================================================================

import pyVRSmorph as mrph
estl = '/home/dg/VRS/pyFormex/plankter.stl'
L = mrph.Layer(stlfile=estl)

scount = 174711, icount = 35053


sim.fig.subplots_adjust(bottom=0.15)


def foo():
    try:
        foo.counter += 1
    except AttributeError:
        foo.counter = 1
    print(counter)


def foo():
    try:
        foo.counter += 1
    except AttributeError:
        foo.counter = 1

class Foo(object):
  # Class variable, shared by all instances of this class
  counter = 0

  def __call__(self):
    Foo.counter += 1
    print Foo.counter

# Create an object instance of class "Foo," called "foo"
foo = Foo()

# Make calls to the "__call__" method, via the object's name itself
foo() #prints 1
foo() #prints 2
foo() #prints 3


#================================================================


def get_mass_props2(mesh=None):
    normals = mesh.get_unit_normals()
    areas = mesh.areas
    total_area = areas.sum()
    centroids = mesh.centroids
    volumes = areas*(centroids*normals).sum(axis=1).reshape([areas.shape[0],1])/3
    total_volume = volumes.sum()
    tet_centroids = 0.75 * centroids
    volume_center = (tet_centroids*volumes.repeat(3,axis=1)).sum(axis=0)/total_volume
    return total_area,total_volume,volume_center




#================================================================



inclusion = Inclusion(stlfile='/home/dg/VRS/pyFormex/STLfiles/blast_test_V8.000000e-06_rh1h2_1.809432e-02_1.200000e-02_9.000000e-03_prct8.000000e-01_incl1.stl',
                              density=700,layer_type='inclusion',material='lipid',immersed_in=1)



s = Surface(stlfile='/home/dg/VRS/pyFormex/STLfiles/beta_series_1.000000e-12_2.000000e+00_1.250000e-01_1.500000e+00_3.125000e-02_7.519608e-01_int.stl')

s.translate_mesh([0.,10e-5,0])
s.rotate_mesh([0,1,0],math.pi/4)

s.get_points()
nfaces = s.mesh.areas.shape[0]

cx=s.mesh.centroids[:,0]
cy=s.mesh.centroids[:,1]
cz=s.mesh.centroids[:,2]

ux = s.unormals[:,0]
uy = s.unormals[:,1]
uz = s.unormals[:,2]

ux = s.rel_Ucilia[:,0]
uy = s.rel_Ucilia[:,1]
uz = s.rel_Ucilia[:,2]


axes.cla()

axes.quiver(cx,cy,cz,ux,uy,uz,length=1.e-5,arrow_length_ratio=0.5)

scale = s.mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

cx=s.singpts[:,0]
cy=s.singpts[:,1]
cz=s.singpts[:,2]
axes.scatter(cx,cy,cz,c='red')

dx=s.ctrlpts[:,0]
dy=s.ctrlpts[:,1]
dz=s.ctrlpts[:,2]
axes.scatter(dx,dy,dz,c='green')

colors = np.zeros([nfaces,3])
colors[:,0] = s.rel_speed.flatten()
colors[:,2] = np.ones([nfaces])-s.rel_speed.flatten()

axes.add_collection3d(mplot3d.art3d.Poly3DCollection(s.mesh.vectors,shade=False,facecolors=colors,alpha=0.5))

#axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors,shade=True,facecolors='white',edgecolors='blue',lightsource=ls,alpha=0.15))

