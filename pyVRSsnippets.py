from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from matplotlib.colors import LightSource
import numpy as np
import math

pyplot.ion()




from pyVRSmorph import *

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

figure = pyplot.figure()
axes = figure.add_subplot(projection='3d')

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

