#
#   Submodule containing utilities to load stl file using the numpy-stl
#   module. References to numpy-stl are sequestered in this submodule
#   to avoid problems loading meshes in other ways when it is not available.
#
#   numpy-stl provides convenient functions for e.g. mass properties.
#   However, these return incorrect answers when vector order is not
#   consistent. Therefore, the loadSTL function returns only the base
#   vectors from the named stl file, and other properties are calculated
#   separately in the pyVRSmorp classes.
#
from stl import mesh


def loadSTL(stlfile=None,translate=None):
    """ Load an stl flie as a numpy-stl mesh.
        numpy-stl creates float32 arrays for vectors and normals, 
        which are subject to errors during calculations. Here they
        are replaced by float64 equivalents.
    """
    mesh_ = mesh.Mesh.from_file(stlfile)
    # replace arrays with float64 equivalents
    vectors = mesh_.vectors.copy().astype('float64')
    if translate is not None:
        # trigger an error if translate does not have 3 entries
        t0 = translate[0]
        t1 = translate[1]
        t2 = translate[2]
        m = vectors.shape[0]
        vectors.reshape([3*m,3])[:,0] += t0
        vectors.reshape([3*m,3])[:,1] += t1
        vectors.reshape([3*m,3])[:,2] += t2
    #
    return vectors
