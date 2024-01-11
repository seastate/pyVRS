#
#   Shared utilities and functions for the pyVRS hydrodynamics codes
#
from numpy import ndarray
#==============================================================================
# A utility to format numbers or lists of numbers for graphical output
#def n2s_fmt(f,fmt='7.3e'):
def n2s_fmt(f,fmt='7.2e'):
    _fmt = '{:'+fmt+'}'
    if type(f)==int or type(f)==float:
        return _fmt.format(f)
    if type(f)==list or type(f)==ndarray:
        f_str = ''
        #return [_fmt.format(_f) for _f in f]
        for _f in f:
            f_str += _fmt.format(_f) + ' '
        return f_str
