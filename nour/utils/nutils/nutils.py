import nour
import numpy as np

def isNode(a):
    return isinstance(a, nour.node)

def isNumber(a):
    return type(a) == int or type(a) == float

def isList(a):
    return type(a) == list

def isTuple(a):
    return type(a) == tuple

def isNumpy(a):
    return isinstance(a, np.ndarray)

def isScalar(a):
    return a.squeeze().shape == (1,) or a.shape == (1,)

def isVector(a):
    return convert2node(a).press().ndim == 1

def convert2node(a):
    if isNode(a):
        return a
    
    elif isNumber(a) or isList(a) or isTuple(a):
        return nour.node(a, requires_grad = False)
    
    elif isNumpy(a):
        return nour.node(a, requires_grad = False, dtype = a.dtype)
        
    else:
        try:
            return nour.node(a, requires_grad = False)
        except:
            raise TypeError('Input must be type int, float, list, tuple or numpy not', type(a))

def transpose_rowcolumn_only(a):
    if a.ndim < 3:
        return np.transpose(a)
    return a.reshape(*a.shape[:-2], a.shape[-1], a.shape[-2])

def identity(n, dtype = None, requires_grad = False):
    return nour.node(np.identity(n = n, dtype=dtype), requires_grad=requires_grad)

def full(value, shape, dtype = None, requires_grad = False):
    return nour.node(np.full(shape = shape, fill_value=value, dtype=dtype), requires_grad = requires_grad)

def frombuffer(buffer, dtype=float, count=-1, offset=0, like=None, requires_grad = False):
    return nour.node(np.frombuffer(buffer, dtype=dtype, count=count, offset=offset, like=like), requires_grad = requires_grad)

def fromfunction(function, shape, dtype=float, requires_grad = False, like=None, **kwargs):
    return nour.node(np.fromfunction(function=function, shape = shape, dtype=dtype, like=like, **kwargs), requires_grad = requires_grad)

def fromiter(iter, dtype, count=-1, like=None, requires_grad = False):
    return nour.node(np.fromiter(iter = iter, dtype = dtype, count=count, like=like), requires_grad = requires_grad)


