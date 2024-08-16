import nour
from ._nrandom import Gen
import math
import numpy as np

def rand(*dims, requires_grad = False, seed = None, **kwargs):
    if seed:
        Gen.set_seed(seed)

    return nour.node(Gen.random_raw(size = dims) / 2 ** 64, dtype=float, requires_grad=requires_grad)

def randint(low, high, shape = (1,), requires_grad = False, seed = None):
    if seed:
        Gen.set_seed(seed)
    return nour.fromiter(low + Gen.random_raw(size=shape) % (high - low), dtype=int, requires_grad = requires_grad)

def uniform(low, high, shape = (1,), requires_grad = False, seed = None):
    if seed:
        Gen.set_seed(seed)
    return nour.node(low + rand(*shape) * (high - low), dtype=float, requires_grad = requires_grad)

def normal(mean = 0.0, std = 1.0, shape = (1,), requires_grad = False, seed = None):
    '''
    Box Muller transform
    '''

    if std < 0:
        raise nour.errors.InputError('Scale must be a positive number')
    
    if seed:
        Gen.set_seed(seed)

    u = rand(np.prod(shape), 2)
    return nour.node((-2 * np.log(u[:, 0]) ) ** 0.5 * np.cos(2 * np.pi * u[:, 1]) * std + mean, dtype=float, requires_grad = requires_grad).reshape(shape)