from typing import Any
import nour

class Initializers:
    def parameters(self):
        return self.__init__.__code__.co_varnames[:self.__init__.__code__.co_argcount]

    def __repr__(self):
        return f'{self.__class__.__name__} Initializer'

class Constant(Initializers):

    def __init__(self, value, dtype = None, **kwargs):
        self.value = value
        self.dtype = dtype

    def __call__(self, shape, requires_grad = False):
        return nour.full(value = self.value, shape = shape, dtype = self.dtype, requires_grad = requires_grad)
    
class GlorotNormal(Initializers):

    def __init__(self, in_shape : int, out_shape : int, seed = None, **kwargs):
        self.seed = seed
        self.std = ( 2 / (in_shape + out_shape) ) ** 0.5

    def __call__(self, shape, requires_grad = False):
        return nour.random.normal(mean = 0, std = self.std, shape=shape, requires_grad = requires_grad, seed=self.seed)

class GlorotUniform(Initializers):

    def __init__(self, in_shape : int, out_shape : int, seed = None, **kwargs):
        self.seed = seed
        self.limit =  ( 6 / ( in_shape + out_shape ) ) ** 0.5 

    def __call__(self, shape, requires_grad = False):
        return nour.random.uniform(low = -self.limit, high = self.limit, shape=shape, requires_grad = requires_grad, seed=self.seed)
 
class HeNormal(Initializers):

    def __init__(self, in_shape : int, seed = None, **kwargs):
        self.seed = seed
        self.std = ( 2 / in_shape ) ** 0.5

    def __call__(self, shape, requires_grad = False):
        return nour.random.normal(mean = 0, std = self.std, shape=shape, requires_grad = requires_grad, seed=self.seed)

class HeUniform(Initializers):

    def __init__(self, in_shape : int, seed = None, **kwargs):
        self.seed = seed
        self.limit =  ( 6 / in_shape ) ** 0.5 

    def __call__(self, shape, requires_grad = False):
        
        return nour.random.uniform(low = -self.limit, high = self.limit, shape=shape, requires_grad = requires_grad, seed=self.seed)

class LeconNormal(Initializers):

    def __init__(self, in_shape, seed = None, **kwargs):
        self.seed = seed
        self.std = ( 1 / in_shape ) ** 0.5

    def __call__(self, in_shape, shape, requires_grad = False):
        return nour.random.normal(mean = 0, std = self.std, shape=shape, requires_grad = requires_grad, seed=self.seed)

class LeconUniform(Initializers):

    def __init__(self, in_shape : int, seed = None, **kwargs):
        self.seed = seed
        self.limit = ( 3 / in_shape ) ** 0.5 

    def __call__(self, shape, requires_grad = False):
        return nour.random.uniform(low = -self.limit, high = self.limit, shape=shape, requires_grad = requires_grad, seed=self.seed)

class Ones(Initializers):

    def __init__(self, **kwargs):
        pass

    def __call__(self, shape, dtype = None ,requires_grad = False):
        return nour.full(value=1, shape=shape, dtype=dtype, requires_grad=requires_grad, seed=self.seed)
    

class RandomNormal(Initializers):

    def __init__(self, mean = 0.0, std = 0.5, seed = None, **kwargs):
        self.seed = seed
        self.mean = mean
        self.std = std

    def __call__(self, shape, requires_grad = False):
        return nour.random.normal(mean = self.mean, std = self.std, shape=shape, requires_grad = requires_grad, seed=self.seed)

class RandomUniform(Initializers):

    def __init__(self, low = -0.05, high = 0.05, seed = None, **kwargs):
        self.seed = seed
        self.low = low
        self.high = high

    def __call__(self, shape, requires_grad = False):
        return nour.random.uniform(low = self.low, high = self.high, shape=shape, requires_grad = requires_grad, seed=self.seed)

class TruncatedNormal(Initializers):

    def __init__(self, mean = 0.0, std = 0.05, seed = None, **kwargs):
        self.seed = seed
        self.mean = mean
        self.std = std

    def __call__(self, shape, requires_grad = False):
        return nour.random.normal(mean = self.mean, std = self.std, shape=shape, requires_grad = requires_grad, seed=self.seed)

class ScaledNormal(Initializers):

    def __init__(self, scale, n, mean = 0.0, seed = None, **kwargs):
        self.seed = seed
        self.mean = mean
        self.std = ( scale / n ) ** 0.5

    def __call__(self, shape, requires_grad = False):
        return nour.random.normal(mean = self.mean, std = self.std, shape=shape, requires_grad = requires_grad, seed=self.seed)

class ScaledUniform(Initializers):

    def __init__(self, scale, n, seed = None, **kwargs):
        self.seed = seed
        self.limit = ( 3 * scale / n ) ** 0.5

    def __call__(self, shape, requires_grad = False):
        return nour.random.uniform(low = -self.limit, high = self.limit, shape=shape, requires_grad = requires_grad, seed=self.seed)

class TorchUniform(Initializers):

    def __init__(self, in_shape : int, seed = None, **kwargs):
        self.seed = seed
        self.limit = ( 1 / in_shape ) ** 0.5 

    def __call__(self, shape, requires_grad = False):
        return nour.random.uniform(low = -self.limit, high = self.limit, shape=shape, requires_grad = requires_grad, seed=self.seed)

class Zeros(Initializers):

    def __init__(self, **kwargs):
        pass

    def __call__(self, shape, dtype = None ,requires_grad = False):
        return nour.full(value=0, shape=shape, dtype=dtype, requires_grad=requires_grad, seed=self.seed)

class Identity(Initializers):
    def __init__(self, value = 1.0, **kwargs):
        self.value = value

    def __call__(self, shape, dtype = None ,requires_grad = False):
        if len(shape) != 2:
            raise nour.errors.ShapeError('number of dimension of identity initializers must be 2 to get a square matrix, got ', len(shape))
        
        if shape[-1] != shape[-2]:
            raise nour.errors.ShapeError(f'first and second dimension must have the same shape to get a square matrix, got {shape[-1]} and {shape[-2]}')

        with nour.utils.no_grad():
            result = nour.identity(shape[-1], dtype=dtype, requires_grad=requires_grad) * self.value

        return result

class _InitsDict:

    def __init__(self):

        self._inits = {
            'constant' : Constant,
            'ones' : Ones,
            'zeors' : Zeros,
            'identity' : Identity,
            'glorot_normal' : GlorotNormal,
            'glorot_uniform' : GlorotUniform,
            'he_normal' : HeNormal,
            'he_uniform' : HeUniform,
            'lecon_normal' : LeconNormal,
            'lecon_uniform' : LeconUniform,
            'random_normal' : RandomNormal,
            'random_uniform' : RandomUniform,
            'scaled_normal' : ScaledNormal,
            'scaled_uniform' : ScaledUniform,
            'truncated_normal' : TruncatedNormal,
            'torch_uniform' : TorchUniform
        }
        self._init_done = True

    # def __setattr__(self, name: str, value: Any):
    #     if getattr(self, '_init_done', False):
    #         raise nour.errors.SetAttribute('Attriputes can not be set or modified for this class')
        
    def get(self, initializer_name):
        return self._inits[initializer_name]
    
    def add(self, initializer_name, initializer_class):
        if initializer_name in self._inits:
            raise nour.errors.UnvalidTask('Can not use initializer name thats already used')
    
        if not isinstance(initializer_class, Initializers):
            raise nour.errors.InputError('initializer class must be an instance of nour.dl.Initializers class')

        self._inits[initializer_name] = initializer_class

    def get_initializer_names(self):
        return list(self._inits)
    
_initsDict = _InitsDict()

def get_initializer(initializer_name):
    return _initsDict.get(initializer_name=initializer_name)

def add_initializer(initializer_name, initializer_class):
    return _initsDict.add(initializer_name=initializer_name, initializer_class = initializer_class)

def get_initializer_names():
    return _initsDict.get_initializer_names()