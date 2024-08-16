import nour.functional as F
import numpy as np
import nour
from nour import errors
from nour.utils import nutils
from nour.nn import initializers

__builtin_modules__ = set()

class Module:

    def parameters(self):
        parameters_dict = {}
        for key, value in vars(self).items():
            if isinstance(value, nour.node):
                if value.requires_grad:
                    parameters_dict[key] = value
            elif isinstance(value, Sequential):
                for j, key_ in enumerate(value.parameters().parameters.keys()):
                    parameters_dict.update({'(Seq)'+ str(key) + f' ({key_})' : tuple(value.parameters().parameters.values())[j]})
            elif isinstance(value, Module):
                for j, key_ in enumerate(value.parameters().parameters.keys()):
                    name = f'{value.__class__.__name__}.' if value.__class__.__name__ not in __builtin_modules__ else ''
                    parameters_dict.update({name + str(key) + f'( {key_} )' : tuple(value.parameters().parameters.values())[j]})

        return Parameters(parameters_dict)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def __repr__(self):
        text = ''
        for attr_name , value in vars(self).items():
            if isinstance(value, Sequential):
                text += f'\n  ' + str(value).replace(f'\n  ', f'\n    ').replace('\n)', '\n  )')
            elif isinstance(value, Module):
                text += f'\n  ' + str(value).replace(f'\n  ', f'\n    ').replace('\n)', '\n  )')
        return f'{self.__class__.__name__}({text}\n)'
            
class Parameters:    
    def __init__(self, parameters):
        if isinstance(parameters, dict):
            self.__parameters = parameters

        elif isinstance(parameters, nour.node):
            self.__parameters = {'node 0' : parameters}

        elif getattr(parameters, '__iter__', False):
            self.__parameters = {}
            for i, value in enumerate(parameters):
                if isinstance(value, nour.node):
                    self.__parameters[f'node {i}'] = value 
                else:
                    raise errors.InputError(f'All given parameters must be nodes not {type(value)}')

        else:
            raise errors.InputError(f'input must be a dictionary or a set of nodes not ', type(parameters))
        
    @property
    def parameters(self):
        return self.__parameters
    
    def __repr__(self):
        return f'Parameters:\n\n' + str(self.__parameters).replace(']), ', ']), \n\n').replace('{', '').replace('}', '').replace(':', ':\n')
    
    def __iter__(self):
        return iter(self.__parameters.values())

    def values(self):
        return self.__parameters.values()
    
    def items(self):
        return self.__parameters.items()
    
class Sequential:
    def __init__(self, *layers):
        self.layers = layers
        self._text = ''
        for layer in self.layers:
            self._text+=f'  {str(layer)}\n'
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        parameters_dict = {}
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, Module):
                for j, key in enumerate(layer.parameters().parameters.keys()):
                    parameters_dict.update({str(key)+f'_{idx}' : tuple(layer.parameters().parameters.values())[j]})
        return Parameters(parameters_dict)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(\n{self._text})'
    
    def __getitem__(self, item):
        return self.layers[item]
    
class Parameter(Module):
    __builtin_modules__.add('Parameter')

    def __init__(self, parameter, fun):
        parameter = nour.convert2node(parameter)
        parameter.requires_grad_(True)
        self.parameter = parameter
        self.function = fun 
        
    def forward(self, x):
        return self.function(self.parameter, x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(parameter_shape = {self.parameter.shape})'

class Linear(Module):
    __builtin_modules__.add('Linear')
    
    def __init__(self, in_shape, out_shape, bias = True, weight_initializer = 'torch_uniform', bias_initializer = 'torch_uniform', weight_initializer_parameters = {}, bias_initializer_parameters = {}):
        weight_params = {'in_shape' : in_shape, 'out_shape' : out_shape}
        weight_params.update(weight_initializer_parameters)
        
        self.in_shape = in_shape
        self.out_shape = out_shape

        weight_init = initializers.get_initializer(weight_initializer)(**weight_params)
        self.weights = weight_init(shape = (in_shape, out_shape), requires_grad = True)

        if bias:
            bias_params = {'in_shape' : in_shape, 'out_shape' : out_shape}
            bias_params.update(bias_initializer_parameters)
            bias_init = initializers.get_initializer(bias_initializer)(**bias_params)
            self.bias = bias_init(shape = (1, out_shape), requires_grad = True)
        else:
            self.bias = None

    def forward(self, input_array):
        if self.bias is not None:
            if input_array.ndim == 1:
                return F.sum(F.linear(input_array[np.newaxis], self.weights, self.bias), axis=0)
            return F.linear(input_array, self.weights, self.bias)
        
        if input_array.ndim == 1:
            return F.sum(F.matmul(input_array[np.newaxis], self.weights), axis=0)
        return F.matmul(input_array, self.weights)
        
    def __repr__(self):
        return f'{self.__class__.__name__}(in_shape = {self.in_shape}, out_shape = {self.out_shape})'

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, additional_padding = False, bias = True, weight_initializer = 'torch_uniform', bias_initializer = 'torch_uniform', weight_initializer_parameters = {}, bias_initializer_parameters = {}):
        self.in_channels, self.out_channels = in_channels, out_channels
        
        if nutils.isNumber(padding):
            padding = (padding, padding)
    
        if nutils.isNumber(stride):
            stride = (stride, stride)
    
        if nutils.isNumber(kernel_size):
            kernel_size = (kernel_size, kernel_size)
            
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.additional_padding = additional_padding

        weight_params = {'in_shape' : in_channels, 'out_shape' : out_channels}
        weight_params.update(weight_initializer_parameters)
        weight_init = initializers.get_initializer(weight_initializer)(**weight_params)
        self.weights = weight_init(shape = (out_channels, in_channels, kernel_size[0], kernel_size[1]), requires_grad = True)
        
        if bias:
            bias_params = {'in_shape' : in_channels, 'out_shape' : out_channels}
            bias_params.update(bias_initializer_parameters)
            bias_init = initializers.get_initializer(bias_initializer)(**bias_params)
            self.bias = bias_init(shape = (out_channels, 1, 1), requires_grad = True)

        else:
            self.bias = None
        
    def forward(self, x):
        if not nour.isNode(x):
            raise nour.errors.InputError('Input must be a nour.Node, got ', type(x))

        if x.ndim != 4:
            raise nour.errors.ShapeError('Input shape expected to be (Batch Size, Channels, Height, Width) with 4 dimensions, got ', x.shape)
        
        output = F.conv2d(x, self.weights, stride = self.stride, padding = self.padding, additional_padding = self.additional_padding)
                   
        if self.bias is not None:
            output += self.bias
            
        return output
            
    def __repr__(self):
        add_text = ''
        if self.additional_padding:
            add_text += f', additional_padding = {self.additional_padding}'
        if self.bias is None:
            add_text += f', bias = {self.bias}'

        return f'{self.__class__.__name__}(in_channels = {self.in_channels}, out_channels = {self.out_channels}, kernel_size = {self.kernel_size}, stride = {self.stride}, padding = {self.padding}{add_text})'

    def calculate_output_shape(self, input_shape):
        out_h = ( ( input_shape[2] + 2 * self.padding[0] - self.kernel_size[0] ) // self.stride[0] ) + 1
        out_w = ( ( input_shape[3] + 2 * self.padding[1] - self.kernel_size[1] ) // self.stride[1] ) + 1
        return (input_shape[0], input_shape[1], out_h, out_w)

class MaxPool2d:
    def __init__(self, kernel_size, stride = 1, padding = 0, additional_padding = False):
        if nutils.isNumber(padding):
            padding = (padding, padding)
    
        if nutils.isNumber(stride):
            stride = (stride, stride)
    
        if nutils.isNumber(kernel_size):
            kernel_size = (kernel_size, kernel_size)
            
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.additional_padding = additional_padding

    def __call__(self, x):
        return F.max_pool2d(x, kernel_size = self.kernel_size, stride=self.stride, padding = self.padding, additional_padding=self.additional_padding)

    def __repr__(self):
        add_text = ''
        if self.additional_padding:
            add_text += f', additional_padding = {self.additional_padding}'
        return f'{self.__class__.__name__}(kernel_size = {self.kernel_size}, stride = {self.stride}, padding = {self.padding}{add_text})'
        
class Relu():
    def __call__(self, a):
        return nour.relu(a)
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'
    
class Flatten():
    def __init__(self, start_axis = 1, end_axis = -1):
        self.start_axis = start_axis
        self.end_axis = end_axis

    def __call__(self, x):
        return F.flatten(x, start_axis=self.start_axis, end_axis=self.end_axis)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(start_axis = {self.start_axis}, end_axis = {self.end_axis})'
    
class Sigmoid():
    def __call__(self, a):
        return nour.sigmoid(a)
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'
    
class Softmax():
    def __init__(self, axis = None, finite = True):
        self.axis = axis if axis else -1
        self.finite = finite
        self.__limit = 100
        
    def __call__(self, a, axis = None):
        axis = axis if axis else self.axis
        if self.finite:
            return nour.functional._utils._functional_finite_softmax(a, axis = axis, limit = self.__limit)
        else:
            return nour.softmax(a, axis = axis)
        
    def set_finite_limit(self, limit):
        '''
        Sets the maximum possible result number of the softmax opertaion.
        Note:
            This won't make any difference if the attribute (self.finit) is set to False.

        Parameters:
            limit: the maximum limit of the softmax operation result in the form `exp(limit)`,
                    Which means thats `limit` is not the pure maximum number but its raised by exceptional
        
        Returns:
            void
        '''

        self.__limit = limit

    def __repr__(self):
        return f'{self.__class__.__name__}(axis = {self.axis}, finite = {self.finite})'

class MSELoss():
    def __init__(self, axis = 0, reduction = None):
        self.axis = axis
        self.reduction = reduction
        
    def __call__(self, input_, target):
        return nour.mse_loss(input_, target, axis = self.axis, reduction = self.reduction)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(axis = {self.axis}, reduction = {self.reduction})'
        

class BinaryCrossEntropyLoss():
    def __init__(self, axis = 0, reduction = None, finite = True):
        self.axis = axis
        self.reduction = reduction
        self.finite = finite
        self.__limit = 100
        
    def __call__(self, input_, target):
        return nour.binary_cross_entropy_loss(input_, target, axis = self.axis, reduction = self.reduction, finite = self.finite, finite_limit=self.__limit)
    
    def set_finite_limit(self, limit):
        '''
        Sets the maximum possible result number of the BinaryCrossEntropyLoss opertaion.
        Note:
            This won't make any difference if the attribute (self.finit) is set to False.

        Parameters:
            limit: the maximum limit of the BinaryCrossEntropyLoss operation result in the form `exp(limit)`,
                    Which means thats `limit` is not the pure maximum number but its raised by exceptional
        
        Returns:
            void
        '''

        self.__limit = limit

    def __repr__(self):
        if self.finite:
            return f'{self.__class__.__name__}(axis = {self.axis}, reduction = {self.reduction}, finite = {self.finite}, finite_limit = {self.__limit})'
        else:
            return f'{self.__class__.__name__}(axis = {self.axis}, reduction = {self.reduction}, finite = {self.finite})'
    
class CrossEntropyLoss():
    def __init__(self, axis = 0, reduction = None, finite = True):
        self.axis = axis
        self.reduction = reduction
        self.finite = finite
        self.__limit = 100
        
    def __call__(self, input_, target):
        return nour.cross_entropy_loss(input_, target, axis = self.axis, reduction = self.reduction, finite = self.finite, finite_limit=self.__limit)
    
    def set_finite_limit(self, limit):
        '''
        Sets the maximum possible result number of the CrossEntropyLoss opertaion.
        Note:
            This won't make any difference if the attribute (self.finit) is set to False.

        Parameters:
            limit: the maximum limit of the CrossEntropyLoss operation result in the form `exp(limit)`,
                    Which means thats `limit` is not the pure maximum number but its raised by exceptional
        
        Returns:
            void
        '''

        self.__limit = limit

    def __repr__(self):
        if self.finite:
            return f'{self.__class__.__name__}(axis = {self.axis}, reduction = {self.reduction}, finite = {self.finite}, finite_limit = {self.__limit})'
        else:
            return f'{self.__class__.__name__}(axis = {self.axis}, reduction = {self.reduction}, finite = {self.finite})'

