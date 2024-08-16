import numpy as np
from nour.utils.nutils import convert2node
from nour.functional import _utils
from nour import errors, utils
from nour.utils import nutils
import nour
import cython

def log_softmax(a):
    a = convert2node(a)
    result = nour.node( np.log(_utils._functional_softmax(a)) , requires_grad = False)
    if a.requires_grad and utils.no_grad._grad_mode_:
        result.requires_grad_(True)
        result.input_nodes.extend([a])
        result._node__op = 'log_softmax'
    return result

def normalize(x, centered = True):
    x = convert2node(x)
    x = x.view(np.ndarray)
    max_ = np.max(x)
    min_ = np.min(x) if centered else 0
    return (x - min_) / (max_ - min_)

def dot(a, b):
    a = convert2node(a)
    b = convert2node(b)
    if a.ndim != 1 and b.ndim != 1:
        raise errors.UnvalidTask(f'Dot product requests two nodes with 1D and the same number of elements. not {a.shape}, and {b.shape}. Instead you could you use scalar multiplication or matrix multiplication')
    result = nour.node(np.dot(a, b))
    if a.requires_grad or b.requires_grad and utils.no_grad._grad_mode_:
        result._node__requires_grad = True
        result._node__op = np.dot
        result._node__input_nodes = [a, b]
    return result
    
def matmul(a, b):
    a = convert2node(a)
    b = convert2node(b)
    if a.ndim < 2 and b.ndim < 2:
        raise errors.UnvalidTask(f'Matrix multiplication requests two nodes with 2D or higher dimensions. not {a.shape}, and {b.shape}. Instead you could you use scalar multiplication or dot product')
    result = nour.node(np.matmul(a, b))
    if a.requires_grad or b.requires_grad and utils.no_grad._grad_mode_:
        result.requires_grad_(True)
        result._node__op = np.matmul
        result._node__input_nodes = [a, b]
    return result

def sum(a, axis = None):
    a = convert2node(a)
    result = nour.node(np.sum(a, axis = axis), requires_grad = False)
    if a.requires_grad and utils.no_grad._grad_mode_:
        result.requires_grad_(True)
        result.input_nodes.extend([a])
        result._node__op = 'reduce_sum'
        result._node__op_parameters['axis'] = axis
        result._node__op_parameters['input_shape'] = a.shape
    return result
        
def sigmoid(a):
    a = convert2node(a)
    result = nour.node(1 / (1 + np.exp(-a)), requires_grad = False)
    if a.requires_grad and utils.no_grad._grad_mode_:
        result.requires_grad_(True)
        result.input_nodes.extend([a])
        result._node__op = 'sigmoid'
    return result

def softmax(a, axis = -1):
    a = convert2node(a)
    shape = list(a.shape)
    shape[axis] = 1
    result = nour.node(np.exp(a) / np.sum(np.exp(a), axis = axis).reshape(*shape), requires_grad = False)
    if a.requires_grad and utils.no_grad._grad_mode_:
        result.requires_grad_(True)
        result.input_nodes.extend([a])
        result._node__op = 'softmax'
    return result

def relu(a):
    a = convert2node(a)
    result = nour.node(np.maximum(a, 0), requires_grad=False)
    if a.requires_grad and utils.no_grad._grad_mode_:
        result.requires_grad_(True)
        result.input_nodes.extend([a])
        result._node__op = 'relu'
    return result

def norm(a):
    a = convert2node(a)
    result = nour.node(np.linalg.norm(a), requires_grad=False)
    if a.requires_grad and utils.no_grad._grad_mode_:
        result.requires_grad_(True)
        result.input_nodes.extend([a])
        result._node__op = 'norm'
    return result

def linear(input_, weights, bias):
    return matmul(input_, weights) + bias

def conv2d(input_, kernel, stride = (1, 1), padding = (0, 0), additional_padding = False):
    input_ : nour.node = convert2node(input_)
    kernel : nour.node =  convert2node(kernel)

    if additional_padding:
        add_pad_0 : cython.int = stride[0] % (2 * padding[0] + input_.shape[2])
        add_pad_1 : cython.int = stride[1] % (2 * padding[1] + input_.shape[3])
    else:
        add_pad_0 : cython.int = 0
        add_pad_1 : cython.int = 0

    pads : tuple = ((padding[0], padding[0] + add_pad_0), (padding[1], padding[1] + add_pad_1))
    full_padding : tuple = ((0, 0), (0, 0), *pads)

    if padding[0] or padding[1]: 
        input_padded : nour.node = np.pad(input_, pad_width=full_padding)
    else:
        input_padded : nour.node = input_

    batch_size, in_channels, input_height, input_width = input_padded.shape
    _, _, kernel_height, kernel_width = kernel.shape
    input_strides = input_padded.strides

    out_h : cython.int = ( ( ( input_height - kernel_height) / stride[0] ) + 1 ).__floor__()
    out_w : cython.int = ( ( ( input_width - kernel_width) / stride[1] ) + 1 ).__floor__()
    
    shape : tuple = (batch_size, in_channels, out_h, out_w, kernel_height, kernel_width)
    strides : tuple = (input_strides[0], input_strides[1], input_strides[2] * stride[0], input_strides[3] * stride[1], input_strides[2], input_strides[3])
    
    input_windows : nour.node = np.lib.stride_tricks.as_strided(input_padded, shape = shape, strides = strides, writeable=False)

    with utils.no_grad():
        result : nour.node = nour.node(np.tensordot(input_windows, kernel, axes = ((1, 4, 5), (1, 2, 3))).transpose(0, 3, 1, 2) ,requires_grad=False)

    if (input_.requires_grad or kernel.requires_grad) and utils.no_grad._grad_mode_:
        result.requires_grad_(True)
        result.input_nodes.extend([input_, kernel])
        result._node__op = 'conv2d'
        result._node__op_parameters['stride'] = stride
        result._node__op_parameters['padding'] = pads
    return result

def max_pool2d(input_, kernel_size, stride = (1, 1), padding = (0, 0), additional_padding = False):
    input_ = convert2node(input_)

    if additional_padding:
        add_pad_0 = stride[0] % (2 * padding[0] + input_.shape[2])
        add_pad_1 = stride[1] % (2 * padding[1] + input_.shape[3])
    else:
        add_pad_0, add_pad_1 = 0, 0

    pads = ((padding[0], padding[0] + add_pad_0), (padding[1], padding[1] + add_pad_1))
    full_padding = ((0, 0), (0, 0), *pads)

    input_padded = np.pad(input_, pad_width = full_padding, mode = 'constant', constant_values = -np.inf)

    out_h = ( ( ( input_padded.shape[2] - kernel_size[0]) / stride[0] ) + 1 ).__floor__()
    out_w = ( ( ( input_padded.shape[3] - kernel_size[1]) / stride[1] ) + 1 ).__floor__()
    
    shape = (input_.shape[0], input_.shape[1], out_h, out_w, kernel_size[0], kernel_size[1])
    new_shape = (input_.shape[0], input_.shape[1], out_h, out_w, kernel_size[0] * kernel_size[1])
    
    strides = (input_padded.strides[0], input_padded.strides[1], input_padded.strides[2] * stride[0], input_padded.strides[3] * stride[1], input_padded.strides[2], input_padded.strides[3])
    
    input_windows = np.lib.stride_tricks.as_strided(input_padded, shape = shape, strides = strides)
    new_input_windows = input_windows.reshape(new_shape)
    argmaxs = np.argmax(new_input_windows, axis = -1, keepdims = True)
    indices = np.indices(argmaxs.shape)
    argmax_indices = (indices[0], indices[1], indices[2], indices[3], argmaxs)

    result = nour.node( np.squeeze(new_input_windows[*argmax_indices], axis = -1), requires_grad = False)
    
    if input_.requires_grad and utils.no_grad._grad_mode_:
        result.requires_grad_(True)
        result.input_nodes.extend([input_])
        result._node__op = 'max_pool2d'
        result._node__op_parameters['argmax_indices'] = argmax_indices
        result._node__op_parameters['stride'] = stride
        result._node__op_parameters['padding'] = pads
        result._node__op_parameters['kernel_size'] = kernel_size
        result._node__op_parameters['input_padded_shape'] = input_padded.shape
    return result

def flatten(a, start_axis = 0, end_axis = -1):
    a = convert2node(a)
    if end_axis < 0:
        end_axis = a.ndim + end_axis
    shape = list(a.shape)
    result = a.reshape(*shape[:start_axis], np.prod([*shape[start_axis:end_axis+1]]), *shape[end_axis+1:])
    if a.requires_grad and utils.no_grad._grad_mode_:
        result.requires_grad_(True)
        result.input_nodes.extend([a])
        result._node__op = 'flatten'
        result._node__op_parameters['shape'] = shape
    return result

 
    


