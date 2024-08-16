import numpy as np
import nour
from nour import errors
from nour.utils import GradientFunction
from nour.functional._utils import _finite_log, _finite_division
from nour.utils import nutils

@GradientFunction(np.add)
def _add_grad(node_):
    if len(node_.input_nodes) > 1:    
        if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(node_.grad)
        if getattr(node_.input_nodes[1], 'requires_grad', None): node_.input_nodes[1]._add_grad(node_.grad)
    else:
        if getattr(node_.input_nodes[0], 'requires_grad', None):
            shape = list(node_.input_nodes[0].shape)
            axis = node_.op_parameters['axis']
            if not isinstance(axis, type(None)): 
                shape[axis] = 1
            else:
                shape = 1
            node_.input_nodes[0]._add_grad(node_.grad.reshape(shape))
            
@GradientFunction('reduce_sum')
def _reduce_sum_grad(node_):
    if getattr(node_.input_nodes[0], 'requires_grad', None):
            shape = list(node_.input_nodes[0].shape)
            axis = node_.op_parameters['axis']
            if not isinstance(axis, type(None)): 
                shape[axis] = 1
            else:
                shape = 1
            node_.input_nodes[0]._add_grad(node_.grad.reshape(shape))
        
@GradientFunction(np.multiply)
def _multiply_grad(node_):
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(node_.grad * node_.input_nodes[1])
    if getattr(node_.input_nodes[1], 'requires_grad', None): node_.input_nodes[1]._add_grad(node_.grad * node_.input_nodes[0])
        
@GradientFunction(np.subtract)
def _subtract_grad(node_):
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(node_.grad)
    if getattr(node_.input_nodes[1], 'requires_grad', None): node_.input_nodes[1]._add_grad(-node_.grad)
        
@GradientFunction(np.divide)
def _divide_grad(node_):
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(node_.grad * (1 / node_.input_nodes[1]))
    if getattr(node_.input_nodes[1], 'requires_grad', None): node_.input_nodes[1]._add_grad(node_.grad * (-node_.input_nodes[0] / (node_.input_nodes[1] ** 2)))

@GradientFunction(np.square)
def _power_grad(node_):
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(node_.grad * 2 * node_.input_nodes[0])
        
@GradientFunction(np.power)
def _power_grad(node_):
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(node_.grad * node_.input_nodes[1] * node_.input_nodes[0] ** (node_.input_nodes[1] - 1))
    if getattr(node_.input_nodes[1], 'requires_grad', None): node_.input_nodes[1]._add_grad(node_.grad * (node_.input_nodes[0] ** node_.input_nodes[1] * np.log(node_.input_nodes[0])))
               
@GradientFunction(np.mod)
def _mod_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None) or getattr(node_.input_nodes[1], 'requires_grad', None): print('The mod operation is non differentiable, Even though the derivative does not exist, it\'s derivative is considered to be zero for all inputs.')

@GradientFunction(np.floor_divide)
def _floor_divide_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None) or getattr(node_.input_nodes[1], 'requires_grad', None): print('The floor_divide operation is non differentiable, Even though the derivative does not exist, it\'s derivative is considered to be zero for all inputs.')
        
@GradientFunction(np.logical_and)
def _logical_and_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None) or getattr(node_.input_nodes[1], 'requires_grad', None): print('The logical_and operation is non differentiable, Even though the derivative does not exist, it\'s derivative is considered to be zero for all inputs.')

@GradientFunction(np.logical_or)
def _logical_or_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None) or getattr(node_.input_nodes[1], 'requires_grad', None): print('The logical_or operation is non differentiable, Even though the derivative does not exist, it\'s derivative is considered to be zero for all inputs.')
        
@GradientFunction(np.logical_xor)
def _logical_xor_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None) or getattr(node_.input_nodes[1], 'requires_grad', None): print('The logical_xor operation is non differentiable, Even though the derivative does not exist, it\'s derivative is considered to be zero for all inputs.')
        
@GradientFunction(np.left_shift)
def _left_shift_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None) or getattr(node_.input_nodes[1], 'requires_grad', None): print('The left_shift operation is non differentiable, Even though the derivative does not exist, it\'s derivative is considered to be zero for all inputs.')
        
@GradientFunction(np.right_shift)
def _right_shift_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None) or getattr(node_.input_nodes[1], 'requires_grad', None): print('The right_shift operation is non differentiable, Even though the derivative does not exist, it\'s derivative is considered to be zero for all inputs.')

@GradientFunction(np.negative)
def _negative_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(-node_.grad)
        
@GradientFunction(np.positive)
def _positive_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(node_.grad)
        
@GradientFunction(np.absolute)
def _absolute_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None):
        if 0 in node_.input_nodes[0]: print('The absolute operation is non differentiable at zero, Even though it\'s derivative is considered to be zero at zero for computaions.')
        node_.input_nodes[0]._add_grad(node_.grad * np.sign(node_.input_nodes[0]))
                                        
@GradientFunction(np.invert)
def _invert_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None): print('The invert operation is non differentiable, Even though the derivative does not exist, it\'s derivative is considered to be zero for all inputs.')                       
                                        
@GradientFunction(np.greater)
def _greater_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None) or getattr(node_.input_nodes[1], 'requires_grad', None): print('The greater operation is non differentiable, Even though the derivative does not exist, it\'s derivative is considered to be zero for all inputs.')
        
@GradientFunction(np.less)
def _less_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None) or getattr(node_.input_nodes[1], 'requires_grad', None): print('The less operation is non differentiable, Even though the derivative does not exist, it\'s derivative is considered to be zero for all inputs.')

@GradientFunction(np.greater_equal)
def _greater_equal_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None) or getattr(node_.input_nodes[1], 'requires_grad', None): print('The greater_equal operation is non differentiable, Even though the derivative does not exist, it\'s derivative is considered to be zero for all inputs.')
        
@GradientFunction(np.less_equal)
def _less_equal_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None) or getattr(node_.input_nodes[1], 'requires_grad', None): print('The less_equal operation is non differentiable, Even though the derivative does not exist, it\'s derivative is considered to be zero for all inputs.')

@GradientFunction(np.equal)
def _equal_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None) or getattr(node_.input_nodes[1], 'requires_grad', None): print('The equal operation is non differentiable, Even though the derivative does not exist, it\'s derivative is considered to be zero for all inputs.')
        
@GradientFunction(np.not_equal)
def _not_equal_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None) or getattr(node_.input_nodes[1], 'requires_grad', None): print('The not_equal operation is non differentiable, Even though the derivative does not exist, it\'s derivative is considered to be zero for all inputs.')
        
@GradientFunction(np.exp)
def _exp_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(node_.grad * np.exp(node_.input_nodes[0]))

@GradientFunction(np.log)
def _log_grad(node_): 
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(1 / node_.input_nodes[0] * node_.grad)
                                        
@GradientFunction(np.dot)
def _dot_grad(node_):
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad((node_.grad * node_.input_nodes[1].view(np.ndarray)).reshape(*node_.input_nodes[0].shape))
    if getattr(node_.input_nodes[1], 'requires_grad', None): node_.input_nodes[1]._add_grad((node_.grad * node_.input_nodes[0].view(np.ndarray)).reshape(*node_.input_nodes[1].shape))
    
@GradientFunction(np.matmul)
def _matmul_grad(node_):
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(np.matmul(node_.grad , nour.transpose_rowcolumn_only(node_.input_nodes[1].view(np.ndarray))))
    if getattr(node_.input_nodes[1], 'requires_grad', None): node_.input_nodes[1]._add_grad(np.matmul(nour.transpose_rowcolumn_only(node_.input_nodes[0].view(np.ndarray)), node_.grad))
        
@GradientFunction('sigmoid')
def _sigmoid_grad(node_):
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(node_.grad * node_.view(np.ndarray) * (1 - node_))
        
@GradientFunction('softmax')
def _softmax_grad(node_):
    if getattr(node_.input_nodes[0], 'requires_grad', None):
        shape = list(node_.shape)
        if node_.ndim > 1:
            shape[-2] = shape[-1]
        node_.input_nodes[0]._add_grad((node_.grad - np.matmul(node_.grad * node_.view(np.ndarray), np.ones(shape = shape))) * node_.view(np.ndarray))          

@GradientFunction('mse_loss')
def _mse_loss_grad(node_):
    a = node_.input_nodes[0].view(np.ndarray)
    b = node_.input_nodes[1].view(np.ndarray)
    axis = node_.op_parameters['axis']
    reduction = node_.op_parameters['reduction']
    
    if a.shape == b.shape:
        shape = a.shape
    else:
        if a.ndim > b.ndim: shape = a.shape
        elif b.ndim < b.ndim: shape = b.shape
        else:
            if a.shape[axis] >= b.shape[axis]: shape = a.shape
            elif a.shape[axis] < b.shape[axis]: shape = b.shape
            else: raise nour.errors.ShapeError('Can not take grad of both the input and the target with different shapes such ', a.shape, ' and ', b.shape, 'shapes excpected to be the same')
    
    reduction_element = shape[axis] if reduction == 'mean' or not reduction else 1
    
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(((a - b) * 2) / reduction_element * node_.grad)
    if getattr(node_.input_nodes[1], 'requires_grad', None): node_.input_nodes[1]._add_grad(((b - a) * 2) / reduction_element * node_.grad)


@GradientFunction('binary_cross_entropy_loss')
def _binary_cross_entropy_loss_grad(node_: nour.node):
    a = node_.input_nodes[0].view(np.ndarray)
    b = node_.input_nodes[1].view(np.ndarray)
    
    axis = node_.op_parameters['axis']
    reduction = node_.op_parameters['reduction']
    a_sigmoid = node_.op_parameters['sigmoid']
    finite = node_.op_parameters['finite']
    
    if a.shape == b.shape:
        shape = a.shape
    else:
        if a.ndim > b.ndim: shape = a.shape
        elif b.ndim < b.ndim: shape = b.shape
        else:
            if a.shape[axis] >= b.shape[axis]: shape = a.shape
            elif a.shape[axis] < b.shape[axis]: shape = b.shape
            else: raise errors.ShapeError('Can not take grad of both the input and the target with different shapes such ', a.shape, ' and ', b.shape, 'shapes excpected to be the same')
    
    log_fn = _finite_log if finite else np.log
    divide_fn = _finite_division if finite else np.divide
    reduction_element = shape[axis] if reduction == 'mean' or not reduction else 1
    grad_a = np.multiply(node_.grad, np.divide(np.subtract(divide_fn(np.subtract(1, b), np.subtract(1, a_sigmoid)), divide_fn(b, a_sigmoid)), reduction_element))
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(grad_a * a_sigmoid * (1 - a_sigmoid))
    if getattr(node_.input_nodes[1], 'requires_grad', None): node_.input_nodes[1]._add_grad(node_.grad * divide_fn(log_fn(1 - a_sigmoid) - log_fn(a_sigmoid), reduction_element))
        
@GradientFunction('cross_entropy_loss')
def _cross_entropy_loss_grad(node_: nour.node):
    a = node_.input_nodes[0].view(np.ndarray)
    b = node_.input_nodes[1].view(np.ndarray)
    axis = node_.op_parameters['axis']
    reduction = node_.op_parameters['reduction']
    a_softmax = node_.op_parameters['softmax']
    
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    
    zeros = np.zeros(shape = a_softmax.shape)
    grid = np.indices((*a_softmax.shape[:-1], 1))
    zeros[*grid[:-1], b] = np.ones(shape = a_softmax.shape)[*grid[:-1], b]
    
    if a.shape == b.shape:
        shape = a.shape
    else:
        if a.ndim > b.ndim: shape = a.shape
        elif b.ndim < b.ndim: shape = b.shape
        else:
            if a.shape[axis] >= b.shape[axis]: shape = a.shape
            elif a.shape[axis] < b.shape[axis]: shape = b.shape
            else: raise errors.ShapeError('Can not take grad of both the input and the target with different shapes such ', a.shape, ' and ', b.shape, 'shapes excpected to be the same')
    
    reduction_element = shape[axis] if reduction == 'mean' or not reduction else 1
    grad_a = node_.grad * (a_softmax - zeros)
    
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(grad_a / reduction_element)
    if getattr(node_.input_nodes[1], 'requires_grad', None):
        n_features = a.shape[-1]
        probs = np.max(np.asarray(a_softmax), axis = -1)
        base_array = np.arange(0, n_features)
        grads = []
        for arr, prob in zip(b, probs):
            base_array_2 = np.delete(arr - base_array, arr)
            base_grads = []
            for i in range(len(base_array_2)):
                base_grads.append(np.prod(np.delete(base_array_2, i)))
            grads.append(np.prod(base_grads) * prob * n_features)
        grads = np.array(grads).reshape(b.shape)
        node_.input_nodes[1]._add_grad(grads * node_.grad)
        
@GradientFunction('relu')
def _relu_grad(node_: nour.node):
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(node_.grad * np.sign(node_.view(np.ndarray)))
        
@GradientFunction('norm')
def _norm_grad(node_: nour.node):
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(node_.grad * node_.input_nodes[0].view(np.ndarray))

@GradientFunction('conv2d')
def _conv2d_grad(node_: nour.node):

    stride_h, stride_w = node_.op_parameters['stride']
    padding = node_.op_parameters['padding']

    input_, kernel = node_.input_nodes[0].view(np.ndarray), node_.input_nodes[1].view(np.ndarray)
    input_padded = np.pad(node_.input_nodes[0], pad_width = ((0, 0), (0, 0), *padding))

    shape = (input_padded.shape[0], input_padded.shape[1], kernel.shape[2], kernel.shape[3], node_.shape[2], node_.shape[3])

    if getattr(node_.input_nodes[0], 'requires_grad', None):
        y_dim = slice(padding[0][0], input_padded.shape[2]-padding[0][1])
        x_dim = slice(padding[1][0], input_padded.shape[3]-padding[1][1])
        
        zeros = np.zeros(input_padded.shape)
        ind = np.indices(input_.shape)
    
        y_axes = np.lib.stride_tricks.as_strided(ind[3], shape = shape, strides = (ind[3].strides[0], ind[3].strides[1], ind[3].strides[2], ind[3].strides[3], ind[3].strides[2] * stride_h, ind[3].strides[3] * stride_w))
        x_axes = np.lib.stride_tricks.as_strided(ind[2], shape = shape, strides = (ind[2].strides[0], ind[2].strides[1], ind[2].strides[2], ind[2].strides[3], ind[2].strides[2] * stride_h, ind[2].strides[3] * stride_w))
        z_axes = np.lib.stride_tricks.as_strided(ind[1], shape = shape, strides = (ind[1].strides[0], ind[1].strides[1], ind[1].strides[2], ind[1].strides[3], ind[1].strides[2] * stride_h, ind[1].strides[3] * stride_w))
        batch_axes = np.lib.stride_tricks.as_strided(ind[0], shape = shape, strides = (ind[0].strides[0], ind[0].strides[1], ind[0].strides[2], ind[0].strides[3], ind[0].strides[2] * stride_h, ind[0].strides[3] * stride_w))
        
        kernel_node_grad = np.sum(np.expand_dims(node_.grad, axis = (2, 3, 4)) * np.expand_dims(kernel, axis = (4, 5)), axis = 1)
        np.add.at(zeros, (batch_axes, z_axes, x_axes, y_axes), kernel_node_grad)
        node_.input_nodes[0]._add_grad(zeros[..., y_dim, x_dim])

    if getattr(node_.input_nodes[1], 'requires_grad', None):
        strides = (input_padded.strides[0], input_padded.strides[1], input_padded.strides[2], input_padded.strides[3], input_padded.strides[2] * stride_h, input_padded.strides[3] * stride_w)
        input_windows = np.lib.stride_tricks.as_strided(input_padded, 
                                                       shape = shape,
                                                       strides = strides)
        node_.input_nodes[1]._add_grad(np.tensordot(node_.grad, input_windows, axes=((0, 2, 3), (0, 4, 5))))

@GradientFunction('max_pool2d')
def _max_pool2d_grad(node_ : nour.node):
    if getattr(node_.input_nodes[0], 'requires_grad', None):
        stride_h, stride_w = node_.op_parameters['stride']
        padding = node_.op_parameters['padding']
        kernel_size = node_.op_parameters['kernel_size']
        argmax_indices = node_.op_parameters['argmax_indices']
        input_padded_shape = node_.op_parameters['input_padded_shape']

        zeros = np.zeros(input_padded_shape)

        y_dim = slice(padding[0][0], zeros.shape[2]-padding[0][1])
        x_dim = slice(padding[1][0], zeros.shape[3]-padding[1][1])

        argmax = np.squeeze(argmax_indices[-1], axis=-1)
        ind = np.indices((node_.shape[2], node_.shape[3]))
        grad_indices = np.indices(node_.shape)

        y_axis = argmax % kernel_size[1] + ind[1] * stride_w
        x_axis = argmax // kernel_size[1] + ind[0] * stride_h

        np.add.at(zeros, (grad_indices[0], grad_indices[1], x_axis, y_axis), node_.grad)

        node_.input_nodes[0]._add_grad(zeros[..., y_dim, x_dim])

@GradientFunction('flatten')
def _flatten_grad(node_ : nour.node):
    if getattr(node_.input_nodes[0], 'requires_grad', None): node_.input_nodes[0]._add_grad(node_.grad.reshape(*node_.op_parameters['shape']))
