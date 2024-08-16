import nour
from nour import utils
from nour.utils.nutils import isNode
import numpy as np
from nour.functional import _utils

@utils.LossFunction('mse_loss')
def mse_loss(input_, target, axis = 0, reduction = None):
    if not isNode(input_) or not isNode(target):
        raise nour.errors.InputError(f'Inputs expected to be nodes not, {type(input_)} and {type(target)}')
    
    result = (target.view(np.ndarray) - input_.view(np.ndarray)) ** 2

    if not reduction or reduction == 'mean':
        result = nour.node(np.mean(result, axis = axis), requires_grad = False)
    elif reduction == 'sum':
        result = nour.node(np.sum(result, axis = axis), requires_grad = False)
    else:
        raise nour.errors.InputError('reduction must be None, `mean` or `sum` not ', reduction)
    if (input_.requires_grad or target.requires_grad) and utils.no_grad._grad_mode_:
        result.requires_grad_(True)
        result.input_nodes.extend([input_, target])
        result._node__op = 'mse_loss'
        result._node__op_parameters['axis'] = axis
        result._node__op_parameters['reduction'] = reduction
    return result

@utils.LossFunction('binary_cross_entropy_loss')   
def binary_cross_entropy_loss(input_, target, axis = 0, reduction = None, finite = False, finite_limit = 100):
    if not isNode(input_) or not isNode(target):
        raise nour.errors.InputError(f'Inputs expected to be nodes not, {type(input_)} and {type(target)}')
    
    target = target.view(np.ndarray)
    input_sigmoid = _utils._functional_finite_sigmoid(input_.view(np.ndarray), limit=finite_limit) if finite else _utils.sigmoid(input_.view(np.ndarray))
    log_fn = _utils._finite_log if finite else np.log
    result = -((target * log_fn(input_sigmoid)) + (1 - target) * log_fn(1 - input_sigmoid))

    if not reduction or reduction == 'mean':
        result = nour.node(np.mean(result, axis = axis), requires_grad = False)
    elif reduction == 'sum':
        result = nour.node(np.sum(result, axis = axis), requires_grad = False)
    else:
        raise nour.errors.InputError('reduction must be None, mean or sum not', reduction)
    if (input_.requires_grad or target.requires_grad) and utils.no_grad._grad_mode_:
        result.requires_grad_(True)
        result.input_nodes.extend([input_, target])
        result._node__op = 'binary_cross_entropy_loss'
        result._node__op_parameters['axis'] = axis
        result._node__op_parameters['reduction'] = reduction
        result._node__op_parameters['sigmoid'] = input_sigmoid
        result._node__op_parameters['finite'] = finite
    return result

@utils.LossFunction('cross_entropy_loss')   
def cross_entropy_loss(input_, target, axis = 0, reduction = None, finite = False, finite_limit = 100):
    if not isNode(input_) or not isNode(target):
        raise nour.errors.InputError(f'Inputs expected to be nodes not, {type(input_)} and {type(target)}')
    
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    
    softmax_fun = _utils._functional_finite_softmax if finite else _utils._functional_softmax
    log_fun = _utils._finite_log if finite else np.log
    
    input_softmax = softmax_fun(input_, axis = -1)
    input_logsoftmax = log_fun(softmax_fun(input_, axis = -1))
    grid = np.indices((*input_logsoftmax.shape[:-1], 1))
    result = -input_logsoftmax[*grid[:-1], target.view(np.ndarray)]

    if not reduction or reduction == 'mean':
        result = nour.node(np.mean(result, axis = axis), requires_grad=False)
    elif reduction == 'sum':
        result = nour.node(np.sum(result, axis = axis), requires_grad=False)
    else:
        raise nour.errors.WrongInput('reduction must be None, mean or sum not', reduction)

    if (input_.requires_grad or target.requires_grad) and utils.no_grad._grad_mode_:
        result.requires_grad_(True)
        result.input_nodes.extend([input_, target])
        result._node__op = 'cross_entropy_loss'
        result._node__op_parameters['axis'] = axis
        result._node__op_parameters['reduction'] = reduction
        result._node__op_parameters['softmax'] = input_softmax
        result._node__op_parameters['logsoftmax'] = input_logsoftmax
    return result