import numpy as np

def _functional_sigmoid(a):
    return np.divide(1, np.add(1, np.exp(np.negative(a))))

def _functional_softmax(a, axis = -1):
    shape = list(a.shape)
    shape[axis] = 1
    return (np.exp(a) / np.sum(np.exp(a), axis = axis).reshape(*shape))

def _functional_finite_exp(a, limit = 1e2):
    return np.exp(np.minimum(a, limit))

def _functional_finite_sigmoid(a, limit = 1e2):
    return 1 / (1 + _functional_finite_exp(-a, limit = limit))

def _functional_finite_softmax(a, axis = 0, limit = 1e2):
    shape = list(a.shape)
    shape[axis] = 1
    return np.divide(_functional_finite_exp(a, limit), np.sum(_functional_finite_exp(a, limit), axis = axis).reshape(*shape))

def _finite_log(a, limit = 1e2):
    return np.log(np.maximum(1e-15, np.minimum(a, np.exp(limit))))

def _finite_logsoftmax(a, axis = 0, limit = 1e2):
    return np.log(np.maximum(1e-15, _functional_finite_softmax(a, axis = axis, limit = limit)))

def _finite_division(a, b, limit = 1e-15):
    return np.divide(a, np.maximum(limit, b))