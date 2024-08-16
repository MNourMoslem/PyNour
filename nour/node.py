import numpy as np
from nour.errors import UnvalidTask
from nour.utils import no_grad, GradientFunction

class Grad(np.ndarray):
    pass

class node(np.ndarray):
    def __new__(cls, array, dtype = None, requires_grad = False):
        if np.isscalar(array) or (not array.shape if (type(array) == int or type(array) == float or isinstance(array, node)) else False): array = [array]
        obj = np.asarray(array, dtype = dtype).view(cls)
        obj.__requires_grad = requires_grad
        obj.__grad = np.zeros((obj.shape)).view(Grad)
        obj.__op = None
        obj.__op_parameters = {}
        obj.__input_nodes = []
        obj.__repr_mode__ = False
        return obj
    
    def  __array_finalize__(self, obj):
        if obj is None: return
        if isinstance(obj, np.ndarray) and not getattr(obj, 'shape', None): 
            self = self[None]

        self.__requires_grad = getattr(obj, 'requires_grad', False)
        self.__grad = np.zeros((self.shape)).view(Grad)
        self.__op = None
        self.__op_parameters = {}
        self.__input_nodes = []
        self.__repr_mode__ = False

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, node):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, node):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        info = {}
        if in_no:
            info['inputs'] = in_no
        if out_no:
            info['outputs'] = out_no
        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        for idx in in_no:
            if results is NotImplemented:
                return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], node):
                inputs[0].info = info
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(node)
                         if output is None else output)
                        for result, output in zip(results, outputs))
        
        if results and isinstance(results[0], node):
            results[0].__requires_grad = any(getattr(input_, 'requires_grad', None) for input_ in inputs)
            results[0].__grad = np.zeros((results[0].shape)).view(Grad)
            if not no_grad._grad_mode_:
                results[0].__op = None
                results[0].__op_parameters = {}
                results[0].__input_nodes = []
            else:
                results[0].__op = ufunc
                results[0].__op_parameters = kwargs
                results[0].__input_nodes = inputs
            results[0].__repr_mode__ = False
            
        return results[0] if len(results) == 1 else results
    
    def __getitem__(self, idx):
        if getattr(self, '__repr_mode__', True):
            return super().__getitem__(idx)
        if np.isscalar(np.asarray(self)[idx]):
            if type(idx) == int or type(idx) == float: idx = slice(idx, idx+1)
            elif type(idx) == tuple and idx != (-1,): idx = np.s_[(*idx[:-1], slice(idx[-1], idx[-1]+1))]
        result = super().__getitem__(idx)
        try:
            result.__grad = self.__grad[idx]
            result.input_nodes.append(self)
        except:
            pass
        return result
        
    @property
    def requires_grad(self):
        return self.__requires_grad
    
    @property
    def grad(self):
        return self.__grad.view(Grad)
    
    @property
    def op(self):
        return self.__op
    
    @property
    def op_parameters(self):
        return self.__op_parameters
    
    @property
    def input_nodes(self):
        return self.__input_nodes
    
    def _add_grad(self, grad):
        grad = grad.view(np.ndarray)
        try:
            self.__grad += grad
        except:
            grad = _broadcast_grad(self, grad)
            self.__grad += grad
            
    def _set_op(self, op):
        self.__op = op
            
    def zero_grad(self):
        self.__grad *= 0
        
    def ones_grad(self):
        self.zero_grad()
        self.__grad += 1
        
    def reset_all_child_grads(self):
        for node_ in _preorder_traversal(self):
            node_.zero_grad()
        
    def requires_grad_(self, value):
        self.__requires_grad = value
        
    def backward(self):
        if _isScalar(self) or not self.shape:
            nodes = _preorder_traversal(self)
            self.ones_grad()
            for node_ in nodes:
                if isinstance(node_, node):
                    if node_.op:
                        grad_f = GradientFunction._gradeint_functions_[node_.op]
                        grad_f(node_)

        else:
            raise UnvalidTask('Output must be a scalar to proivde backpropagation')
            
    def copy(self):
        return np.array(self).view(node)
    
    def numpy(self):
        return self.view(np.ndarray)
    
    def __repr__(self):
        self.__repr_mode__ = True
        result = super().__repr__()
        self.__repr_mode__ = False
        return result

    def __str__(self):
        try:
            return super.__str__()
        except:
            return self.numpy().__str__()
    
    def __iadd__(self, a):
        return self.__add__(a)
    
    def __isub__(self, a):
        return self.__sub__(a)
    
    def __itruediv__(self, a):
        return self.__truediv__(a)
    
    def __imul__(self, a):
        return self.__mul__(a)
    
    def __imod__(self, a):
        return self.__mod__(a)
    
    def __ifloordiv__(self, a):
        return self.__floordiv__(a)
    
    def __ipow__(self, a):
        return self.__pow__(a)
    
    def __imatmul__(self, a):
        return self.__matmul__(a)
    
    def __iand__(self, a):
        return self.__and__(a)
    
    def __ior__(self, a):
        return self.__or__(a)
    
    def __ixor__(self, a):
        return self.__xor__(a)
    
    def __irshift__(self, a):
        return self.__rshift__(a)
    
    def __ilshift__(self, a):
        return self.__lshift__(a)
    
def _isScalar(a):
    return a.squeeze().shape == (1,) or a.shape == (1,)

def _preorder_traversal(nodes):
    preordered_list = []
    def order_(nodes):
        preordered_list.append(nodes)
        if getattr(nodes, "input_nodes", None) and getattr(nodes, "requires_grad", None):
            for input_node in nodes.input_nodes:
                order_(input_node)
                
    order_(nodes)
    return preordered_list

def _broadcast_grad2(a, grad):
    if not a.shape:
        grad = grad.reshape(())
    
    a_shape = list(a.shape)
    b_shape = list(grad.shape)

    if len(a_shape) != len(b_shape):
        if len(a_shape) > len(b_shape):
            for _ in range(len(a_shape) - len(a_shape)):
                b_shape.insert(0, 1)

        if len(b_shape) > len(a_shape):
            for _ in range(len(b_shape) - len(a_shape)):
                a_shape.insert(0, 1)

    a_shape = np.asarray(a_shape)
    b_shape = np.asarray(b_shape)
    
    a_shape = np.flip(a_shape)
    for idx, i in enumerate(a_shape):
        idx = len(a_shape) - (idx + 1)
        if i == 1:
            grad = np.sum(grad, axis = idx)
            b_shape[idx] = 1
            grad = grad.reshape(*b_shape)
    
    return grad

def _broadcast_grad(a, grad):
    if not a.shape:
        grad = grad.reshape(())

    i = 0
    j = 0

    if len(a.shape) < len(grad.shape):
        while i < len(a.shape) and j < len(grad.shape):
            if a.shape[i] != grad.shape[j]:
                grad = np.sum(grad, axis=i, keepdims=a.shape[i] == 1)
                continue
            i += 1
            j += 1
    else:
        return _broadcast_grad2(a, grad)

    if j < len(grad.shape):
        grad = np.sum(grad, axis = tuple(range(j, len(grad.shape))))

    return grad