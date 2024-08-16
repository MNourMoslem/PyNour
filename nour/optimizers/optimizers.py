import numpy as np
from nour import utils, errors
import nour


class Optimizer:
    def __init__(self, parameters):
        self.__parameters = parameters
        
    def step(self):
        if 'function' not in self.__class__.__dict__:
            raise errors.MessingFunctionality('"function" method must be defined to the Optimizer to apply the operations on the trainable nodes')
        self.start()
        with nour.utils.no_grad():
            for key, node_ in self.__parameters.items():
                node_[:] = self.function(node_, key)
        self.end()
                    
    def zero_grad(self):
        for node_ in self.__parameters:
            node_.zero_grad()
        
    @property
    def parameters(self):
        return self.__parameters
    
    def start(self):
        pass
    
    def end(self):
        pass

@utils.OptimizerClass('sgd')    
class SGD(Optimizer):
    def __init__(self, parameters, lr = 1e-3, weight_decay = 0):
        super().__init__(parameters)
        self.lr = lr
        self.weight_decay = weight_decay
        
    def function(self, x, key):
        grad = x.grad if not self.weight_decay else x.grad + self.weight_decay * sum(np.sum(np.asarray(node_)) for node_ in list(self.parameters))
        return x - grad * self.lr

@utils.OptimizerClass('adam') 
class Adam(Optimizer):
    def __init__(self, parameters, lr = 1e-3, beta = (0.9, 0.999), epsilon = 1e-8, weight_decay = 0):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}
        for key, value in self.parameters.items():
            self.m[key] = np.zeros(value.shape)
            self.v[key] = np.zeros(value.shape)
    
    def function(self, x, key):
        grad = x.grad if not self.weight_decay else x.grad + self.weight_decay * sum(np.sum(np.asarray(node_)) for node_ in list(self.parameters))
        self.m[key] = self.beta1 * self.m[key] + (1-self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1-self.beta2) * grad ** 2
        mhat = self.m[key] / (1 - self.beta1**self.t)
        vhat = self.v[key] / (1 - self.beta2**self.t)
        return x - self.lr * mhat / (vhat ** (1/2) + self.epsilon) 
    
    def start(self):
        self.t += 1

        
@utils.OptimizerClass('rms_prop')   
class RMSProp(Optimizer):
    def __init__(self, parameters, lr = 1e-3, beta = 0.9, epsilon = 1e-8, weight_decay = 0):
        super().__init__(parameters)
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.eg2 = {}
        for key, value in self.parameters.items():
            self.eg2[key] = np.zeros(value.shape)
            
    def function(self, x, key):
        grad = x.grad if not self.weight_decay else x.grad + self.weight_decay * sum(np.sum(np.asarray(node_)) for node_ in list(self.parameters))
        self.eg2[key] = self.beta * self.eg2[key] + (1-self.beta) * grad ** 2
        return x - grad * self.lr / (self.eg2[key] + self.epsilon) ** (1/2)
    
    def start(self):
        self.t += 1


@utils.OptimizerClass('nrms_prop')           
class NRMSProp(Optimizer):
    def __init__(self, parameters, lr = 1e-3, beta = (0.9, 0.999), epsilon = 1e-7, weight_decay = 0):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.n = {}
        self.s = {}
        for key, value in self.parameters.items():
            self.n[key] = np.zeros(value.shape)
            self.s[key] = np.zeros(value.shape)
    
    def function(self, x, key):
        grad = x.grad if not self.weight_decay else x.grad + self.weight_decay * sum(np.sum(np.asarray(node_)) for node_ in list(self.parameters))
        self.n[key] = self.beta2 * self.n[key] + (1-self.beta2) * grad
        self.s[key] = self.beta1 * self.s[key] + (1-self.beta1) * (grad - self.n[key]) ** 2
        nhat = self.n[key] / (1 - self.beta2**self.t)
        return x - self.lr * nhat / (self.s[key] + self.epsilon) ** (1/2)
    
    def start(self):
        self.t += 1