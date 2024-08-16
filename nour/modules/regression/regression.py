import nour
from nour import utils, nn
from nour.utils import *
from nour.errors.errors import *

class LogisticRegression(nn.Module):
    def __init__(self, lr = 1e-1, epochs = 1000, optimizer = 'sgd', loss_fn = 'mse_loss', weight_decay = 0, random_state = None):
        self.optimizer = utils._optimizer_classes_[optimizer]
        self.loss_fn = utils._loss_functions_[loss_fn]
        self.weight_decay = weight_decay
        self.lr = lr 
        self.epochs = epochs
        self.random_state = random_state
        self.__module = None

    def fit(self, x, y):
        self.weight = nour.node(np.random.randn(x.shape[-1], 1), requires_grad = True)
        self.bias = nour.node(np.random.randn(x.shape[-2], 1), requires_grad = True)
        self.__module = lambda x: np.add(np.matmul(x, self.weight), self.bias * np.ones(self.bias.shape))
        optimizer = self.optimizer(nour.nn.Parameters({'weights' : self.weight, 'bias':self.bias}), lr = self.lr, weight_decay = self.weight_decay)
        for _ in range(self.epochs):
            y_hat = self.__module(x)
            loss = self.loss_fn(y_hat, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    def predict(self, x):
        if self.__module:
            return self.__module(x)
        return UnvalidTask('Module can not predict without being trained on any data')        
    
    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)
    
class SVM(nn.Module):
    def __init__(self, lr = 0.1, epochs = 1000, alpha = 0.001, optimizer = 'sgd', loss_fn = 'mse_loss', weight_decay = 0, random_state = None):
        self.optimizer = utils._optimizer_classes_[optimizer]
        self.loss_fn = utils._loss_functions_[loss_fn]
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.lr = lr 
        self.epochs = epochs
        self.random_state = random_state
        self.__module = None
        
    def fit(self, x, y):
        self.weight = nour.node(np.random.randn(x.shape[-1], 1), requires_grad = True)
        self.bias = nour.node(np.random.randn(x.shape[-2], 1), requires_grad = True)
        self.__module = lambda x: np.add(np.matmul(x, self.weight), self.bias * np.ones(self.bias.shape))
        optimizer = self.optimizer(nn.Parameters({'weights' : self.weight, 'bias':self.bias}), lr = self.lr, weight_decay = self.weight_decay)
        for _ in range(self.epochs):
            y_hat = self.__module(x)
            loss = -(nour.norm(self.weight)**2 / 2 + nour.reduce_sum(self.alpha * y_hat))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
    def predict(self, x):
        if self.__module:
            return self.__module(x)
        return UnvalidTask('Module can not predict without being trained on any data')
    
    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)