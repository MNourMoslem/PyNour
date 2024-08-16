import numpy as np
import nour

def make_groups(n_samples, n_classes = 2, distance = 2, noise = 0.5, r = 0.5, slope = np.pi / 4, center = (0, 0)):
    degreeses = np.linspace(0, 2*np.pi, int(n_samples/n_classes))
    classes = []
    data = []
    
    for i in range(n_classes):
        x = np.cos(degreeses)*r + np.random.uniform(-noise, noise, degreeses.shape) + np.cos(slope) * (i*distance) + center[0]
        y = np.sin(degreeses)*r + np.random.uniform(-noise, noise, degreeses.shape) + np.sin(slope) * (i*distance) + center[1]
        
        data.append(np.concatenate((x.reshape(len(x), 1), y.reshape(len(y), 1)), axis = 1))
        classes.append(np.ones(degreeses.shape)*i)

    classes = np.concatenate(classes)
    data = np.concatenate(data)
        
    return nour.node(data, requires_grad = False), nour.node(classes.reshape(-1, 1).astype(int), dtype = np.int64, requires_grad = False)

def make_circles(n_samples, n_classes = 2, noise = 0.4, r = 1, center = (0, 0)):
    degreeses = np.linspace(0, 2*np.pi, int(n_samples/n_classes))
    classes = []
    data = []
    
    for i in range(n_classes):
        x = np.cos(degreeses)*(r+i*r) + np.random.uniform(-noise, noise, degreeses.shape) + center[0]
        y = np.sin(degreeses)*(r+i*r) + np.random.uniform(-noise, noise, degreeses.shape) + center[1]
        
        data.append(np.concatenate((x.reshape(len(x), 1), y.reshape(len(y), 1)), axis = 1))
        classes.append(np.ones(degreeses.shape)*i)

    classes = np.concatenate(classes)
    data = np.concatenate(data)
        
    return nour.node(data, requires_grad = False), nour.node(classes.reshape(-1, 1).astype(int), dtype = np.int64, requires_grad = False)
