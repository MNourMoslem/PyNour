import numpy as np
from nour.nn import Parameters
import nour

def save(file, parameters):
    np.savez(file, *parameters.values())
    
def load(file, module):
    f = np.load(file)
    params = Parameters(dict(f))
    for i, j in zip(module.parameters() ,params):
        if i.shape != j.shape:
            raise nour.errors.ShapeError('The input parameters must have the same shape of the Module paramters') 
        i[:] = j