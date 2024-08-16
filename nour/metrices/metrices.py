import numpy as np

def accuracy(target, preds):
    return np.sum(np.equal(target, preds)).item()  / len(preds)