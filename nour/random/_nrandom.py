from numpy.random import PCG64

class NRandom:
    def __init__(self):
        self._generator = PCG64()

    @property
    def generator(self):
        return self._generator
    
    def set_seed(self, seed):
        self._generator = PCG64(seed = seed)

    def random_raw(self, size = None):
        return self._generator.random_raw(size=size)
    
Gen = NRandom()

