from nour.datasets._utils import _download_gzip, NDataset
import os
import gzip
import struct
import numpy as np
import nour
from array import array


class MNIST(NDataset):

    mirrors = [
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]

    resources = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    diractory = 'MNIST\\ra\\'

    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    def __init__(self, root_dir, download = False, train = True, batch_size = 1, shuffle = False, drop_last = False):
        self._init_mnist_done = False
        self.root_dir = os.path.join(self.diractory, root_dir)
        self.train = train

        os.makedirs(self.root_dir, exist_ok=True)

        if not os.path.exists(self.root_dir):
            if download:
                self.download()
            else:
                raise nour.errors.MissingData('Data has not been found. You might like to set `download = True` to be able to dowanload it.')
        
        else:
            if not self._check_all_files():
                if download:
                    self.download()
                else:
                    raise nour.errors.MissingData('Root diractory does exits but data seems missing. You might like to set `download = True` to be able to dowanload it.')

        self._read_data()
        super().__init__(data = self.data, target = self.target, batch_size = batch_size, shuffle=shuffle, drop_last = drop_last)
        self._init_mnist_done = True

    def _check_all_files(self):
        for file_name in self.resources:
            path = os.path.join(self.root_dir, file_name)
            if not os.path.exists(path):
                return False
        return True

    def _read_data(self):
        if self.train:
            images_file, labels_file = self.resources[:2]
        else:
            images_file, labels_file = self.resources[2:]
        
        with gzip.open(os.path.join(self.root_dir, images_file),'r') as fin:
            _, _, row, col = struct.unpack('>IIII', fin.read(16))
            buffer = fin.read()
            images = np.frombuffer(buffer, dtype=np.uint8).reshape(-1, 1, row, col).view(nour.node)

        with gzip.open(os.path.join(self.root_dir, labels_file),'r') as fin:
            struct.unpack('>II', fin.read(8))
            buffer = fin.read()
            labels = np.frombuffer(buffer, dtype=np.uint8).view(nour.node)

        self.data = images
        self.target = labels
            
    def download(self):
        for mirror in self.mirrors:
            for file_name in self.resources:
                file_path = os.path.join(self.root_dir, file_name)
                if not os.path.exists(file_path):
                    url = mirror + file_name
                    _download_gzip(self.root_dir, file_name = file_name, url = url)

    def __setattr__(self, name: str, value) -> None:
        if name in {'root_dir', 'train', '_init_mnist_done'} and getattr(self, '_init_mnist_done', False):
            raise nour.errors.SetAttribute(f'{name} can\'t be reassigned after initialized')
        
        return super().__setattr__(name, value)

    def __repr__(self):
        split = 'Train' if self.train else 'Test'
        return f'Dataset {self.__class__.__name__}:\n  Split : {split}\n  Length : {self.__len__()}'