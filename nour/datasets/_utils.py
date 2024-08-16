import requests
from tqdm import tqdm
import gzip
import os
import shutil
import nour

class NDataset:

    def __init__(self, data, target, batch_size = 1, shuffle = False, drop_last=False):
        self._init_done = False
        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.__upper = 0
        self.__lower = -self.batch_size
        self.reminder = (self.__len__() % batch_size)
        self.offset = self.__len__() - self.reminder
        self._init_done = True

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item], self.target[item]

    def __iter__(self):
        self.__upper = 0
        self.__lower = -self.batch_size
        return self

    def __next__(self):
        if self.offset >= self.__upper:
            self.__upper += self.batch_size
            self.__lower += self.batch_size
        else:
            if not self.drop_last and len(self.data) > self.__upper:
                self.__upper += self.reminder
                self.__lower += self.batch_size
            else:
                raise StopIteration()

        return self.__getitem__(slice(self.__lower, self.__upper))
    
    def __setattr__(self, name: str, value) -> None:
        if name in {'data', 'batch_size', 'drop_last', 'reminder', 'offset', '_init_done'} and getattr(self, '_init_done', False):
            raise nour.errors.SetAttribute(f'{name} can\'t be reassigned after initialized')
        
        return super().__setattr__(name, value)

    def set_batch_size(self, batch_size, drop_last = False):
        super().__setattr__('batch_size', batch_size)
        super().__setattr__('drop_last', drop_last)
        super().__setattr__('reminder', self.__len__() % batch_size)
        super().__setattr__('offset', self.__len__() - self.reminder)

def _download_gzip(file_dir, file_name, url):
    r = requests.get(url, stream = True)

    if r.status_code != 200:
        return None
    
    total_size_in_bytes = int(r.headers["Content-Length"])
    
    dir_path = os.path.join(file_dir)
    os.makedirs(dir_path, exist_ok = True)
    file_path = dir_path + file_name
    
    with tqdm(total = total_size_in_bytes, unit='iB', unit_scale=True) as pb:
        with open(file_path, 'wb') as file:
            for chunk in r.iter_content(1024):
                pb.update(len(chunk))
                file.write(chunk)
                
    print("Download complete")
    print(f"Extracting {file_name}...")
    with gzip.open(file_path, 'rb') as f_in:
        with open(''.join(file_path.split('.')[:-1]), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("Extracting complete")