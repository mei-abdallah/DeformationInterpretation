from typing import Optional, Mapping, Union
import csv
from collections import OrderedDict
from typing import Optional, Sequence, Mapping

class Hyperparameters():
    def __init__(self, checkpointsdir:str, strtime:str, Prefix:str='Net') -> None:
        self.filename = f'{checkpointsdir}/{strtime}/{Prefix}_{strtime}.txt'
    
    def update(self, logs:Mapping[str, Union[int, float, str, tuple, list]]) -> None:
        with open(self.filename, 'a') as file:
            for key , value in logs.items():
                file.write(f'{key} : {value} \n')
            file.write(f'\n')  
            file.flush()

class Logger:
    """Class to log training progress"""
    def __init__(self, checkpointsdir:str, strtime:str, Prefix:str='Net') -> None:
        self.filename = f'{checkpointsdir}/{strtime}/{Prefix}_results.txt'
        # create a new hyperparameter logger object and write the header of columns into it's text file

    def update(self, logs:Mapping[str, float]) -> None:
        with open(self.filename, 'a') as file:
            for key , value in logs.items():
                file.write(f'{key} : {value} \n')
            file.write(f'\n')  
            file.flush()


class CSVLogger():
    def __init__(self, checkpointsdir:str, strtime:str, prefix:str='Net', keys:Optional[Sequence[str]]=None) -> None:
        self.csvfile = open(f'{checkpointsdir}/{strtime}/{prefix}_{strtime}.csv', mode='a', newline='') 
        self.writer = None
        self.addheader = True
        self.keys = keys
    
    def update(self, epoch:int, logs:Mapping[str, float]) -> None:
        if self.keys is None:
            self.keys = sorted(logs.keys())
        
        if not self.writer:
            fieldnames = ['epoch'] + self.keys
            self.writer = csv.DictWriter(self.csvfile, fieldnames)

            if self.addheader:
                self.writer.writeheader()
        
        data = OrderedDict({'epoch' : epoch}) 
        data.update((key, logs[key]) for key in self.keys)

        self.writer.writerow(data)
        self.csvfile.flush()

    def close(self) -> None:
        self.csvfile.close()
        self.writer = None