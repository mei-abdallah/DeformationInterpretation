from typing import Literal, Tuple, Optional, Mapping
import torch
import os
import numpy as np
import bz2, pickle
from glob import glob


class Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                 datadir:str, 
                 phase:Literal['train', 'valid', 'test']='train', 
                 mode:Literal['wrapped', 'unwrapped']='unwrapped',
                 color:Literal['gray', 'color']='color',
                 num_classes:Optional[Literal[3, 4]]=3, 
                 num_locations:Optional[Literal[4]]=4,
                 scale:Optional[Mapping[str, float]]={'min' : -1.0, 'max' : 1.0}) -> None:
        super().__init__()
        self.files = glob(f'{datadir}/{phase}/*/*.pkl')
        self.mode = mode
        self.scale = scale
        self.color = color
        self.clssize, self.possize = None, None

        if num_classes:
            self.clssize = (num_classes,)
            assert len(os.listdir(f'{datadir}/{phase}')) == num_classes, "there is an error in the number of classes"
            
        if num_locations:
            self.possize = (num_locations,)

        

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx:int) -> Mapping[str, torch.Tensor]:
        data = self.open(self.files[idx])
        return {key : self.totensor(value) for key, value in data.items()}


    def open(self, filename:str) -> Mapping[str, np.ndarray]:
        with bz2.BZ2File(f"{filename}", "rb") as file:
            indata = pickle.load(file)
        file.close()

        outdata = {'ifg' : self.tocolor(self.toscale(indata[self.mode]))}

        if self.clssize is not None:
            label = np.zeros(self.clssize)
            label[indata['label']] = 1
            outdata['cls'] =  np.array(label, dtype=np.float32)
        
        if self.possize is not None:
            outdata['pos'] = np.array(indata['loc'], dtype=np.float32)

        return outdata
    
    def toscale(self, data:np.ndarray) -> np.ndarray:
        data = data - np.ma.min(data)
        data = data / np.ma.max(data)
        data = (data * (self.scale['max'] - self.scale['min'])) + self.scale['min']
        data = data.filled(fill_value=0.0)
        return data  

    def tocolor(self, data:np.ndarray) -> np.ndarray:
        data = np.expand_dims(data, axis=0)

        if self.color == 'color':
            data = np.concatenate((data, data, data), axis=0)

        return data                                                                         

    def totensor(self, data:np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data).float()



