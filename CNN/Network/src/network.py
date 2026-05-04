from typing import Tuple, Literal, Optional, Mapping
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
from .model import CNN

class Network(nn.Module):
    def __init__(self,
                 backbone:Literal['resnet18', 'resnet34', 'resnet50'], 
                 num_classes:Optional[Literal[3, 4]]=3, 
                 num_locations:Optional[Literal[4]]=4,  
                 color:Literal['gray', 'color']='color', 
                 freeze:bool=False, 
                 pretrained:bool=True,
                 lr:float=2e-4, 
                 betas:Tuple[float, float]=(0.5, 0.999), 
                 ngpus:int=1) -> None:
        super().__init__()
        
        self.model = CNN(backbone, num_classes, num_locations, color, freeze, pretrained)
        
        if ngpus > 1:
            self.model = nn.DataParallel(self.model, list(range(ngpus)))

        self.optim = optim.Adam(params=self.model.parameters(),
                                       lr=lr,
                                       betas=betas)
        
        self.scheduler = scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.1, patience=5, verbose=True)

    def forward(self, ifgs:torch.Tensor) -> Mapping[str, torch.Tensor]:
        outputs = self.model(ifgs)
        return outputs
    
    def process(self, ifgs:torch.Tensor) -> Mapping[str, torch.Tensor]:
        self.optim.zero_grad() # set_to_none=True here can modestly improve performance
        outputs = self(ifgs)
        return outputs
    
    def predict(self, ifgs:torch.Tensor) -> Mapping[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self(ifgs)
        return outputs

    def backward(self, loss:torch.Tensor) -> None:
        loss.backward()
        self.optim.step()

    def save(self, checkpointdir:str) -> None:
        filedir = f'{checkpointdir}/{self.getName()}.pth'
        print(f'Saving checkpoint to {filedir}')
        torch.save({
            'model' : self.model.state_dict(),},
            filedir)
        
    def load(self, checkpointdir:str):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        filedir = f'{checkpointdir}/{self.getName()}.pth' 
        print(f'Loading checkpoint from {filedir}')
        if os.path.exists(filedir):
            data = torch.load(filedir, device)
            self.model.load_state_dict({key.replace('module.', '') : value for key, value in data['model'].items()})
        else:
            print(f'No saved model found under {filedir}.')

    def getName(self) -> str:
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.__class__.__name__
        return self.model.__class__.__name__