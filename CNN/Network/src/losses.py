from typing import Literal
import torch
import torch.nn as nn


class CLSLoss:
    def __init__(self) -> None:
        self.loss = nn.CrossEntropyLoss()
    
    def __call__(self, outputs:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
        loss = self.loss(outputs, targets)
        return loss
    

class POSLoss:
    def __init__(self) -> None:
        self.loss = nn.MSELoss()
    
    def __call__(self, outputs:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
        loss = self.loss(outputs, targets)
        return loss