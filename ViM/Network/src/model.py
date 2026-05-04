from typing import Literal, Optional, Mapping
import torch
import models_mamba
import torch.nn as nn

class ViM(nn.Module):
    def __init__(self, 
                 backbone:Literal['vim_small_patch16_stride16_224', 'vim_small_patch16_stride8_224', 'vim_tiny_patch16_stride16_224', 'vim_tiny_patch16_stride8_224'], 
                 num_classes:Optional[Literal[3, 4]]=3, 
                 num_locations:Optional[Literal[4]]=4,  
                 color:Literal['gray', 'color']='color', 
                 freeze:bool=False, 
                 pretrained:bool=True) -> None:
        super().__init__()
        
        self.backbone = models_mamba.create_model(backbone, pretrained=pretrained, in_chans={'gray' : 1, 'color' : 3}.get(color))#, global_pool='avg', num_classes=0)
        # self.backbone.reset_classifier(0, global_pool='avg')
        
        assert num_classes or num_locations, 'This model must return a classfiction head or a localization head'
        self.clshead, self.poshead = None, None
        
        if num_classes:
            self.clshead = nn.Linear(self.backbone.num_features, num_classes)
        
        if num_locations:
            self.poshead = nn.Linear(self.backbone.num_features, num_locations)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        


    def forward(self, x:torch.Tensor) -> Mapping[str, torch.Tensor]:
        x = self.backbone(x)
        
        if self.clshead is not None and self.poshead is None:
            cls = self.clshead(x)
            return {'cls' : cls}
        
        if self.clshead is None and self.poshead is not None:
            pos = self.poshead(x)
            return {'pos' : pos}
        
        cls = self.clshead(x)
        pos = self.poshead(x)
        
        return {'cls' : cls, 'pos' : pos}

