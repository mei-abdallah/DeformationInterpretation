import torch

class Metric():
    def __init__(self) -> None:
        self.total = torch.tensor([0.0])
        self.count = torch.tensor([0.0])

    def update(self, values:torch.Tensor) -> float:
        
        if not len(values.size()):
            values = torch.tensor([values.item()])
        
        self.count += values.size(0)
        self.total += values.sum(0)

        return values.mean().item()

    def result(self,)-> float:
        return (self.total/self.count).item()

    def reset(self,) -> None:
        self.total = torch.tensor([0.0])
        self.count = torch.tensor([0.0])

class Accurcy(Metric):
    def update(self, outputs:torch.Tensor, targets:torch.Tensor) -> float:
        outputs = torch.argmax(outputs, dim=1, keepdim=True).squeeze()
        targets = torch.argmax(targets, dim=1, keepdim=True).squeeze()
        values = (outputs == targets.long()).double()
        return super().update(values)


class IoU(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.epsilon = 1e-6

    def update(self, outputs:torch.Tensor, targets:torch.Tensor) -> float:
        outputs = self.anchortobox(outputs)
        targets = self.anchortobox(targets)

        ixmax = torch.max(targets[...,0], outputs[...,0])
        ixmin = torch.min(targets[...,1], outputs[...,1])

        iymax = torch.max(targets[...,2], outputs[...,2])
        iymin = torch.min(targets[...,3], outputs[...,3])

        # Intersection height and width.
        iheight = torch.max(torch.tensor([0.]), iymin - iymax)
        iwidth = torch.max(torch.tensor([0.]), ixmin - ixmax)

        # Ground Truth dimensions.
        theight = targets[...,3] - targets[...,2] 
        twidth = targets[...,1] - targets[...,0]

        # Prediction dimensions
        pheight = outputs[...,3] - outputs[...,2]
        pwidth = outputs[...,1] - outputs[...,0]
        values = torch.divide(iheight * iwidth + self.epsilon,
                             (theight * twidth + pheight * pwidth - iheight * iwidth + self.epsilon))
        return super().update(values)

    def anchortobox(self, anchor:torch.Tensor) -> torch.Tensor:
        xstart = anchor[...,0] - anchor[..., 2]
        xstop = anchor[...,0] + anchor[..., 2]
        ystart = anchor[...,1] - anchor[..., 3]
        ystop = anchor[...,1] + anchor[..., 3]
        return torch.stack([xstart, xstop, ystart, ystop], axis=-1)
    
class MAE(Metric):
    def update(self, outputs:torch.Tensor, targets:torch.Tensor) -> float:
        values = torch.abs((targets - outputs)).mean(dim=1)
        return super().update(values)