from typing import Tuple, Literal, Optional, Mapping
from torch.utils.data import DataLoader
import torch, time, random, numpy as np
from .dataset import Dataset
from .network import Network
from .losses import CLSLoss, POSLoss
from .metrics import Metric, Accurcy, IoU, MAE
from .checkpoint import CheckPoint
from .logger import CSVLogger, Hyperparameters
from .callbacks import CallBacks
from .progbar import ProgBar


class Trainer:
    metrics:Mapping[str, Metric]
    def __init__(self,
                 datadir:str,
                 backbone:Literal['resnet18', 'resnet34', 'resnet50']='resnet18', 
                 mode:Literal['wrapped', 'unwrapped']='unwrapped',
                 checkpointsdir:str='./CheckPoints', 
                 strtime:Optional[str]=None,
                 num_classes:Optional[Literal[3, 4]]=3, 
                 num_locations:Optional[Literal[4]]=4,  
                 color:Literal['gray', 'color']='color', 
                 freeze:bool=False, 
                 pretrained:bool=True,
                 lr:float=2e-4, 
                 betas:Tuple[float, float]=(0.5, 0.999), 
                 device:str="cuda" if torch.cuda.is_available() else "cpu", 
                 ngpus:int=torch.cuda.device_count(),
                 batch_size:int=32,
                 num_workers:int=4,
                 seed:int=42, **kwargs) -> None:
        
        self.setseed(seed)
        strtime = strtime or time.strftime('%Y_%m_%d_%H_%M_%S')
        print(f'seed was set manually to [{seed}]')
        print(f'start time: {strtime}')
        print(f'attached Devices: {device}')

        self.net = Network(backbone, num_classes, num_locations, color, freeze, pretrained, lr, betas, ngpus).to(device)

        self.trainloader = DataLoader(Dataset(datadir, 'train', mode, color, num_classes, num_locations), 
                                      batch_size=batch_size * (1 if not torch.cuda.is_available() else ngpus), 
                                      shuffle=True, 
                                      num_workers=num_workers * (1 if not torch.cuda.is_available() else ngpus), 
                                      pin_memory=True,
                                      drop_last=True)
        self.validloader = DataLoader(Dataset(datadir, 'valid', mode, color, num_classes, num_locations), 
                                      batch_size=batch_size * (1 if not torch.cuda.is_available() else ngpus), 
                                      shuffle=False, 
                                      num_workers=num_workers * (1 if not torch.cuda.is_available() else ngpus), 
                                      pin_memory=True,
                                      drop_last=True,)
        
        self.checkpoint = CheckPoint(checkpointsdir, strtime)
        self.logger = CSVLogger(checkpointsdir, strtime, self.net.getName())
        self.hyperparameters = Hyperparameters(checkpointsdir, strtime, self.net.getName())

        self.clsloss = CLSLoss() if num_classes else None
        self.posloss = POSLoss() if num_locations else None

        self.metrics = {**({'loss'         : Metric(), 
                            'val_loss'     : Metric()}), 
                        **({'cls_loss'     : Metric(),
                            'val_cls_loss' : Metric(),
                            'acc'          : Accurcy(),
                            'val_acc'      : Accurcy(),} if num_classes else {}),
                        **({'pos_loss'     : Metric(),
                            'val_pos_loss' : Metric(),
                            'iou'          : IoU(),
                            'val_iou'      : IoU(),
                            'mae'          : MAE(),
                            'val_mae'      : MAE()} if num_locations else {})}
        
        self.callback = CallBacks(monitor='val_loss', patience=10, min_delta=1e-4)

        self.hyperparameters.update({'strtime'    : strtime,
                                     'datadir'    : datadir,
                                     'mode'       : mode,
                                     'backbone'   : backbone,
                                     'classes'    : num_classes,
                                     'locations'  : num_locations,
                                     'color'      : color,
                                     'freeze'     : freeze,
                                     'pretrained' : pretrained,
                                     'lr'         : lr,
                                     'betas'      : betas,
                                     'batchsize'  : batch_size,
                                     'workers'    : num_workers,
                                     'device'     : device, 
                                     'ngpus'      : ngpus})
        
    

    def setseed(self, seed:int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load(self) -> None:
        # Load the model from checkpoint
        self.net.load(self.checkpoint)

    def save(self) -> None:
        # Save current state of network to disk for later use
        self.net.save(self.checkpoint)


    def fit(self, epochs:int=100, lambdaCLSQuality:float=1.0, lambdaPOSQuality:float=1.0) -> None:
        for epoch in range(epochs):
            # training
            self.train(epoch + 1, lambdaCLSQuality, lambdaPOSQuality)

            # evaluating
            self.valid(epoch + 1, lambdaCLSQuality, lambdaPOSQuality)

            logs = {key : self.metrics[key].result() for key in self.metrics.keys()}

            self.net.scheduler.step(logs['val_loss'])

            self.logger.update(epoch, logs)
            self.callback.update(epoch, logs)

            print(f"Epoch: [{epoch+1:03d}/{epochs}] | {', '.join(f'{key} = {value:0.4f}' for key, value in logs.items())}")

            if self.callback.savecheckpoint:
                self.save()
                self.test(epoch + 1)

            for key in self.metrics.keys():
                self.metrics[key].reset() 


            if self.callback.stop_training:
                break

        self.hyperparameters.update({'epochs'    : epochs,
                                     'clsweight' : lambdaCLSQuality,
                                     'posweight' : lambdaPOSQuality,})
        
        self.logger.close()
        self.callback.close()


    def train(self, epoch:int, lambdaCLSQuality:float=1.0, lambdaPOSQuality:float=1.0) -> None:
        print(f'Training epoch: {str(epoch).zfill(3)}')
        progbar = ProgBar(len(self.trainloader), width=20, stateful_metrics=('epoch', 'loss', 'cls_loss', 'pos_loss', 'acc', 'iou', 'mae'))
        self.net.train()
        
        for inputs in self.validloader:
            logs = [("epoch", epoch)]
            loss = 0.0
            inputs = self.todevice(inputs)
            outputs = self.net.process(inputs['ifg'])
            
            if self.clsloss is not None:
                cls_loss = self.clsloss(outputs['cls'], inputs['cls'])
                loss += cls_loss * lambdaCLSQuality
                logs += [("cls_loss", self.metrics['cls_loss'].update(cls_loss)),
                         ("acc", self.metrics['acc'].update(outputs['cls'], inputs['cls']))]
            
            if self.posloss is not None:
                pos_loss = self.posloss(outputs['pos'], inputs['pos'])
                loss += pos_loss * lambdaPOSQuality
                logs += [("pos_loss", self.metrics['pos_loss'].update(pos_loss)),
                         ("iou", self.metrics['iou'].update(outputs['pos'], inputs['pos'])),
                         ("mae", self.metrics['mae'].update(outputs['pos'], inputs['pos']))]
                
            self.net.backward(loss)
            
            logs.append(("loss", self.metrics['loss'].update(loss)))
            
            progbar.add(1, logs)

    def valid(self, epoch:int, lambdaCLSQuality:float=1.0, lambdaPOSQuality:float=1.0) -> None:
        print(f'Validating epoch: {str(epoch).zfill(3)}')
        progbar = ProgBar(len(self.validloader), width=20, stateful_metrics=('epoch', 'val_loss', 'val_cls_loss', 'val_pos_loss', 'val_acc', 'val_iou', 'val_mae'))
        self.net.eval()
        
        for inputs in self.validloader:
            logs = [("epoch", epoch)]
            val_loss = 0.0
            inputs = self.todevice(inputs)
            outputs = self.net.predict(inputs['ifg'])

            if self.clsloss is not None:
                val_cls_loss = self.clsloss(outputs['cls'], inputs['cls'])
                val_loss += val_cls_loss * lambdaCLSQuality
                logs += [("val_cls_loss", self.metrics['val_cls_loss'].update(val_cls_loss)),
                         ("val_acc", self.metrics['val_acc'].update(outputs['cls'], inputs['cls']))]
            
            if self.posloss is not None:
                val_pos_loss = self.posloss(outputs['pos'], inputs['pos'])
                val_loss += val_pos_loss * lambdaPOSQuality
                logs += [("val_pos_loss", self.metrics['val_pos_loss'].update(val_pos_loss)),
                         ("val_iou", self.metrics['val_iou'].update(outputs['pos'], inputs['pos'])),
                         ("val_mae", self.metrics['val_mae'].update(outputs['pos'], inputs['pos']))]
            
            logs.append(("val_loss", self.metrics['val_loss'].update(val_loss)))
            
            progbar.add(1, logs)

    def test(self, epoch:int) -> None: ...

    def todevice(self, kwargs:Mapping[str, torch.Tensor], device="cuda" if torch.cuda.is_available() else "cpu") -> Mapping[str, torch.Tensor]:
        return {key : value.to(device) for key, value in kwargs.items()}