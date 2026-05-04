import numpy as np
from typing import Optional, Mapping, Literal

class CallBacks():
    def __init__(self, monitor:str='val_loss', min_delta:float=0, patience:int=0, verbose:bool=True, mode:Literal['auto', 'min', 'max']='auto', start_epoch:int=0) -> None:
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.start_epoch = start_epoch
    
        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater

        else:
            if (self.monitor.endswith("acc") or self.monitor.endswith("accuracy") or self.monitor.endswith("auc")):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
        
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_epoch = 0
        self.stop_training = False
        self.savecheckpoint = False

    
    def update(self, epoch:int, logs:Optional[Mapping[str, float]] = None) -> None:
        current = self.get_monitor_value(logs)
        if current is None or epoch < self.start_epoch:
            # If no monitor value exists or still in initial warm-up stage.
            return
        
        if self.is_improvement(current, self.best):
            print(f'Epoch: [{epoch+1:03d}] | metric `{self.monitor}` improved from {self.best:0.4f} to {current:0.4f}')
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            self.savecheckpoint = True

        else:
            print(f'Epoch: [{epoch+1:03d}] | metric `{self.monitor}` did not improve from {self.best:0.4f}')
            self.wait += 1
            self.savecheckpoint = False

        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.stop_training = True


    def close(self) -> None:
        if self.stopped_epoch > 0 and self.verbose:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")
    
    def get_monitor_value(self, logs:Optional[Mapping[str, float]]=None) -> Optional[float]:
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            print(f"Early stopping conditioned on metric `{self.monitor}` "
                  f"which is not available. Available metrics are: {','.join(list(logs.keys()))}",)
        return monitor_value

    def is_improvement(self, monitor:float, reference:float) -> bool:
        return self.monitor_op(monitor - self.min_delta, reference)