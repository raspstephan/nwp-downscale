
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import logging
from typing import Any, Callable, Dict, Optional, Tuple

log = logging.getLogger(__name__)

class StopIfNan(Callback):
    def __init__(self,
                 monitor: list = ['loss'],
                 every_n_train_steps: int = 1, 
                 model_save_every_n_train_steps: int = 10000):
        super().__init__()
        self.monitor = monitor
        self._every_n_train_steps = every_n_train_steps
        self._model_save_every_n_train_steps = model_save_every_n_train_steps
    
    
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Stop training if meet the criterion, checked at `every_n_train_steps`"""
        step = trainer.global_step
        skip_batch = ((step + 1) % self._every_n_train_steps != 0) and (max(step - 2, 1) % self._model_save_every_n_train_steps != 0)
        
        if skip_batch:
            return
        logs = trainer.callback_metrics

        if trainer.fast_dev_run:  # short circuit if metric not present
            return
                 
        current = [logs.get(i) for i in self.monitor]
        should_stop = self._evaluate_stopping_criteria(current)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.training_type_plugin.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            print("nan-loss")
            self.stopped_epoch = trainer.current_epoch
            self.stopped_step = trainer.global_step
            self._log_info(trainer, "Nan loss encountered")
        
    def _evaluate_stopping_criteria(self, current):
        for i in current:
            if i and (not torch.isfinite(i)):
                return True
        return False
    
    @staticmethod
    def _log_info(trainer: Optional["pl.Trainer"], message: str) -> None:
        if trainer is not None and trainer.world_size > 1:
            log.info(f"[rank: {trainer.global_rank}] {message}")
        else:
            log.info(message)
            