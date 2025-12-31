import pytorch_lightning as pl
import torch
import logging
from typing import Dict, Any
from PIL import Image

logger = logging.getLogger(__name__)

class LoraLightningModule(pl.LightningModule):
    def __init__(self, trainer):
        super().__init__()
        self.trainer_module = trainer
        self._trainer_config = dict(trainer.config) if hasattr(trainer, 'config') else {}
        serializable_config = {
            'n_epochs': getattr(trainer, 'epochs', 10),
            'epoch_len': getattr(trainer, 'epoch_len', 200),
            'log_step': getattr(trainer, 'log_step', 50),
            'save_period': getattr(trainer, 'save_period', 10),
            'max_grad_norm': getattr(trainer, 'max_grad_norm', None),
            'device': str(getattr(trainer, 'device', 'cpu')),
        }
        
        self.save_hyperparameters(serializable_config)
        self._best_loss = float('inf')
        self._best_epoch = 0
        self.automatic_optimization = False
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.trainer_module.process_batch(batch, self.trainer_module.train_metrics)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch_with_idx = batch.copy()
        batch_with_idx['batch_idx'] = batch_idx
        
        processed_batch = self.trainer_module.process_batch(batch_with_idx, self.trainer_module.train_metrics)
        loss = processed_batch['loss']

        self.log('train_loss', loss, 
                on_step=True, on_epoch=True, 
                prog_bar=True, logger=True,
                batch_size=len(batch.get('pixel_values', [])))

        if batch_idx % self.trainer_module.log_step == 0:
            grad_norms = self.trainer_module._get_grad_norms()
            for part_name, part_norm in grad_norms.items():
                self.log(f'train/grad_norm_{part_name}', part_norm,on_step=True, on_epoch=False, logger=True)

            lrs = self.trainer_module._get_lrs()
            for part_name, part_lr in lrs.items():
                self.log(f'train/lr_{part_name}', part_lr,
                        on_step=True, on_epoch=False, logger=True)

            if hasattr(self.trainer_module, 'writer') and self.trainer_module.writer:
                step = self.current_epoch * self.trainer_module.epoch_len + batch_idx
                self.trainer_module.writer.set_step(step)
                self.trainer_module._log_batch(processed_batch, None, batch_idx, 'train', self.current_epoch)

        return loss
    
    def on_train_epoch_end(self):
        train_results = self.trainer_module.train_metrics.result()
        for metric_name, metric_value in train_results.items():
            self.log(f'train_epoch/{metric_name}', metric_value,
                    on_step=False, on_epoch=True, logger=True)

        self.trainer_module.train_metrics.reset()

        if (self.current_epoch + 1) % self.trainer_module.save_period == 0:
            self.trainer_module._save_checkpoint(self.current_epoch + 1)
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:

        if batch_idx == 0:
            self.trainer_module._update_pipeline_with_current_weights(self.current_epoch)

        processed_batch = self.trainer_module.process_evaluation_batch(
            batch,
            self.trainer_module.evaluation_metrics,
            is_training_mode=False
        )

        if 'loss' in processed_batch:
            self.log('val_loss', processed_batch['loss'],
                    on_step=True, on_epoch=True,
                    prog_bar=True, logger=True,
                    batch_size=len(batch.get('pixel_values', [])))

        if hasattr(self.trainer_module, 'writer') and self.trainer_module.writer:
            step = (self.current_epoch + 1) * self.trainer_module.epoch_len
            self.trainer_module.writer.set_step(step, 'val')
            self.trainer_module._log_batch(
                processed_batch, None, batch_idx, 'val', self.current_epoch
            )
        
        return processed_batch
    
    def on_validation_epoch_end(self):
        val_results = self.trainer_module.evaluation_metrics.result()

        for metric_name, metric_value in val_results.items():
            self.log(f'val/{metric_name}', metric_value,
                    on_step=False, on_epoch=True, logger=True)

        current_loss = val_results.get('loss', float('inf'))
        if current_loss < self._best_loss:
            self._best_loss = current_loss
            self._best_epoch = self.current_epoch
            self.log('val/best_loss', self._best_loss,on_step=False, on_epoch=True, logger=True)
            self.trainer_module._save_checkpoint(self.current_epoch + 1)
        self.trainer_module.evaluation_metrics.reset()
    
    def configure_optimizers(self):
        optimizer = self.trainer_module.optimizer
        lr_scheduler = self.trainer_module.lr_scheduler

        if lr_scheduler is not None:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        return optimizer
    
    def train_dataloader(self):
        return self.trainer_module.train_dataloader
    
    def val_dataloader(self):
        val_loaders = []
        for part, loader in self.trainer_module.evaluation_dataloaders.items():
            val_loaders.append(loader)
        
        if len(val_loaders) == 1:
            return val_loaders[0]
        return val_loaders
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint['best_loss'] = self._best_loss
        checkpoint['best_epoch'] = self._best_epoch
        try:
            if hasattr(self.trainer_module.model, 'get_state_dict'):
                checkpoint['model_state'] = self.trainer_module.model.get_state_dict()
        except Exception as e:
            logger.warning(f"Could not save model state: {e}")
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        self._best_loss = checkpoint.get('best_loss', float('inf'))
        self._best_epoch = checkpoint.get('best_epoch', 0)