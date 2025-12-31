import os
import warnings
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import logging
import pytorch_lightning as pl
from pathlib import Path
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.data.data_manager import data_manager 
from src.data.data_manager import data_manager
from src.utils.io_utils import ROOT_PATH
from src.lightning.lora_module import LoraLightningModule
from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"

import os
import requests
import zipfile
from io import BytesIO
from pathlib import Path
from dvc.repo import Repo

@hydra.main(version_base=None, config_path="src/configs", config_name="persongen_train_lora")
def main(config):
    data_path = Path("./src/data/my_object_512")
    if not data_path.exists() or not any(data_path.iterdir()):
        logger.error(f"Данные не найдены в {data_path}")
        logger.error("Пожалуйста, скачайте данные с помощью: python -m src.data.data_manager")
        return
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    
    writer = None
    if hasattr(config, 'writer') and config.writer:
        try:
            writer = instantiate(config.writer, logger, project_config)
        except Exception as e:
            logger.warning(f"Could not initialize writer: {e}")

    device = torch.device(config.trainer.device)
    dataloaders, batch_transforms = get_dataloaders(config, device, logger)

    model = instantiate(config.model, device=device)
    model.prepare_for_training()
    loss_function = instantiate(config.loss_function).to(device)

    metrics = []
    for metric_name in config.inference_metrics:
        metric_config = config.metrics[metric_name]
        metrics.append(instantiate(metric_config, name=metric_name, device=device))

    trainable_params = model.get_trainable_params(config)
    optimizer = instantiate(config.optimizer, params=trainable_params)

    for i, group in enumerate(optimizer.param_groups):
        logger.info(f"Param group <{group.get('name', f'group_{i}')}>:")
        logger.info(f"  learning rate: {group['lr']}")
        logger.info(f"  weight decay:  {group['weight_decay']}")
        logger.info(f"  num params:    {len(group['params'])}")

    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    pipeline = instantiate(
        config.pipeline,
        model=model,
        device=None,
        device_map="auto",
        torch_dtype=torch.float32,
    )

    trainer_instance = instantiate(
        config.trainer,
        model=model,
        pipe=pipeline,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        global_config=config,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        _recursive_=False,
    )

    lightning_module = LoraLightningModule(trainer_instance)

    checkpoint_dir = ROOT_PATH / config.trainer.save_dir / config.writer.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = []
    
    checkpoint_config = config.get('checkpoint', {})
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=checkpoint_config.get('filename', 'checkpoint-{epoch:02d}-{val_loss:.2f}'),
        monitor=checkpoint_config.get('monitor', 'val/loss'),
        mode=checkpoint_config.get('mode', 'min'),
        save_top_k=checkpoint_config.get('save_top_k', 3),
        save_last=checkpoint_config.get('save_last', True),
        every_n_epochs=checkpoint_config.get('every_n_epochs', config.trainer.save_period),
    )
    callbacks.append(checkpoint_callback)
    
    lr_monitor = LearningRateMonitor(logging_interval=config.get('lr_monitor', {}).get('logging_interval', 'step'))
    callbacks.append(lr_monitor)

    pl_logger = None
    logger_config = config.get('lightning_logger', {})
    
    if writer is not None:
        if hasattr(writer, '__class__') and 'TensorBoard' in str(writer.__class__):
            pl_logger = TensorBoardLogger(
                save_dir=logger_config.get('save_dir', str(checkpoint_dir)),
                name=logger_config.get('name', 'lightning_logs'),
                version=logger_config.get('version', None),
            )
        elif hasattr(writer, '__class__') and 'WandB' in str(writer.__class__):
            pl_logger = WandbLogger(
                project=logger_config.get('project', project_config.get('project_name', 'lora_train')),
                name=logger_config.get('name', project_config.get('run_name', 'experiment_1')),
                save_dir=logger_config.get('save_dir', str(checkpoint_dir)),
            )

    epoch_len = config.trainer.get('epoch_len', 200)
    logger.info(f"Using epoch_len: {epoch_len}")
    lightning_config = config.get('lightning_trainer', {})
    
    pl_trainer = pl.Trainer(
        max_epochs=config.trainer.n_epochs,
        limit_train_batches=epoch_len,
        limit_val_batches=lightning_config.get('limit_val_batches', 1.0),
        devices=lightning_config.get('devices', 1),
        accelerator=lightning_config.get('accelerator', 'cpu' if str(device) == 'cpu' else 'auto'),
        logger=pl_logger,
        callbacks=callbacks,
        log_every_n_steps=config.trainer.log_step,
        check_val_every_n_epoch=lightning_config.get('check_val_every_n_epoch', 1),
        enable_progress_bar=lightning_config.get('enable_progress_bar', True),
        enable_model_summary=lightning_config.get('enable_model_summary', True),
        accumulate_grad_batches=lightning_config.get('accumulate_grad_batches', 1),
        gradient_clip_val=config.trainer.max_grad_norm,
        deterministic=lightning_config.get('deterministic', False),
        benchmark=lightning_config.get('benchmark', True),
        precision=lightning_config.get('precision', '32-true'),
        default_root_dir=str(checkpoint_dir),
        fast_dev_run=lightning_config.get('fast_dev_run', False),
        overfit_batches=lightning_config.get('overfit_batches', 0.0),
        val_check_interval=lightning_config.get('val_check_interval', 1.0),
    )

    pl_trainer.fit(
        lightning_module,
        train_dataloaders=lightning_module.train_dataloader(),
        val_dataloaders=lightning_module.val_dataloader(),
        ckpt_path=config.trainer.resume_from if config.trainer.resume_from else None
    )

    logger.info("<============| Training completed |============>")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-download', action='store_true', help='Пропустить скачивание данных')
    args = parser.parse_args()
    
    if not args.skip_download:
        success = data_manager.download_data()
        if not success:
            print("Не удалось скачать данные. Завершение работы.")
            exit(1)
    else:
        print("Скачивание данных пропущено")
    main()