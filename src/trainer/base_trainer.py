from abc import abstractmethod

import torch
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH

import os
import logging

logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(
        self,
        model,
        pipe,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        global_config,
        device,
        dataloaders,
        logger,
        writer,
        batch_transforms,
        # trainer args
        max_grad_norm,
        cfg_step,
        log_step,
        n_epochs,
        epoch_len,
        resume_from,
        from_pretrained,
        save_period,
        save_dir,
        seed,
    ):
        self.is_train = True

        self.config = global_config
        self.device = device

        self.logger = logger
        self.log_step = log_step

        self.model = model
        self.pipe = pipe
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms
        self.writer = writer

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = n_epochs

        self.save_period = save_period  # checkpoint each save_period epochs
        self.max_grad_norm = max_grad_norm
        self.cfg_step = cfg_step

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker()
        self.evaluation_metrics = MetricTracker()

        # define checkpoint dir and init everything if required
        self.checkpoint_dir = ROOT_PATH / save_dir / self.config.writer.run_name

        if from_pretrained is not None:
            self._from_pretrained(from_pretrained)

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch)
            raise e

    def _train_process(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # logger.info logged information to the screen
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch):
        logs = {}
        self.is_train = True
        pid = os.getpid()
        self.train_metrics.reset()

        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("general/epoch", epoch)

        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc=f"train_{pid}", total=self.epoch_len)
        ):
            batch["batch_idx"] = batch_idx
            batch = self.process_batch(
                batch,
                train_metrics=self.train_metrics,
            )

            grad_norms = self._get_grad_norms()
            for part_name, part_norm in grad_norms.items():
                self.train_metrics.update(f"grad_norm/{part_name}", part_norm)

            self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Reduced Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )

                lrs = self._get_lrs()
                for part_name, part_lr in lrs.items():
                    self.writer.add_scalar(f"lrs/{part_name}", part_lr)

                self._log_scalars(self.train_metrics, "train")
                self._log_batch(batch, None, batch_idx, "train", epoch)

            if batch_idx + 1 >= self.epoch_len:
                break

        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}/{name}": value for name, value in val_logs.items()})

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        self.is_train = False
        self.evaluation_metrics.reset()

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        for metric in self.metrics:
            metric.to_cuda()

        self.writer.set_step(epoch * self.epoch_len, part)
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_evaluation_batch(
                    batch,
                    eval_metrics=self.evaluation_metrics,
                    is_training_mode=(part == "train_val"),
                )
                self._log_batch(batch, None, batch_idx, part, epoch)
            self._log_scalars(self.evaluation_metrics, part)

        for metric in self.metrics:
            metric.to_cpu()

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.evaluation_metrics.result()

    def _clip_grad_norm(self):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

    @torch.no_grad()
    def _get_grad_norms(self, norm_type=2):
        def compute_params_grad_norm(parameters):
            grad_norms = []
            for p in parameters:
                if p.grad is not None:
                    grad_norms.append(torch.norm(p.grad.detach(), norm_type))

            if not grad_norms:
                return 0.0

            return torch.norm(torch.stack(grad_norms), norm_type).item()

        grad_norms = {}
        for group in self.optimizer.param_groups:
            grad_norms[group["name"]] = compute_params_grad_norm(group["params"])

        # Compute total norm
        total_norm = torch.norm(
            torch.tensor(list(grad_norms.values())), norm_type
        ).item()
        grad_norms["total_norm"] = total_norm
        self.optimizer.zero_grad()

        return grad_norms

    @torch.no_grad()
    def _get_lrs(self):
        lrs = {}
        for last_lr, group in zip(
            self.lr_scheduler.get_last_lr(), self.optimizer.param_groups
        ):
            lrs[group["name"]] = last_lr

        return lrs

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch, output, batch_idx, part, epoch):
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker, part):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(
                f"{part}/{metric_name}", metric_tracker.avg(metric_name)
            )

    def _save_checkpoint(self, epoch):
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.get_state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "config": self.config,
        }

        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")

        torch.save(state, filename)
        logger.info(f"Чекпойнт сохранен: {filename}")

        try:
            lora_weights = self.model.get_state_dict()
            lora_path = str(self.checkpoint_dir / f"lora_weights_epoch{epoch}.pth")
            torch.save(lora_weights, lora_path)
            logger.info(f"LoRA веса сохранены: {lora_path}")
        except Exception as e:
            logger.info(f"Не удалось сохранить LoRA веса отдельно: {e}")

        self.logger.info(f"Saving checkpoint: {filename} ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1

        self.model.load_state_dict_(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path):
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            logger.info(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict_(checkpoint["state_dict"])
        else:
            self.model.load_state_dict_(checkpoint)
