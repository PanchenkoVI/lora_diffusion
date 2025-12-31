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
    """
    Base class for all trainers.
    """

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
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
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

        # if resume_from is not None:
        #     resume_path = self.checkpoint_dir / resume_from
        #     self._resume_checkpoint(resume_path)

        if from_pretrained is not None:
            self._from_pretrained(from_pretrained)

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        # if self.resume_from is not None:
        #     self._resume_checkpoint(self.resume_from)
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch and evaluating it on non-train partitions
        """
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
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.evaluation_metrics.reset()
        torch.mps.empty_cache()

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

        torch.mps.empty_cache()
        return self.evaluation_metrics.result()

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

    @torch.no_grad()
    def _get_grad_norms(self, norm_type=2):
        """
        Calculates the gradient norm for logging.
        """

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
        """
        Returns lrs for logging.

        Returns:
            lres (dict): last lrs
        """
        lrs = {}
        for last_lr, group in zip(
            self.lr_scheduler.get_last_lr(), self.optimizer.param_groups
        ):
            lrs[group["name"]] = last_lr

        return lrs

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
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
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker, part):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(
                f"{part}/{metric_name}", metric_tracker.avg(metric_name)
            )

    def _save_checkpoint(self, epoch):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
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
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
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
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            logger.info(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict_(checkpoint["state_dict"])
        else:
            self.model.load_state_dict_(checkpoint)
