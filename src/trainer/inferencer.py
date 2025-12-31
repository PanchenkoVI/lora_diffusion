import torch
import json
import os
from pathlib import Path
from PIL import Image
import glob
import yaml
import logging
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class BaseInferencer(BaseTrainer):
    def __init__(
        self,
        model,
        pipe,
        metrics,
        global_config,
        device,
        dataloaders,
        logger,
        writer,
        batch_transforms=None,
        epoch_len=1,
        epochs_to_infer=[0],
        ckpt_dir=None,
        exp_save_dir=None,
        seed=42,
        data_config_path="./src/configs/data_source/main_source.yaml",
    ):
        self.is_train = False
        self.model = model
        self.pipe = pipe
        self.device = device
        self.logger = logger
        self.writer = writer
        self.batch_transforms = batch_transforms
        self.config = global_config
        self.epochs_to_infer = epochs_to_infer
        self.ckpt_dir = ckpt_dir
        self.exp_save_dir = exp_save_dir
        self.seed = seed
        self._current_epoch = 0
        self._batch_counter = 0
        self.images_storage = []
        self._is_clearml = hasattr(self.writer, "task")
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }
        self.metrics = metrics
        self.evaluation_metrics = MetricTracker()

        with open(data_config_path, "r") as f:
            data_cfg = yaml.safe_load(f)
        self.concept_dir = Path(
            data_cfg.get("root_dir", "./src/data/my_object_512/my_object_512")
        )

        self.concept_images = self._load_concept_images()

        self._fix_metric_devices()

    def _fix_metric_devices(self):
        for metric in self.metrics:
            try:
                if hasattr(metric, "model"):
                    try:
                        current_device = next(metric.model.parameters()).device
                        if current_device.type != self.device.type:
                            self.logger.info(
                                f"Перемещаем {getattr(metric, 'name', 'unknown')} с {current_device} на {self.device}"
                            )
                            metric.model = metric.model.to(self.device)
                    except StopIteration:
                        self.logger.warning(
                            f"Метрика {getattr(metric, 'name', 'unknown')} не имеет параметров"
                        )

                if hasattr(metric, "device"):
                    metric.device = self.device
            except Exception as e:
                self.logger.warning(
                    f"Ошибка при исправлении устройства метрики {getattr(metric, 'name', 'unknown')}: {e}"
                )

    def _load_concept_images(self):
        images = []
        pattern = str(self.concept_dir / "*.jpg")
        for img_path in glob.glob(pattern):
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception as e:
                self.logger.warning(f"Не удалось загрузить {img_path}: {e}")
        self.logger.info(
            f"Загружено {len(images)} reference изображений из {self.concept_dir}"
        )
        return images

    def inference(self):
        self.logger.info(f"Запуск инференса на эпохах: {self.epochs_to_infer}")

        for epoch in self.epochs_to_infer:
            self._current_epoch = epoch

            if epoch != 0 and self.ckpt_dir is not None:
                ckpt_pth = Path(self.ckpt_dir) / f"checkpoint-epoch{epoch}.pth"
                if ckpt_pth.exists():
                    self.logger.info(f"Загружаем чекпоинт {ckpt_pth}")
                    if not self._from_pretrained(ckpt_pth):
                        self.logger.error(f"Ошибка загрузки эпохи {epoch}")
                        continue
                else:
                    self.logger.warning(f"Чекпоинт не найден: {ckpt_pth}")
                    continue

            self._fix_metric_devices()

            logs = {"epoch": epoch}
            for part, dataloader in self.evaluation_dataloaders.items():
                self._current_part = part
                self.images_storage = []
                part_logs = self._evaluation_epoch(epoch, part, dataloader)
                self.save_results(epoch, part)
                logs.update({f"{part}/{k}": v for k, v in part_logs.items()})

            self.logger.info(f"Результаты эпохи {epoch}: {json.dumps(logs, indent=2)}")
            if self._is_clearml:
                self._send_to_clearml(logs, epoch)

    def _evaluation_epoch(self, epoch, part, dataloader):
        self.evaluation_metrics.reset()
        for metric in self.metrics:
            if hasattr(metric, "reset"):
                metric.reset()
        for batch_idx, batch in enumerate(dataloader):
            if "concept" not in batch or not batch["concept"]:
                batch["concept"] = self.concept_images
            self.process_evaluation_batch(batch, self.evaluation_metrics, batch_idx)

        return self.evaluation_metrics.result()

    @torch.no_grad()
    def process_evaluation_batch(self, batch, eval_metrics, batch_idx=0):
        prompt = batch.get("prompt", "")
        concept_images = batch.get("concept", [])

        epoch = self._current_epoch

        val_args = getattr(self.config, "validation_args", {})
        num_images_per_prompt = getattr(val_args, "num_images_per_prompt", 1)
        num_inference_steps = getattr(val_args, "num_inference_steps", 10)
        guidance_scale = getattr(val_args, "guidance_scale", 7.5)
        height = getattr(val_args, "height", 512)
        width = getattr(val_args, "width", 512)
        negative_prompt = getattr(val_args, "negative_prompt", "")
        seed = getattr(val_args, "seed", 42)

        generator = torch.Generator(device=self.device).manual_seed(
            epoch * 1000 + batch_idx * 10 + seed
        )

        if not hasattr(self, "pipe") or self.pipe is None:
            self.logger.warning("Пайплайн не найден — создаём заново.")
            self.prepare_pipeline()

        try:
            result = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            )
            images = result.images
            batch["generated"] = images
            batch["concept"] = concept_images
            self.store_batch(images, prompt)
            self.logger.info(
                f"Сгенерировано {len(images)} изображений для prompt='{prompt}'"
            )
        except Exception as e:
            self.logger.error(f"Ошибка генерации: {e}")
            import traceback

            traceback.print_exc()
            batch["generated"] = []

        self._process_metrics_with_device_fix(batch, eval_metrics)

        return batch

    def _process_metrics_with_device_fix(self, batch, eval_metrics):
        for metric in self.metrics:
            try:
                metric_value = self._compute_metric_safe(metric, batch)

                if metric_value is not None:
                    if isinstance(metric_value, dict):
                        for k, v in metric_value.items():
                            eval_metrics.update(k, v)
                            if self._is_clearml:
                                self.writer.task.get_logger().report_scalar(
                                    title=k,
                                    series=k,
                                    value=v,
                                    iteration=self._current_epoch,
                                )
                    else:
                        metric_name = getattr(metric, "name", metric.__class__.__name__)
                        eval_metrics.update(metric_name, metric_value)
                        if self._is_clearml:
                            self.writer.task.get_logger().report_scalar(
                                title=metric_name,
                                series=metric_name,
                                value=metric_value,
                                iteration=self._current_epoch,
                            )

            except Exception as e:
                self.logger.error(
                    f"Ошибка в метрике {getattr(metric, 'name', 'unknown')}: {e}"
                )

    def _compute_metric_safe(self, metric, batch):
        try:
            return metric(**batch)
        except RuntimeError as e:
            if "should be the same" in str(e):
                return self._compute_metric_with_device_fix(metric, batch)
            else:
                raise e

    def _compute_metric_with_device_fix(self, metric, batch):
        safe_batch = {}

        for key, value in batch.items():
            if key in ["generated", "concept"] and isinstance(value, list):
                safe_batch[key] = value
            elif isinstance(value, torch.Tensor):
                if hasattr(metric, "model"):
                    try:
                        metric_device = next(metric.model.parameters()).device
                        if value.device != metric_device:
                            safe_batch[key] = value.to(metric_device)
                        else:
                            safe_batch[key] = value
                    except StopIteration:
                        safe_batch[key] = value.cpu()
                else:
                    safe_batch[key] = value.cpu()
            else:
                safe_batch[key] = value

        return metric(**safe_batch)

    def store_batch(self, images, prompt):
        self.images_storage.append((prompt, images))

    def save_results(self, epoch, part):
        output_dir = Path(self.exp_save_dir) / f"checkpoint_{epoch}/{part}"
        os.makedirs(output_dir, exist_ok=True)
        metrics_dict = {"prompts": []}

        for prompt, images_batch in self.images_storage:
            metrics_dict["prompts"].append(prompt)
            batch_dir = output_dir / prompt.replace(" ", "_")[:50]
            os.makedirs(batch_dir, exist_ok=True)
            for i, img in enumerate(images_batch):
                img.save(batch_dir / f"{i}.jpg")

        metrics_dict.update(self.evaluation_metrics.result())
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=2)
        self.logger.info(f"Результаты сохранены: {output_dir}")

    def _send_to_clearml(self, logs, epoch):
        if not self._is_clearml:
            return
        task = self.writer.task

        for key, value in logs.items():
            if isinstance(value, (int, float)):
                task.get_logger().report_scalar(
                    title=key, series=key, value=value, iteration=epoch
                )

        # Отправляем изображения в ClearML
        output_dir = Path(self.exp_save_dir) / f"checkpoint_{epoch}"

        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    file_path = Path(root) / file
                    try:
                        relative_dir = Path(root).relative_to(output_dir)
                        artifact_name = f"epoch_{epoch}_{relative_dir}_{file}"

                        task.upload_artifact(
                            name=artifact_name, artifact_object=file_path
                        )

                        task.get_logger().report_image(
                            title=f"Generated Images Epoch {epoch}",
                            series=artifact_name,
                            local_path=file_path,
                            iteration=epoch,
                        )

                    except Exception as e:
                        self.logger.warning(
                            f"Не удалось загрузить изображение {file_path} в ClearML: {e}"
                        )

        self.logger.info(f"Изображения эпохи {epoch} отправлены в ClearML")

    def _from_pretrained(self, pretrained_path):
        self.logger.info(f"Базовая загрузка чекпоинта: {pretrained_path}")
        return True

    def prepare_pipeline(self):
        self.logger.info("Базовый prepare_pipeline - должен быть переопределен")


class LoraInferencer(BaseInferencer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_created = False

    def _from_pretrained(self, pretrained_path):
        self.logger.info(f"Загрузка чекпоинта: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            state_dict = (
                checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint
            )
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            self._debug_after_loading()
            if not self.pipe_created:
                self._create_pipeline()
            return True
        except Exception as e:
            self.logger.error(f"Ошибка загрузки чекпоинта: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _debug_after_loading(self):
        self.logger.info("=== Диагностика после загрузки ===")
        if hasattr(self.model, "unet"):
            unet_params = sum(p.numel() for p in self.model.unet.parameters())
            self.logger.info(f"UNet params: {unet_params:,}")
        else:
            self.logger.warning("Нет self.model.unet")

        if hasattr(self.model.unet, "peft_config"):
            self.logger.info("UNet LoRA активен")
        else:
            self.logger.warning("UNet LoRA не обнаружен")

        if hasattr(self.model, "text_encoder") and hasattr(
            self.model.text_encoder, "peft_config"
        ):
            self.logger.info("Text Encoder LoRA активен")
        else:
            self.logger.warning("Text Encoder LoRA не обнаружен")

    def _create_pipeline(self):
        self.logger.info("Создание пайплайна для инференса...")
        self.model.to(self.device)

        try:
            if (
                hasattr(self.model, "text_encoder_2")
                and self.model.text_encoder_2 is not None
            ):
                # SDXL
                self.pipe = StableDiffusionXLPipeline(
                    vae=self.model.vae,
                    text_encoder=self.model.text_encoder,
                    text_encoder_2=self.model.text_encoder_2,
                    tokenizer=self.model.tokenizer,
                    tokenizer_2=self.model.tokenizer_2,
                    unet=self.model.unet,
                    scheduler=self.model.noise_scheduler,
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False,
                )
                self.logger.info("Используем SDXL пайплайн")
            else:
                # SD 1.x - БЕЗ text_encoder_2 и tokenizer_2!
                self.pipe = StableDiffusionPipeline(
                    vae=self.model.vae,
                    text_encoder=self.model.text_encoder,
                    tokenizer=self.model.tokenizer,
                    unet=self.model.unet,
                    scheduler=self.model.noise_scheduler,
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False,
                )
                self.logger.info("Используем SD 1.5 пайплайн")
            self.pipe = self.pipe.to(self.device)
            self.pipe_created = True
            self.logger.info("Пайплайн создан и готов к инференсу")

        except Exception as e:
            self.logger.error(f"Ошибка создания пайплайна: {e}")
            import traceback

            traceback.print_exc()

    def prepare_pipeline(self):
        if not self.pipe_created:
            self._create_pipeline()
        else:
            self.logger.info("Пайплайн уже создан, повторное создание не требуется")
