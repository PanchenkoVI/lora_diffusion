import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torchvision.transforms as T

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class LoraTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = getattr(self, "logger", logger)

        # best tracking
        self._best_loss: float = float("inf")
        self._best_epoch: int = 0

        # storage / bookkeeping
        self.images_storage: List[Path] = []
        self._last_saved_weights = None
        self._batch_counter = 0
        self._current_epoch = 0

        # writer / clearml detection
        self.writer = getattr(self, "writer", None)
        self._clearml_task = None
        if self.writer is not None and hasattr(self.writer, "task"):
            try:
                self._clearml_task = getattr(self.writer, "task")
            except Exception:
                self._clearml_task = None
        else:
            # try to get ClearML Task.current_task() if clearml installed and used without wrapper
            try:
                from clearml import Task

                try:
                    task = Task.current_task()
                    if task is not None:
                        self._clearml_task = task
                except Exception:
                    self._clearml_task = None
            except Exception:
                self._clearml_task = None

        # convenience boolean
        self._is_clearml = self._clearml_task is not None

        # save dir
        if not hasattr(self, "save_dir") or self.save_dir is None:
            self.save_dir = Path("saved/default_run")
        else:
            self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _report_scalar(self, title: str, series: str, value: float, iteration: int):
        try:
            if self._clearml_task is not None:
                try:
                    self._clearml_task.get_logger().report_scalar(
                        title=title,
                        series=series,
                        value=float(value),
                        iteration=int(iteration),
                    )
                except Exception as e:
                    if hasattr(self.writer, "report_scalar"):
                        try:
                            self.writer.report_scalar(
                                title=title,
                                series=series,
                                value=float(value),
                                iteration=int(iteration),
                            )
                        except Exception:
                            self.logger.debug(
                                f"_report_scalar: fallback writer.report_scalar failed: {e}"
                            )
                    else:
                        self.logger.debug(
                            f"_report_scalar: clearml report_scalar failed: {e}"
                        )
            else:
                if self.writer is not None and hasattr(self.writer, "add_scalar"):
                    try:
                        tag = f"{title}/{series}" if title else series
                        self.writer.add_scalar(tag, float(value), int(iteration))
                    except Exception as e:
                        self.logger.debug(
                            f"_report_scalar: writer.add_scalar failed: {e}"
                        )
        except Exception as e:
            self.logger.debug(f"Не удалось отправить scalar ({title}/{series}): {e}")

    def _train_epoch(self, epoch: int):
        self.logger.info(f"===== EPOCH {epoch} =====")
        self._debug_trainable_params(epoch)
        result = super()._train_epoch(epoch)
        self._save_epoch_checkpoint(epoch)
        self.logger.info(f"ЗАВЕРШЕНИЕ ЭПОХИ {epoch}")
        return result

    def _debug_trainable_params(self, epoch: int) -> bool:
        total_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        if total_trainable == 0:
            self.logger.error("НЕТ ОБУЧАЕМЫХ ПАРАМЕТРОВ!")
            return False
        self.logger.debug(f"Trainable params: {total_trainable} (epoch {epoch})")
        return True

    def _gather_metrics_results(self) -> Dict[str, float]:
        results: Dict[str, float] = {}
        try:
            for metric in getattr(self, "metrics", []) or []:
                for finalize_name in ("finalize", "compute", "get_result", "aggregate"):
                    if hasattr(metric, finalize_name) and callable(
                        getattr(metric, finalize_name)
                    ):
                        try:
                            getattr(metric, finalize_name)()
                        except Exception:
                            pass
                metric_data = {}
                if (
                    hasattr(metric, "_data")
                    and isinstance(metric._data, dict)
                    and metric._data
                ):
                    metric_data = dict(metric._data)
                else:
                    if hasattr(metric, "value"):
                        metric_data = {
                            getattr(metric, "name", "metric"): getattr(metric, "value")
                        }
                    elif hasattr(metric, "get_value") and callable(
                        getattr(metric, "get_value")
                    ):
                        try:
                            v = metric.get_value()
                            metric_data = {getattr(metric, "name", "metric"): v}
                        except Exception:
                            metric_data = {}
                for k, v in metric_data.items():
                    try:
                        if v is None:
                            continue
                        if isinstance(v, (list, tuple)):
                            nums = [float(x) for x in v if self._is_number_like(x)]
                            if len(nums) == 0:
                                continue
                            val = float(sum(nums) / len(nums))
                        elif self._is_number_like(v):
                            val = float(v)
                        else:
                            try:
                                import numpy as _np
                                import torch as _torch

                                if isinstance(v, _torch.Tensor):
                                    val = float(v.mean().item())
                                elif isinstance(v, _np.ndarray):
                                    val = float(v.mean())
                                else:
                                    val = float(v)
                            except Exception:
                                continue
                        results[k] = val
                    except Exception:
                        continue
        except Exception as e:
            self.logger.debug(f"_gather_metrics_results failed: {e}")
        return results

    def _is_number_like(self, x) -> bool:
        try:
            import numbers

            if isinstance(x, numbers.Number):
                return True
            return False
        except Exception:
            return False

    def _evaluation_epoch(self, epoch: int, part: str, dataloader) -> Dict[str, Any]:
        self._current_epoch = epoch
        self._batch_counter = 0
        self._update_pipeline_with_current_weights(epoch)

        result = super()._evaluation_epoch(epoch, part, dataloader)

        gathered = {}
        if not result or (isinstance(result, dict) and len(result) == 0):
            self.logger.debug(
                "super()._evaluation_epoch returned empty result — собираем метрики из self.metrics"
            )
            gathered = self._gather_metrics_results()
            loss_val = float(gathered.get("loss", float("inf")))
            result = dict(gathered)
            result.setdefault("loss", loss_val)
        else:
            try:
                gathered = self._gather_metrics_results()
                for k, v in gathered.items():
                    if k not in result:
                        result[k] = v
            except Exception:
                pass

        # Debug what we will log
        try:
            self.logger.debug(
                f"Available metrics in result (final): {list(result.keys()) if isinstance(result, dict) else 'non-dict'}"
            )
        except Exception:
            pass

        # Log and report everything numeric
        self.logger.info(f"МЕТРИКИ ЭПОХИ {epoch} ({part}):")
        if isinstance(result, dict):
            for metric_name, metric_value in result.items():
                if metric_name in ["epoch"]:
                    continue
                try:
                    val = float(metric_value)
                except Exception:
                    # skip non-numeric
                    continue
                self.logger.info(f"   {metric_name}: {val:.6f}")
                try:
                    self._report_scalar(
                        title=f"metrics/{part}",
                        series=metric_name,
                        value=val,
                        iteration=epoch,
                    )
                except Exception as e:
                    self.logger.debug(f"Failed to report scalar {metric_name}: {e}")

        # ensure loss & best logged
        current_loss = (
            float(result.get("loss", float("inf")))
            if isinstance(result, dict)
            else float("inf")
        )
        try:
            self._report_scalar(
                title="loss", series=part, value=current_loss, iteration=epoch
            )
            self._report_scalar(
                title="best",
                series="loss",
                value=float(self._best_loss),
                iteration=epoch,
            )
        except Exception:
            pass

        if current_loss < self._best_loss:
            self._best_loss = current_loss
            self._best_epoch = epoch
            self.logger.info(f"Новый лучший loss: {current_loss:.6f} (эпоха {epoch})")

        return result

    def _update_pipeline_with_current_weights(self, epoch: int):
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
            else:
                # SD 1.x
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
            self.pipe = self.pipe.to(self.device)
        except Exception as e:
            self.logger.error(f"Ошибка обновления пайплайна: {e}")

    def _save_epoch_checkpoint(self, epoch: int):
        try:
            checkpoint_path = self.save_dir / f"checkpoint-epoch{epoch}.pth"
            lora_path = self.save_dir / f"lora_weights_epoch{epoch}.pth"

            self.logger.info(f"СОХРАНЕНИЕ ЭПОХИ {epoch}")
            current_weights = self.model.get_state_dict()

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": current_weights,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": self._best_loss,
                    "best_epoch": self._best_epoch,
                },
                checkpoint_path,
            )
            torch.save(current_weights, lora_path)

            self._check_weight_changes(epoch, current_weights)
            self._show_saved_files(epoch)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения: {e}")

    def _check_weight_changes(
        self, epoch: int, current_weights: Dict[str, torch.Tensor]
    ):
        if self._last_saved_weights is None:
            self._last_saved_weights = current_weights
            self.logger.info("Первое сохранение весов")
            return

        total_diff = 0.0
        changed_params = 0

        for key in current_weights:
            if key in self._last_saved_weights:
                try:
                    diff = (
                        (current_weights[key] - self._last_saved_weights[key])
                        .abs()
                        .mean()
                        .item()
                    )
                    total_diff += diff
                    if diff > 1e-6:
                        changed_params += 1
                except Exception:
                    continue

        self.logger.info(
            f"Изменения весов: {changed_params}/{len(current_weights)} параметров"
        )
        self.logger.info(f"Общая разница: {total_diff:.8f}")

        if changed_params > 0:
            self.logger.info("Веса изменяются - обучение работает!")
        else:
            self.logger.error("ВЕСА НЕ ИЗМЕНЯЮТСЯ!")

        self._last_saved_weights = current_weights

    def _show_saved_files(self, epoch: int):
        files = list(self.save_dir.glob("*"))
        self.logger.info(f"Файлы в {self.save_dir}:")
        for f in sorted(files):
            try:
                size_mb = f.stat().st_size / 1024 / 1024
                self.logger.info(f"  - {f.name} ({size_mb:.1f} MB)")
            except Exception:
                self.logger.info(f"  - {f.name}")

    def _prepare_models_for_training(self):
        try:
            self.model.unet.train()
            self.model.text_encoder.train()
            if (
                hasattr(self.model, "text_encoder_2")
                and self.model.text_encoder_2 is not None
            ):
                self.model.text_encoder_2.train()
        except Exception as e:
            self.logger.debug(f"_prepare_models_for_training failed: {e}")

    def _prepare_models_for_inference(self):
        try:
            self.model.unet.eval()
            self.model.text_encoder.eval()
            if (
                hasattr(self.model, "text_encoder_2")
                and self.model.text_encoder_2 is not None
            ):
                self.model.text_encoder_2.eval()
        except Exception as e:
            self.logger.debug(f"_prepare_models_for_inference failed: {e}")

    def process_batch(
        self, batch: Dict[str, Any], train_metrics: MetricTracker
    ) -> Dict[str, Any]:
        batch["pixel_values"] = batch["pixel_values"].to(self.device)
        self.optimizer.zero_grad()
        self._prepare_models_for_training()

        # Forward pass
        model_output = self.model(
            pixel_values=batch["pixel_values"], prompt=batch["prompt"], do_cfg=False
        )
        batch.update(model_output)

        loss_dict = self.criterion(
            model_pred=batch["model_pred"], target=batch["target"]
        )
        loss = loss_dict["loss"]
        batch["loss"] = loss

        # Backward
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        train_metrics.update("loss", loss.item())
        return batch

    @torch.no_grad()
    def process_evaluation_batch(
        self,
        batch: Dict[str, Any],
        eval_metrics: MetricTracker,
        is_training_mode: bool = False,
    ) -> Dict[str, Any]:
        prompt = batch.get("prompt")
        batch_id = self._batch_counter
        self._batch_counter += 1

        if not hasattr(self, "pipe") or self.pipe is None:
            self._update_pipeline_with_current_weights(self._current_epoch)

        validation_args = getattr(self.config, "validation_args", {})
        num_images_per_prompt = getattr(validation_args, "num_images_per_prompt", 1)
        num_inference_steps = getattr(validation_args, "num_inference_steps", 50)
        guidance_scale = getattr(validation_args, "guidance_scale", 7.5)
        height = getattr(validation_args, "height", 256)
        width = getattr(validation_args, "width", 256)
        seed = getattr(validation_args, "seed", 42)

        epoch = self._current_epoch
        dynamic_seed = epoch * 1000 + batch_id * 100 + seed
        generated_images: List[Image.Image] = []
        try:
            generator = torch.Generator(device=self.device).manual_seed(
                int(dynamic_seed)
            )

            if isinstance(prompt, list):
                for p in prompt:
                    out = self.pipe(
                        p,
                        num_images_per_prompt=num_images_per_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        generator=generator,
                    )
                    generated_images.extend(getattr(out, "images", []))
            else:
                out = self.pipe(
                    prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                )
                generated_images = getattr(out, "images", [])

            self.logger.info(f"Сгенерировано {len(generated_images)} изображений")
            batch["generated"] = generated_images
            self._save_generated_images(generated_images, epoch, batch_id, prompt)
            if generated_images and self.writer is not None:
                for i, img in enumerate(generated_images[:2]):
                    try:
                        self._add_image_to_writer(
                            f"val_epoch_{epoch}/batch_{batch_id}_{i}",
                            img,
                            global_step=epoch,
                            writer_idx=batch_id,
                        )
                    except Exception as e:
                        self.logger.debug(
                            f"Failed to add generated image to writer: {e}"
                        )

        except Exception as e:
            self.logger.error(f"Ошибка генерации: {e}")
            batch["generated"] = []

        # prepare reference image from pixel_values for metrics
        if "pixel_values" in batch and len(batch["pixel_values"]) > 0:
            try:
                img_tensor = batch["pixel_values"][0].cpu()
                # normalize if in [-1,1]
                if img_tensor.min() < 0:
                    img_tensor = (img_tensor + 1.0) / 2.0
                # if single channel -> repeat to 3 channels
                if img_tensor.shape[0] == 1:
                    img_tensor = img_tensor.repeat(3, 1, 1)
                pil_image = T.ToPILImage()(img_tensor)
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                batch["image"] = pil_image
                batch["reference"] = pil_image
            except Exception as e:
                self.logger.error(f"Ошибка подготовки reference: {e}")
                batch["image"] = None
                batch["reference"] = None

        # compute metrics
        try:
            for metric in getattr(self, "metrics", []):
                metric_result = metric(**batch)
                # metric can return dict or scalar
                try:
                    if isinstance(metric_result, dict):
                        for k, v in metric_result.items():
                            # store into metric._data if exists (preserve your previous pattern)
                            if hasattr(metric, "_data"):
                                metric._data[k] = v
                    else:
                        if hasattr(metric, "_data"):
                            metric._data[
                                getattr(metric, "name", "metric")
                            ] = metric_result
                except Exception:
                    # be tolerant
                    continue
        except Exception as e:
            self.logger.error(f"Ошибка в метриках: {e}")
        return batch

    def _save_generated_images(
        self,
        images: List[Image.Image],
        epoch: int,
        batch_id: int,
        prompt: Optional[Union[str, List[str]]],
    ):
        try:
            images_dir = Path("outputs/generated_images") / f"epoch_{epoch}"
            images_dir.mkdir(parents=True, exist_ok=True)

            for i, img in enumerate(images):
                # build safe filename from prompt
                if isinstance(prompt, list) and len(prompt) > 0:
                    prompt_text = prompt[0]
                else:
                    prompt_text = prompt or "prompt"

                safe_prompt = (
                    "".join(
                        c
                        for c in str(prompt_text)[:40]
                        if c.isalnum() or c in (" ", "-", "_")
                    )
                    .rstrip()
                    .replace(" ", "_")
                )
                filename = f"epoch{epoch}_batch{batch_id}_{i}_{safe_prompt}.png"
                filepath = images_dir / filename

                try:
                    img.save(filepath)
                    self.logger.info(f"Сохранено: {filepath}")
                except Exception:
                    # try convert to RGB then save
                    try:
                        img.convert("RGB").save(filepath)
                        self.logger.info(f"Сохранено (конвертировано): {filepath}")
                    except Exception as e:
                        self.logger.warning(
                            f"Не удалось сохранить изображение локально: {e}"
                        )
                        continue

                # log first image to clearml
                if i == 0 and self._is_clearml:
                    try:
                        self._log_image_to_clearml(
                            img,
                            filepath,
                            epoch,
                            batch_id,
                            tag=f"generated/epoch_{epoch}/batch_{batch_id}",
                        )
                    except Exception as e:
                        self.logger.warning(f"Ошибка логирования в ClearML: {e}")

            # also save a short named copy in main outputs dir
            main_dir = Path("outputs/generated_images")
            main_dir.mkdir(parents=True, exist_ok=True)
            if images:
                try:
                    summary_name = (
                        f"custom_dreambooth_val_epoch{epoch}_batch{batch_id}.png"
                    )
                    images[0].save(main_dir / summary_name)
                except Exception:
                    try:
                        images[0].convert("RGB").save(main_dir / summary_name)
                    except Exception:
                        pass
        except Exception as e:
            self.logger.error(f"Ошибка сохранения изображений: {e}")

    def _log_image_to_clearml(
        self,
        img: Image.Image,
        filepath: Path,
        epoch: int,
        batch_id: int,
        tag: Optional[str] = None,
    ):
        try:
            if self._clearml_task is not None:
                try:
                    art_name = tag or f"generated_epoch_{epoch}_batch_{batch_id}"
                    self._clearml_task.upload_artifact(
                        name=art_name,
                        artifact_object=str(filepath),
                        metadata={"epoch": epoch, "batch_id": batch_id},
                    )
                    try:
                        self._clearml_task.get_logger().report_image(
                            title="Generated Images",
                            series=art_name,
                            local_path=str(filepath),
                            iteration=epoch,
                        )
                    except Exception:
                        try:
                            self._clearml_task.get_logger().report_media(
                                title="Generated Images",
                                series=art_name,
                                local_path=str(filepath),
                                iteration=epoch,
                            )
                        except Exception:
                            pass
                except Exception as e:
                    self.logger.warning(f"ClearML upload/report failed: {e}")

            if self.writer is not None and hasattr(self.writer, "add_image"):
                try:
                    img_tensor = T.ToTensor()(img)  # CHW float [0,1]
                    step = int(epoch) if isinstance(epoch, int) else 0
                    tag_name = tag or f"Generated/epoch_{epoch}/batch_{batch_id}"

                    writer_cls_name = getattr(
                        self.writer, "__class__", type(self.writer)
                    ).__name__
                    is_clearml_writer = "ClearML" in writer_cls_name or self._is_clearml

                    if is_clearml_writer:
                        try:
                            # most ClearML writer wrappers accept (tag, img_tensor, step)
                            self.writer.add_image(tag_name, img_tensor, step)
                        except TypeError:
                            # fallback: try without step (rare)
                            try:
                                self.writer.add_image(tag_name, img_tensor)
                            except Exception:
                                # final fallback: ignore
                                self.logger.debug(
                                    "ClearML writer add_image fallback failed"
                                )
                    else:
                        # tensorboard-like: pass dataformats
                        try:
                            self.writer.add_image(
                                tag_name, img_tensor, step, dataformats="CHW"
                            )
                        except TypeError:
                            # some custom writers may not accept dataformats
                            try:
                                self.writer.add_image(tag_name, img_tensor, step)
                            except Exception:
                                self.logger.debug("writer.add_image fallback failed")
                except Exception as e:
                    self.logger.debug(f"TensorBoard add_image failed: {e}")
        except Exception as e:
            self.logger.warning(
                f"Не удалось залогировать изображение в ClearML/TB: {e}"
            )

    def _add_image_to_writer(
        self,
        tag: str,
        img: Union[Image.Image, torch.Tensor],
        global_step: int = 0,
        writer_idx: int = 0,
    ):
        if self.writer is None or not hasattr(self.writer, "add_image"):
            return

        try:
            # if PIL -> convert
            if isinstance(img, Image.Image):
                img_tensor = T.ToTensor()(img)
            else:
                img_tensor = img

            writer_cls_name = getattr(
                self.writer, "__class__", type(self.writer)
            ).__name__
            is_clearml_writer = "ClearML" in writer_cls_name or self._is_clearml

            if is_clearml_writer:
                try:
                    self.writer.add_image(tag, img_tensor, int(global_step))
                except TypeError:
                    try:
                        self.writer.add_image(tag, img_tensor)
                    except Exception:
                        self.logger.debug(
                            "_add_image_to_writer: ClearML writer add_image fallback failed"
                        )
            else:
                try:
                    self.writer.add_image(
                        tag, img_tensor, int(global_step), dataformats="CHW"
                    )
                except TypeError:
                    try:
                        self.writer.add_image(tag, img_tensor, int(global_step))
                    except Exception:
                        self.logger.debug(
                            "_add_image_to_writer: writer.add_image fallback failed"
                        )
        except Exception as e:
            self.logger.debug(f"_add_image_to_writer failed: {e}")

    def _log_batch(
        self,
        batch: Dict[str, Any],
        output: Any,
        batch_idx: int,
        part: str = "train",
        epoch: int = 0,
    ):
        try:
            # generated
            if "generated" in batch and batch["generated"]:
                gen = batch["generated"][0]
                # if PIL image:
                if isinstance(gen, Image.Image):
                    tmp_path = Path("outputs/tmp")
                    tmp_path.mkdir(parents=True, exist_ok=True)
                    tmp_file = tmp_path / f"epoch{epoch}_batch{batch_idx}_gen.png"
                    try:
                        gen.save(tmp_file)
                    except Exception:
                        try:
                            gen.convert("RGB").save(tmp_file)
                        except Exception:
                            tmp_file = None

                    if tmp_file is not None:
                        if self._is_clearml:
                            try:
                                self._log_image_to_clearml(
                                    gen,
                                    tmp_file,
                                    epoch,
                                    batch_idx,
                                    tag=f"{part}/generated",
                                )
                            except Exception as e:
                                self.logger.debug(
                                    f"ClearML log failed for generated img: {e}"
                                )

                        # always try to add to writer
                        try:
                            self._add_image_to_writer(
                                f"{part}/epoch_{epoch}/generated",
                                gen,
                                global_step=epoch,
                                writer_idx=batch_idx,
                            )
                        except Exception:
                            pass
                else:
                    # gen could be tensor or numpy
                    try:
                        self._add_image_to_writer(
                            f"{part}/epoch_{epoch}/generated",
                            gen,
                            global_step=epoch,
                            writer_idx=batch_idx,
                        )
                    except Exception as e:
                        self.logger.debug(f"Failed to add generated as image: {e}")

            # original pixel_values (log every 10th)
            if "pixel_values" in batch and batch_idx % 10 == 0:
                try:
                    original_img = batch["pixel_values"][0]
                    if hasattr(original_img, "min") and original_img.min() < 0:
                        original_img = (original_img + 1.0) / 2.0
                    if original_img.shape[0] == 1:
                        original_img = original_img.repeat(3, 1, 1)

                    self._add_image_to_writer(
                        f"{part}/epoch_{epoch}/original",
                        original_img,
                        global_step=epoch,
                        writer_idx=batch_idx,
                    )
                except Exception as e:
                    # clear log but don't break training
                    self.logger.debug(f"Не удалось отправить оригинал в writer: {e}")

        except Exception as e:
            self.logger.error(f"Ошибка логирования: {e}")
