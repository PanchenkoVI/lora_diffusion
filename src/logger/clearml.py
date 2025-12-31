from clearml import Task
import numpy as np
import pandas as pd


class ClearMLWriter:
    def __init__(self, logger, project_config, project_name, run_name=None, **kwargs):
        try:
            self.task = Task.init(
                project_name=project_name, task_name=run_name, reuse_last_task_id=False
            )
            self.task.connect(project_config)
            self.logger = self.task.get_logger()
            logger.info(f"ClearML initialized: {project_name}/{run_name}")

        except Exception as e:
            logger.warning(f"ClearML initialization failed: {e}")
            logger.warning("For use ClearML install it via: pip install clearml")
            self.logger = None

        self.step = 0
        self.mode = ""

    def set_step(self, step, mode="train"):
        self.mode = mode
        self.step = step

    def add_scalar(self, scalar_name, scalar):
        if self.logger:
            self.logger.report_scalar(
                title=scalar_name, series=scalar_name, value=scalar, iteration=self.step
            )

    def add_scalars(self, scalars):
        if self.logger:
            for name, value in scalars.items():
                self.logger.report_scalar(
                    title=name, series=name, value=value, iteration=self.step
                )

    def add_image(self, image_name, image):
        if self.logger:
            if hasattr(image, "detach"):
                image = image.detach().cpu().numpy()
                if len(image.shape) == 3 and image.shape[0] == 3:  # CHW -> HWC
                    image = image.transpose(1, 2, 0)
                # Normalize to 0-255 if needed
                if image.dtype != np.uint8:
                    image = (image - image.min()) / (image.max() - image.min()) * 255
                    image = image.astype(np.uint8)

            self.logger.report_image(
                title=image_name, series=image_name, image=image, iteration=self.step
            )

    def add_images(self, images_name, images):
        if self.logger:
            for i, image in enumerate(images):
                if hasattr(image, "detach"):
                    image = image.detach().cpu().numpy()
                    if len(image.shape) == 3 and image.shape[0] == 3:  # CHW -> HWC
                        image = image.transpose(1, 2, 0)
                    if image.dtype != np.uint8:
                        image = (
                            (image - image.min()) / (image.max() - image.min()) * 255
                        )
                        image = image.astype(np.uint8)

                self.logger.report_image(
                    title=images_name,
                    series=f"{images_name}_{i}",
                    image=image,
                    iteration=self.step,
                )

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        if self.logger:
            values_for_hist = values_for_hist.detach().cpu().numpy()
            self.logger.report_histogram(
                title=hist_name,
                series=hist_name,
                values=values_for_hist,
                iteration=self.step,
                xaxis="Bins",
                yaxis="Count",
            )

    def add_text(self, text_name, text):
        if self.logger:
            self.logger.report_text(text, iteration=self.step)

    def add_audio(self, audio_name, audio, sample_rate=None):
        pass

    def add_table(self, table_name, table: pd.DataFrame):
        if self.logger:
            self.task.upload_artifact(table_name, table)
