from pathlib import Path
import random
from src.datasets.base_dataset import BaseDataset
from src.datasets.data_utils import IMG_EXTENTIONS
import logging

logger = logging.getLogger(__name__)


class CustomDreamBoothDataset(BaseDataset):
    def __init__(self, **kwargs):
        self.placeholder_token = kwargs.get("placeholder_token")
        self.class_name = kwargs.get("class_name")
        self.data_path = Path(kwargs.get("data_path"))
        self.mode = kwargs.get("mode", "train")

        train_ratio = kwargs.get("train_ratio")
        val_ratio = kwargs.get("val_ratio")
        random_seed = kwargs.get("random_seed")
        captions_file = kwargs.get("captions_file")

        all_index = self._build_index(captions_file)
        index = self._split_data(all_index, train_ratio, val_ratio, random_seed)

        limit = kwargs.get("limit")
        shuffle_index = kwargs.get("shuffle_index", False)
        instance_transforms = kwargs.get("instance_transforms")

        super().__init__(
            index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def _build_index(self, captions_file):
        """Строим индекс из captions файла или из папки с изображениями"""

        index = []

        if captions_file and Path(captions_file).exists():
            logger.info(f"Загружаем captions из: {captions_file}")
            with open(captions_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if "|" in line:
                        filename, prompt = line.split("|", 1)
                        file_path = self.data_path / filename.strip()
                        if file_path.exists():
                            index.append(
                                {
                                    "path": str(file_path),
                                    "label": prompt.strip(),
                                }
                            )
                        else:
                            logger.info(f"Файл не найден: {file_path}")
        else:
            logger.info("Используем все изображения из папки")
            for file_path in self.data_path.iterdir():
                if file_path.suffix.lower() in IMG_EXTENTIONS:
                    index.append(
                        {
                            "path": str(file_path),
                            "label": f"a photo of a {self.placeholder_token} {self.class_name}",
                        }
                    )
        return index

    def _split_data(self, all_index, train_ratio, val_ratio, random_seed):
        random.seed(random_seed)
        random.shuffle(all_index)

        train_size = int(len(all_index) * train_ratio)

        if self.mode == "train":
            index = all_index[:train_size]
            train_files = {item["path"] for item in index}
            logger.info(f"TRAIN files: {len(train_files)}")
        else:
            index = all_index[train_size:]
            val_files = {item["path"] for item in index}
            logger.info(f"VAL files: {len(val_files)}")

            if hasattr(self, "train_files"):
                intersection = train_files & val_files
                if intersection:
                    logger.info(f"ОПАСНО! Пересечение train/val: {intersection}")
                else:
                    logger.info("Пересечения нет!")
        return index
