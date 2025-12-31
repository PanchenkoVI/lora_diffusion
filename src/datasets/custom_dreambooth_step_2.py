from pathlib import Path
import random
from src.datasets.base_dataset import BaseDataset
from src.datasets.data_utils import IMG_EXTENTIONS
import logging

logger = logging.getLogger(__name__)


class CustomDreamBoothDataset2(BaseDataset):
    def __init__(self, **kwargs):
        self.placeholder_token = kwargs.get("placeholder_token")
        self.class_name = kwargs.get("class_name")
        self.data_path = Path(kwargs.get("data_path"))
        self.mode = kwargs.get("mode", "train")

        self.regularization_ratio = kwargs.get("regularization_ratio", 0.0)
        self.regular_prompts = kwargs.get("regular_prompts", [])

        train_ratio = kwargs.get("train_ratio")
        val_ratio = kwargs.get("val_ratio")
        random_seed = kwargs.get("random_seed")
        captions_file = kwargs.get("captions_file")

        all_index = self._build_index(captions_file)

        if self.mode == "train" and self.regularization_ratio > 0:
            all_index = self._add_regularization_prompts(all_index)

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

    def _add_regularization_prompts(self, index):
        """Добавляет регуляризационные промпты (без placeholder token)"""

        if not self.regular_prompts:
            self.regular_prompts = [
                # Основные описания
                "a fluffy toy with blue eyes",
                "a hairy creature toy",
                "blue-eyed plush toy",
                "furry monster toy",
                "fluffy stuffed animal",
                "plush creature with fangs",
                # Эмоции
                "a smiling hairy toy",
                "toy with surprised expression",
                "happy monster plush",
                "fluffy toy with frightened look",
                "blue-eyed creature with wide eyes",
                # Позы
                "toy lying on surface",
                "plush animal from above",
                "side view of fluffy toy",
                "toy looking to the side",
                "small stuffed animal centered",
                "toy on its side",
                # Окружение
                "fluffy toy on furniture",
                "plush creature on table",
                "stuffed animal on bed",
                "toy on dresser",
                "blue-eyed creature on blanket",
                "toy on windowsill",
                # Детали
                "toy with messy hair",
                "fluffy creature with tousled fur",
                "plush with wavy hair",
                "hairy toy with visible eyes",
                "smiling creature with blue eyes",
                "toy with shaggy fur",
                # Стили
                "closeup of fluffy toy",
                "photo of plush animal",
                "image of stuffed creature",
                "portrait of hairy toy",
                "shot of blue-eyed monster",
                "picture of fuzzy toy",
            ]

        regular_count = int(len(index) * self.regularization_ratio)
        if regular_count == 0:
            return index

        logger.info(f"Добавляем {regular_count} регуляризационных промптов для йети")

        regular_items = []
        for i in range(regular_count):
            original_item = random.choice(index)
            regular_prompt = random.choice(self.regular_prompts)

            regular_items.append(
                {"path": original_item["path"], "label": regular_prompt}
            )

        result_index = index + regular_items
        random.shuffle(result_index)

        original_with_token = sum(1 for item in index if "sksToy" in item["label"])
        regular_without_token = sum(
            1 for item in regular_items if "sksToy" not in item["label"]
        )

        logger.info(f"Итоговый датасет: {len(result_index)} элементов")
        logger.info(f"   - С токеном: {original_with_token}")
        logger.info(f"   - Без токена: {regular_without_token}")
        logger.info(
            f"   - Соотношение: {regular_without_token/len(result_index)*100:.1f}% регуляризации"
        )

        return result_index

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
