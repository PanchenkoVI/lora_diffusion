import logging
import random
from typing import List
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.
    """

    def __init__(
        self, index, limit=None, shuffle_index=False, instance_transforms=None
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset.
            instance_transforms (Callable | None): transforms that should be
                applied on the image. This is a Compose object, not a dict.
        """
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: List[dict] = index
        self.instance_transforms = instance_transforms

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it.
        """
        data_dict = self._index[ind]
        data_path = data_dict["path"]
        data_object = self.load_object(data_path)

        instance_data = {
            "pixel_values": data_object,
            "prompt": data_dict["label"],
        }

        instance_data = self.preprocess_data(instance_data)
        return instance_data

    def __len__(self):
        return len(self._index)

    def load_object(self, path):
        return Image.open(path).convert("RGB")

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.
        """
        if self.instance_transforms is not None:
            instance_data["pixel_values"] = self.instance_transforms(
                instance_data["pixel_values"]
            )
        return instance_data

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
