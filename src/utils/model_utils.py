import torch
import numpy as np
from transformers import PretrainedConfig


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def cos_sim(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    return arr1 @ arr2 / np.linalg.norm(arr1) / np.linalg.norm(arr2)


class BaseTimer:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.start = torch.mps.Event(enable_timing=True)
            self.end = torch.mps.Event(enable_timing=True)
        elif torch.cuda.is_available():
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)

        self.start.record()

    def stop(self):
        self.end.record()

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        return self.start.elapsed_time(self.end) / 1000
