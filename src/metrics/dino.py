import torch
from src.metrics.base_metric import BaseMetric
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


class DinoMetric(BaseMetric):
    def __init__(self, model_name, to_norm, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.to_norm = to_norm
        try:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        except TypeError:
            self.model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    256,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        self.reset()

    def reset(self):
        self._scores = []

    def compute(self):
        if not self._scores:
            return {"dino_score": 0.0}

        avg_score = sum(self._scores) / len(self._scores)
        return {"dino_score": avg_score}

    def to_cpu(self):
        self.model = self.model.to("cpu")

    def to_cuda(self):
        self.model = self.model.to(self.device)

    def get_features(self, images):
        preprecessed = [self.preprocess(img) for img in images]
        preprecessed = torch.stack(preprecessed).to(self.device)
        features = self.model(preprecessed)
        if self.to_norm:
            features = features / features.clone().norm(dim=-1, keepdim=True)
        return features

    def __call__(self, **batch):
        if "concept" in batch:
            concept = batch["concept"]
        elif "image" in batch:
            concept = batch["image"]
        else:
            logger.warning("DINO - No concept or image found in batch")
            return {"dino_score": 0.0}

        generated = batch.get("generated", [])

        if not isinstance(concept, list):
            concept = [concept]

        if len(concept) == 0:
            logger.warning("DINO - Concept list is empty")
            return {"dino_score": 0.0}

        if not isinstance(generated, list) or len(generated) == 0:
            logger.warning("DINO - Generated list is empty or invalid")
            return {"dino_score": 0.0}

        try:
            gen_features = self.get_features(generated)
            concept_features = self.get_features(concept)

            similarity_matrix = gen_features @ concept_features.T
            score = similarity_matrix.mean().item()

            logger.info(
                f"DINO - Similarity matrix shape: {similarity_matrix.shape}, mean score: {score}"
            )

            self._scores.append(score)

            result = {"dino_score": score}
            return result

        except Exception as e:
            logger.error(f"DINO processing error: {e}")
            import traceback

            traceback.print_exc()
            return {"dino_score": 0.0}
