import numpy as np
from src.metrics.base_metric import BaseMetric
from src.metrics.aligner import Aligner
from src.utils.model_utils import cos_sim


class IDSimBest(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.aligner = Aligner()

    def to_cpu(self):
        pass

    def to_cuda(self):
        pass

    def __call__(self, **batch):
        batch_bboxes, batch_embeds = self.aligner(batch["generated"])

        result = 0
        for face_bboxes, facce_embeds in zip(batch_bboxes, batch_embeds):
            if facce_embeds is None:
                continue
            score = self.choose_face(facce_embeds, face_bboxes, batch["id"])
            result += score
        result = result / len(batch_embeds)
        return {"id_sim": result}

    def choose_face(self, embeds, bboxes, person_id):
        best_score = -np.inf
        for embed in embeds:
            best_score = max(cos_sim(embed, self.id_embeds[person_id]), best_score)
        return best_score


class IDSimMax(IDSimBest):
    def choose_face(self, embeds, bboxes, person_id):
        best_score = -np.inf
        pairs = list(zip(embeds, bboxes))
        pairs = sorted(pairs, key=lambda x: -(x[1][3] - x[1][1]) * (x[1][2] - x[1][0]))
        best_embed = pairs[0][0]
        best_score = cos_sim(best_embed, self.id_embeds[person_id])
        return best_score
