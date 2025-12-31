import torch
from src.metrics.base_metric import BaseMetric
import clip


class TextSimMetric(BaseMetric):
    def __init__(self, model_name, device, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """

        super().__init__(*args, **kwargs)
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        self.reset()

    def reset(self):
        self._scores = []

    def compute(self):
        if not self._scores:
            return {"text_sim": 0.0}

        avg_score = sum(self._scores) / len(self._scores)
        return {"text_sim": avg_score}

    def to_cpu(self):
        self.model = self.model.to("cpu")

    def to_cuda(self):
        self.model = self.model.to(self.device)

    def __call__(self, **batch):
        prompt = batch["prompt"]
        tokenized_prompt = clip.tokenize([prompt]).to(self.device)
        generated = batch["generated"]
        # assert type(generated) is list, type(generated)
        # assert type(prompt) is str, type(prompt)
        assert isinstance(generated, str) is str, isinstance(generated, str)
        assert isinstance(prompt, str) is str, isinstance(prompt, str)

        if len(generated) == 0:
            return {"text_sim": 0.0}

        preprecessed = [self.preprocess(img) for img in generated]
        images = torch.stack(preprecessed).to(self.device)
        _, logits_per_text = self.model(images, tokenized_prompt)
        score = logits_per_text.mean().item()

        self._scores.append(score)

        result = {"text_sim": score}
        return result
