import torch
from diffusers import StableDiffusionXLPipeline
import logging

logger = logging.getLogger(__name__)


class SDXLPipeline:
    def __init__(self, model, device, pipe):
        self.model = model
        self.device = device
        self.pipe = pipe

    @classmethod
    def from_pretrained(
        cls,
        model,
        pretrained_model_name_or_path,
        torch_dtype=torch.float32,
        use_safetensors=True,
        **kwargs,
    ):
        """
        Создает pipeline из предобученной модели
        """

        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        logger.info(
            f"logger.infoCreating SDXL pipeline from: {pretrained_model_name_or_path}"
        )

        pipe = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path,
            dtype=dtype,
            use_safetensors=use_safetensors,
        )

        if hasattr(model, "vae"):
            pipe.vae = model.vae
        if hasattr(model, "text_encoder"):
            pipe.text_encoder = model.text_encoder
        if hasattr(model, "text_encoder_2"):
            pipe.text_encoder_2 = model.text_encoder_2
        if hasattr(model, "unet"):
            pipe.unet = model.unet
        if hasattr(model, "tokenizer"):
            pipe.tokenizer = model.tokenizer
        if hasattr(model, "tokenizer_2"):
            pipe.tokenizer_2 = model.tokenizer_2

        device = model.device
        pipe = pipe.to(device)

        logger.info("SDXL pipeline created successfully")

        return cls(model=model, device=device, pipe=pipe)

    def generate(
        self,
        prompt,
        negative_prompt,
        num_inference_steps,
        guidance_scale,
        height,
        width,
        num_images_per_prompt,
        **kwargs,
    ):
        logger.info(f"Generating {num_images_per_prompt} images")

        with torch.no_grad():
            try:
                images = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    num_images_per_prompt=num_images_per_prompt,
                ).images
                logger.info(f"Generated {len(images)} images")
                return images
            except Exception as e:
                logger.info(f"Generation error: {e}")
                return []
