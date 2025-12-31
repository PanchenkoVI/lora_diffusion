import logging
from typing import Optional

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

logger = logging.getLogger(__name__)


class DiffusionLora(nn.Module):
    """
    LoRA wrapper for diffusion training (focused on SD 1.5).
    Key points:
      - placeholder token becomes a registered nn.Parameter named `placeholder_embedding`
      - LoRA applied via get_peft_model(...) and only LoRA params + placeholder_embedding are trainable
      - forward replaces placeholder token positions in text encoder last_hidden_state
    """

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

        if not hasattr(self, "device"):
            raise ValueError("device must be provided in config")
        if not hasattr(self, "pretrained_model_name_or_path"):
            raise ValueError("pretrained_model_name_or_path must be provided in config")

        self.lora_modules = getattr(
            self,
            "lora_modules",
            ["to_k", "to_v", "to_q", "to_out.0", "add_k_proj", "add_v_proj"],
        )
        self.init_lora_weights = getattr(self, "init_lora_weights", True)
        self.rank = getattr(self, "rank", 4)
        self.lora_alpha = getattr(self, "lora_alpha", 4)
        self.lora_dropout = getattr(self, "lora_dropout", 0.0)
        self.use_unet_lora = getattr(self, "use_unet_lora", True)
        self.use_text_encoder_lora = getattr(self, "use_text_encoder_lora", True)

        wdt = getattr(self, "weight_dtype", None)
        if wdt == "bf16":
            self.weight_dtype = torch.bfloat16
        elif wdt == "fp16":
            self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32

        self.placeholder_token = getattr(self, "placeholder_token", "<placeholder>")
        self.initializer_token = getattr(self, "initializer_token", None)
        self.is_sdxl = "xl" in str(self.pretrained_model_name_or_path).lower()
        self.target_size = getattr(self, "target_size", 1024)

        self.placeholder_embedding: Optional[nn.Parameter] = None

        logger.info(
            f"Initializing DiffusionLora: model={self.pretrained_model_name_or_path}, device={self.device}, dtype={self.weight_dtype}, is_sdxl={self.is_sdxl}"
        )

        self._load_components()
        self._ensure_placeholder_token_single()
        self._add_placeholder_embedding()

    def _load_components(self):
        try:
            if self.is_sdxl:
                logger.info("Loading SDXL components (best-effort)...")
                self.tokenizer = CLIPTokenizer.from_pretrained(
                    self.pretrained_model_name_or_path, subfolder="tokenizer"
                )
                self.text_encoder = CLIPTextModel.from_pretrained(
                    self.pretrained_model_name_or_path,
                    subfolder="text_encoder",
                    torch_dtype=self.weight_dtype,
                ).to(self.device)
                self.tokenizer_2 = None
                self.text_encoder_2 = None
            else:
                logger.info("Loading SD 1.5 components...")
                self.tokenizer = CLIPTokenizer.from_pretrained(
                    self.pretrained_model_name_or_path, subfolder="tokenizer"
                )
                self.text_encoder = CLIPTextModel.from_pretrained(
                    self.pretrained_model_name_or_path,
                    subfolder="text_encoder",
                    torch_dtype=self.weight_dtype,
                ).to(self.device)
                self.tokenizer_2 = None
                self.text_encoder_2 = None

            logger.info("Loading VAE...")
            self.vae = AutoencoderKL.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="vae",
                torch_dtype=self.weight_dtype,
            ).to(self.device)

            logger.info("Loading UNet...")
            self.unet = UNet2DConditionModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="unet",
                torch_dtype=self.weight_dtype,
            ).to(self.device)

            logger.info("Loading scheduler...")
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="scheduler"
            )

            logger.info("Components loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading components: {e}")
            raise

    def _ensure_placeholder_token_single(self):
        """
        Ensure tokenizer maps placeholder_token to a single new token id.
        If tokenizer.encode(placeholder_token) returns >1 ids, try wrapping with <> and add token again.
        This makes it much more likely that prompt like 'sksToy' is treated as a single token.
        """
        tok = self.placeholder_token
        try:
            added = self.tokenizer.add_tokens(tok)
            ids = self.tokenizer.encode(tok, add_special_tokens=False)
            if len(ids) == 1:
                pid = ids[0]
                logger.info(
                    f"Placeholder token resolved as '{tok}', id={pid} (added={added})"
                )
                self.placeholder_token = tok
                return
            wrapped = f"<{tok.strip('<>')}>"
            added2 = self.tokenizer.add_tokens(wrapped)
            ids2 = self.tokenizer.encode(wrapped, add_special_tokens=False)
            if len(ids2) == 1:
                self.placeholder_token = wrapped
                logger.info(
                    f"Placeholder token adjusted to wrapped form '{wrapped}', id={ids2[0]} (added={added2})"
                )
                return
            if len(ids) > 0:
                logger.warning(
                    f"Placeholder token '{tok}' tokenized into {len(ids)} ids, using first id (may be subword). Consider using a token like <{tok}>."
                )
                self.placeholder_token = tok
                return
            logger.warning(
                "Placeholder token could not be forced to a single token. Will proceed but mapping may be multi-token."
            )
        except Exception as e:
            logger.warning(
                f"Could not ensure placeholder token single-token property: {e}"
            )

    def _add_placeholder_embedding(self):
        """
        Create a trainable placeholder_embedding (nn.Parameter) and register it.
        Also copy its current value into text_encoder embedding matrix so initial forwards match.
        """
        pid = self.tokenizer.convert_tokens_to_ids(self.placeholder_token)
        if pid is None:
            ids = self.tokenizer.encode(
                self.placeholder_token, add_special_tokens=False
            )
            pid = ids[0] if ids else None

        if pid is None:
            raise RuntimeError("Could not determine placeholder token id.")
        try:
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        except Exception:
            # ignore if not supported
            pass

        init_vec = None
        if self.initializer_token:
            try:
                init_ids = self.tokenizer.encode(
                    self.initializer_token, add_special_tokens=False
                )
                if init_ids:
                    with torch.no_grad():
                        init_vec = (
                            self.text_encoder.get_input_embeddings()
                            .weight[init_ids[0]]
                            .detach()
                            .cpu()
                            .clone()
                        )
            except Exception:
                init_vec = None

        if init_vec is None:
            emb_dim = self.text_encoder.get_input_embeddings().weight.shape[1]
            init_vec = torch.randn(emb_dim, dtype=self.weight_dtype) * 1e-3

        # register parameter
        self.placeholder_embedding = nn.Parameter(init_vec.to(self.device))
        self.register_parameter("placeholder_embedding", self.placeholder_embedding)

        # write into text encoder embedding matrix for initial consistency
        with torch.no_grad():
            emb = self.text_encoder.get_input_embeddings().weight
            try:
                emb.data[pid] = self.placeholder_embedding.data.to(emb.device)
            except Exception:
                # if direct assignment fails, ignore (we replace hidden states at forward anyway)
                logger.debug(
                    "Could not write placeholder embedding into text_encoder embedding matrix; forward will replace hidden states directly."
                )

        logger.info(f"Placeholder embedding created and registered (token id = {pid}).")

    def _create_lora_config(self, r=None, alpha=None, modules=None):
        r = self.rank if r is None else r
        alpha = self.lora_alpha if alpha is None else alpha
        modules = self.lora_modules if modules is None else modules
        return LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=modules,
            lora_dropout=self.lora_dropout,
            init_lora_weights=self.init_lora_weights,
        )

    def _create_unet_lora_config(self):
        return self._create_lora_config()

    def _create_text_encoder_lora_config(self):
        text_modules = getattr(self, "text_encoder_lora_modules", ["q_proj", "v_proj"])
        text_rank = getattr(self, "text_encoder_rank", 2)
        text_alpha = getattr(self, "text_encoder_alpha", 2)
        return LoraConfig(
            r=text_rank,
            lora_alpha=text_alpha,
            target_modules=text_modules,
            lora_dropout=self.lora_dropout,
            init_lora_weights=self.init_lora_weights,
        )

    def _enable_peft_trainable(self, module):
        """
        After get_peft_model set all params requires_grad=False then enable those containing 'lora' in name.
        """
        for n, p in module.named_parameters():
            p.requires_grad = False
        enabled = 0
        for n, p in module.named_parameters():
            if "lora" in n.lower():
                p.requires_grad = True
                enabled += 1
        logger.info(
            f"Enabled {enabled} LoRA parameters in module {type(module).__name__}"
        )
        return enabled

    def prepare_for_training(self):
        logger.info("=== prepare_for_training ===")
        try:
            self.vae.requires_grad_(False)
            logger.info("VAE frozen.")
        except Exception:
            logger.warning("Could not freeze VAE (ignored).")

        # apply LoRA to UNet
        if self.use_unet_lora:
            logger.info("Applying LoRA to UNet...")
            cfg = self._create_unet_lora_config()
            self.unet = get_peft_model(self.unet, cfg)
            self.unet.to(self.device)
            try:
                self._enable_peft_trainable(self.unet)
            except Exception as e:
                logger.warning(f"Could not explicitly enable LoRA params for UNet: {e}")

        # apply LoRA to text encoder
        if self.use_text_encoder_lora:
            logger.info("Applying LoRA to Text Encoder...")
            cfg = self._create_text_encoder_lora_config()
            self.text_encoder = get_peft_model(self.text_encoder, cfg)
            self.text_encoder.to(self.device)
            try:
                self._enable_peft_trainable(self.text_encoder)
            except Exception as e:
                logger.warning(
                    f"Could not explicitly enable LoRA params for Text Encoder: {e}"
                )

        # ensure placeholder embedding is present and trainable
        if getattr(self, "placeholder_embedding", None) is None:
            logger.info("Placeholder embedding missing — creating fallback.")
            emb_dim = self.text_encoder.get_input_embeddings().weight.shape[1]
            self.placeholder_embedding = nn.Parameter(
                torch.randn(emb_dim, device=self.device) * 1e-3
            )
            self.register_parameter("placeholder_embedding", self.placeholder_embedding)
        self.placeholder_embedding.requires_grad = True

        # final diagnostics
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total trainable params after prepare_for_training: {total:,}")
        if total == 0:
            logger.error("No trainable params found! Check PEFT and placeholder setup.")
            self._debug_trainable()

    def _debug_trainable(self):
        logger.info("=== trainable parameters list ===")
        for n, p in self.named_parameters():
            if p.requires_grad:
                logger.info(f"  TRAINABLE: {n} {tuple(p.shape)}")

    def get_trainable_params(self, config):
        logger.info("=== get_trainable_params ===")
        unet_params = []
        text_params = []
        embed_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "unet" in name:
                unet_params.append(p)
            elif "text_encoder" in name:
                if "embedding" in name or name == "placeholder_embedding":
                    embed_params.append(p)
                else:
                    text_params.append(p)
            elif name == "placeholder_embedding":
                embed_params.append(p)
            else:
                # fallback to unet group
                unet_params.append(p)

        groups = []
        if unet_params:
            groups.append(
                {
                    "params": unet_params,
                    "lr": getattr(config, "lr_for_unet", 1e-4),
                    "name": "unet_lora",
                }
            )
            logger.info(f"UNet LoRA params: {sum(x.numel() for x in unet_params):,}")
        if text_params:
            groups.append(
                {
                    "params": text_params,
                    "lr": getattr(config, "lr_for_text_encoder", 1e-5),
                    "name": "text_encoder_lora",
                }
            )
            logger.info(
                f"TextEncoder LoRA params: {sum(x.numel() for x in text_params):,}"
            )
        if embed_params:
            groups.append(
                {
                    "params": embed_params,
                    "lr": getattr(config, "lr_for_text_encoder", 1e-5),
                    "name": "embeddings",
                }
            )
            logger.info(f"Embeddings params: {sum(x.numel() for x in embed_params):,}")

        total = sum(sum(p.numel() for p in g["params"]) for g in groups)
        logger.info(f"Total params for optimizer: {total:,}")
        return groups

    def get_state_dict(self):
        logger.info("=== get_state_dict ===")
        sd = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                sd[name] = p.detach().cpu().clone()
                if len(sd) <= 3:
                    logger.info(f"  saving param: {name} {tuple(p.shape)}")
        try:
            if hasattr(self.unet, "peft_config"):
                u_sd = get_peft_model_state_dict(self.unet)
                for k, v in u_sd.items():
                    sd[f"unet_lora.{k}"] = v.detach().cpu().clone()
                logger.info(f"Saved UNet PEFT state: {len(u_sd)} tensors")
        except Exception as e:
            logger.warning(f"Could not save UNet PEFT state: {e}")

        try:
            if hasattr(self.text_encoder, "peft_config"):
                t_sd = get_peft_model_state_dict(self.text_encoder)
                for k, v in t_sd.items():
                    sd[f"text_encoder_lora.{k}"] = v.detach().cpu().clone()
                logger.info(f"Saved TextEncoder PEFT state: {len(t_sd)} tensors")
        except Exception as e:
            logger.warning(f"Could not save TextEncoder PEFT state: {e}")

        logger.info(f"State dict prepared: {len(sd)} tensors")
        return sd

    def load_state_dict_(self, state_dict):
        logger.info("=== load_state_dict_ ===")

        for name, p in self.named_parameters():
            if name in state_dict and p.requires_grad:
                try:
                    p.data.copy_(state_dict[name].to(p.device))
                    logger.info(f"Loaded param {name}")
                except Exception as e:
                    logger.warning(f"Failed to load param {name}: {e}")
        if (
            "placeholder_embedding" in state_dict
            and getattr(self, "placeholder_embedding", None) is not None
        ):
            try:
                self.placeholder_embedding.data.copy_(
                    state_dict["placeholder_embedding"].to(
                        self.placeholder_embedding.device
                    )
                )
                logger.info("Loaded placeholder_embedding.")
            except Exception as e:
                logger.warning(f"Could not load placeholder_embedding: {e}")
        try:
            u = {
                k[len("unet_lora.") :]: v
                for k, v in state_dict.items()
                if k.startswith("unet_lora.")
            }
            if u and hasattr(self.unet, "peft_config"):
                set_peft_model_state_dict(self.unet, u)
                logger.info(f"Loaded UNet PEFT state ({len(u)} tensors).")
        except Exception as e:
            logger.warning(f"Could not set UNet PEFT state: {e}")

        try:
            t = {
                k[len("text_encoder_lora.") :]: v
                for k, v in state_dict.items()
                if k.startswith("text_encoder_lora.")
            }
            if t and hasattr(self.text_encoder, "peft_config"):
                set_peft_model_state_dict(self.text_encoder, t)
                logger.info(f"Loaded TextEncoder PEFT state ({len(t)} tensors).")
        except Exception as e:
            logger.warning(f"Could not set TextEncoder PEFT state: {e}")

    def forward(
        self, pixel_values: torch.Tensor, prompt, do_cfg: bool = False, *args, **kwargs
    ):
        """
        Forward pass for training:
         - encode images into latents (VAE frozen -> no_grad)
         - get text embeddings (text_encoder) and substitute placeholder positions
         - pass to unet (peft-wrapped)
        """

        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        batch_size = pixel_values.shape[0]

        # VAE encode in no_grad (vae frozen)
        with torch.no_grad():
            latents = self.vae.encode(
                pixel_values.to(self.weight_dtype)
            ).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        if self.is_sdxl:
            raise NotImplementedError(
                "SDXL forward not implemented in this SD1.5-focused class."
            )
        else:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            input_ids = text_inputs.input_ids  # (B, L)

            outputs = self.text_encoder(input_ids)
            if isinstance(outputs, tuple):
                text_embeddings = outputs[0]
            else:
                text_embeddings = (
                    outputs.last_hidden_state
                    if hasattr(outputs, "last_hidden_state")
                    else outputs[0]
                )

            pid = self.tokenizer.convert_tokens_to_ids(self.placeholder_token)
            if pid is None:
                ids = self.tokenizer.encode(
                    self.placeholder_token, add_special_tokens=False
                )
                pid = ids[0] if ids else None

            if pid is not None:
                matches = input_ids == pid
                if matches.any():
                    idxs = torch.nonzero(matches, as_tuple=False)
                    for b_idx, pos_idx in idxs:
                        text_embeddings[b_idx, pos_idx, :] = self.placeholder_embedding
                else:
                    pass
            else:
                logger.warning(
                    "Placeholder token id couldn't be resolved in forward; no replacement performed."
                )

            unet_out = self.unet(
                noisy_latents, timesteps, encoder_hidden_states=text_embeddings
            )
            model_pred = unet_out.sample if hasattr(unet_out, "sample") else unet_out

        pred_type = getattr(self.noise_scheduler.config, "prediction_type", "epsilon")
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {pred_type}")

        return {"model_pred": model_pred, "target": target}

    def debug_trainable_parameters(self):
        logger.info("=== DEBUG TRAINABLE PARAMETERS ===")
        total = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                total += p.numel()
                logger.info(f"  {name}: {tuple(p.shape)} ({p.numel():,})")
        logger.info(f"TOTAL TRAINABLE: {total:,}")
        if total == 0:
            logger.error("NO TRAINABLE PARAMETERS!")
