import os
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import logging
import warnings
from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed, setup_saving_and_logging_inference

warnings.filterwarnings("ignore")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


@hydra.main(
    version_base=None, config_path="src/configs", config_name="persongen_inference_lora"
)
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """

    set_random_seed(config.inferencer.seed)

    project_config = OmegaConf.to_container(config)

    logger = setup_saving_and_logging_inference(config)
    writer = instantiate(config.writer, logger, project_config)

    device = torch.device(config.inferencer.device)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device, logger)

    # build model architecture, then print to console
    model = instantiate(config.model, device=device)
    model.prepare_for_training()

    metrics = []
    for metric_name in config.inference_metrics:
        metric_config = config.metrics[metric_name]
        metrics.append(instantiate(metric_config, name=metric_name, device=device))

    pipeline = instantiate(
        config.pipeline,
        model=model,
        device=None,
        device_map="auto",
        torch_dtype=torch.float32,
    )

    inferencer = instantiate(
        config.inferencer,
        model=model,
        pipe=pipeline,
        metrics=metrics,
        global_config=config,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        _recursive_=False,
    )

    inferencer.inference()


if __name__ == "__main__":
    main()
    logger.info("<============| Done |============>\n")
