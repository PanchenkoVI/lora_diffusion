from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate import collate_fn, collate_fn_val
from src.utils.init_utils import set_worker_seed

IMG_EXTENTIONS = set([".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"])


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device, logger):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    config_name = config.get("_config_name_", "")
    if config_name == "persongen_inference_free":
        logger.info("Free inference mode - skipping dataloaders initialization")
        return None, None

    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    assert (
        config.train_dataset_name in config.datasets.train
    ), f"Choosen dataset {config.train_dataset_name} not in config. Available datasets: {config.datasets.train.keys()}"

    # Создаем train dataset с transforms
    train_dataset_config = config.datasets.train[config.train_dataset_name]
    train_dataset_config.instance_transforms = (
        config.transforms.instance_transforms.train
    )
    train_dataset = instantiate(train_dataset_config)

    val_datasets = {}
    for val_dataset_name in config.val_datasets_names:
        assert (
            val_dataset_name in config.datasets.val
        ), f"Choosen dataset {val_dataset_name} not in config. Available datasets: {config.datasets.val.keys()}"

        # Создаем val dataset с transforms
        val_dataset_config = config.datasets.val[val_dataset_name]
        val_dataset_config.instance_transforms = (
            config.transforms.instance_transforms.val
        )
        val_dataset = instantiate(val_dataset_config)
        val_datasets[val_dataset_name] = val_dataset

    dataloaders = {}
    train_dataloader_config = config.dataloaders.train_default

    train_batch_size = train_dataloader_config.get("batch_size", 1)
    assert train_batch_size <= len(train_dataset), (
        f"The train batch size ({train_batch_size}) cannot "
        f"be larger than the train dataset length ({len(train_dataset)})"
    )

    dataloaders["train"] = instantiate(
        train_dataloader_config,
        dataset=train_dataset,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=True,
        worker_init_fn=set_worker_seed,
    )

    for val_dataset_name, val_dataset in val_datasets.items():
        val_dataloadre_config = config.dataloaders["val_default"]

        assert val_dataloadre_config.batch_size <= len(val_dataset), (
            f"The batch size ({val_dataloadre_config.batch_size}) cannot "
            f"be larger than the dataset length ({len(val_dataset)})"
        )

        val_dataloader = instantiate(
            val_dataloadre_config,
            dataset=val_dataset,
            collate_fn=collate_fn_val,
            drop_last=False,
            shuffle=False,
            worker_init_fn=set_worker_seed,
        )

        dataloaders[val_dataset_name] = val_dataloader

    return dataloaders, batch_transforms
