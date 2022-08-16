from config._base import ConfigBase


class BaseConfig(ConfigBase):
    # dataset config
    dataset_root = "/data/datasets/DG/NICO"
    num_classes = 60
    val_ratio = 0.05
    train_loader_workers = 16
    val_loader_workers = 8
    pin_memory = True

    # train config
    continue_training = True
    warmup_epochs = 10
    weight_decay = 5e-4

    # augmentation
    rand_aug_magnitude = 15
    rand_aug_num_layers = 4
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    mixup = True
    mixup_prob = 0.8
    mixup_alpha = 10
    fmix_alpha = 10
    cutmix_alpha = 10
    label_smoothing = 0.1

    # save
    save_latest_checkpoint_interval = 1
    save_checkpoint_interval = 50
    log_interval = 10


if __name__ == "__main__":
    from pprint import pprint

    config = BaseConfig.instance()
    print(config.to_md5())
    pprint(config.to_dict())
