from config.base_config import BaseConfig


class BaseScratch(BaseConfig):
    # train config
    input_size = (224, 224)
    epochs = 300
    train_batch_size = 512
    val_batch_size = train_batch_size * 2
    lr = train_batch_size / 256 * 0.1

    # model config
    pretrained_backbone = None


if __name__ == "__main__":
    from pprint import pprint

    config = BaseScratch.instance()
    print(config.to_md5())
    pprint(config.to_dict())
