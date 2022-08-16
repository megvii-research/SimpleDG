from config.base_config import BaseConfig


class BaseFinetune(BaseConfig):
    # train config
    input_size = (448, 448)
    epochs = 100
    train_batch_size = 128
    val_batch_size = train_batch_size * 2
    lr = train_batch_size / 256 * 0.1


if __name__ == "__main__":
    from pprint import pprint

    config = BaseFinetune.instance()
    print(config.to_md5())
    pprint(config.to_dict())
