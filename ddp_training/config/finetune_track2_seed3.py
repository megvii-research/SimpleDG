import os
import torch

from config.base_finetune import BaseFinetune


class Config(BaseFinetune):
    # parallel config
    ngpus_per_node = torch.cuda.device_count()
    dist_url = "tcp://localhost:12345"
    world_size = ngpus_per_node
    backend = "nccl"

    # dataset config
    track = 2
    seed = 3

    # model config
    backbone = "resnet101"
    pretrained_backbone = os.path.join(
        os.getcwd(),
        "outputs",
        f"track_{track}",
        backbone,
        f"seed_{seed}",
        "scratch",
        "models",
        "best",
    )

    # path config
    local_output_dir = os.path.join(
        os.getcwd(), "outputs", f"track_{track}", backbone, f"seed_{seed}", "finetune"
    )
    local_log_path = os.path.join(local_output_dir, "worklog.log")
    local_model_dir = os.path.join(local_output_dir, "models")


if __name__ == "__main__":
    from pprint import pprint

    config = Config.instance()
    print(config.to_md5())
    pprint(config.to_dict())
