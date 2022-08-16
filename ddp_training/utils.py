import os
import time
import logging
import torch
import torch.distributed as dist
from functools import lru_cache
from importlib.util import spec_from_file_location, module_from_spec


def load_config_from_file(config_file, class_name="Config"):
    spec = spec_from_file_location("config", config_file)
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    config_class = getattr(m, class_name)
    return config_class


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


class Clock:
    def __init__(self, ndigits=3) -> None:
        self.start_time = 0
        self.stop_time = 0
        self.checkpoint_time = 0
        self.total_time = 0
        self.ndigits = ndigits

    def start(self):
        self.start_time = time.time()
        self.checkpoint_time = self.start_time

    def lap(self):
        lap_time = round(time.time() - self.checkpoint_time, self.ndigits)
        self.checkpoint_time = time.time()
        return lap_time

    def stop(self):
        self.stop_time = time.time()
        self.total_time = round(self.stop_time - self.start_time, self.ndigits)

    def get(self):
        return self.total_time

    def reset(self):
        self.__init__(ndigits=self.ndigits)


class AverageMeter:
    def __init__(self, ndigits=5):
        self.sum_value = 0
        self.num_iter = 0
        self.ndigits = ndigits

    def update(self, value):
        self.sum_value += value
        self.num_iter += 1

    def get(self):
        return round(self.sum_value / self.num_iter, self.ndigits)

    def clear(self):
        self.__init__(ndigits=self.ndigits)

    def pop(self):
        average_value = self.get()
        self.clear()
        return average_value

    def sync(self):
        t = torch.tensor(
            [self.sum_value, self.num_iter], dtype=torch.float64, device="cuda"
        )
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum_value = t[0]
        self.num_iter = int(t[1])


class Accuracy:
    def __init__(self, ndigits=5):
        self.num_correct = 0
        self.num_sample = 0
        self.ndigits = ndigits

    @torch.no_grad()
    def update(self, predict, target):
        pred = torch.argmax(predict, dim=1)
        self.num_correct += (pred == target).sum().item()
        self.num_sample += target.size(0)

    def get(self):
        return round(self.num_correct / self.num_sample, self.ndigits)

    def clear(self):
        self.__init__(ndigits=self.ndigits)

    def pop(self):
        average_value = self.get()
        self.clear()
        return average_value

    def sync(self):
        t = torch.tensor(
            [self.num_correct, self.num_sample], dtype=torch.int64, device="cuda"
        )
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.num_correct = t[0]
        self.num_sample = t[1]


def load_pretrained_backbone(pretrained_backbone):
    with open(pretrained_backbone, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    network_params = checkpoint["model"]
    return network_params


def save_checkpoint(checkpoint_dir, checkpoint_name, checkpoint):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    try:
        with open(checkpoint_path, "wb") as f:
            torch.save(checkpoint, f)
        return True
    except:
        return False


def load_checkpoint(checkpoint_dir, checkpoint_name):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
            return checkpoint
        except:
            pass
    return None


@lru_cache(maxsize=1)
def get_logger(config):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=config.local_log_path, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
