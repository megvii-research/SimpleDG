import os
import csv
import json
import argparse
from glob import glob
from collections import OrderedDict
import torch
from torch.utils import data
from tqdm.auto import tqdm
from model import build_model
from dataset import NICOTestDataset
from transform import TestTransform
from utils import load_config_from_file


def make_batch(iterable, batch_size=512, drop_last=False):
    iters = len(iterable) // batch_size
    batch = [
        [iterable[(i * batch_size) + j] for j in range(batch_size)]
        for i in range(iters)
    ]
    if not drop_last and len(iterable) % batch_size:
        batch.append(iterable[iters * batch_size :])
    return batch


def build_dataset(track, root, transform, test_data_dir=None):
    if not os.path.exists(test_data_dir):
        track_dir = os.path.join(root, f"track_{track}")
        if track == 1:
            data_dir = os.path.join(track_dir, "public_dg_0416", "public_test_flat")
        elif track == 2:
            data_dir = os.path.join(
                track_dir, "public_ood_0412_nodomainlabel", "public_test_flat"
            )
        else:
            raise NotImplementedError()
    else:
        data_dir = test_data_dir

    image_path_list = glob(f"{data_dir}/*.jpg")
    image_key_list = [path.split(os.sep)[-1] for path in image_path_list]

    dataset = NICOTestDataset(image_path_list, transform)
    return dataset, image_key_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--test_data_dir", type=str, default=None)
    parser.add_argument("-t", "--track", type=int, choices=[1, 2])
    parser.add_argument("-m", "--models", type=str, nargs='+')
    parser.add_argument("-M", "--merged_model", type=str, default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    args = parser.parse_args()

    config_file = f'config/finetune_track{args.track}_seed1.py'
    config = load_config_from_file(config_file).instance()

    # dataset
    dataset, image_key_list = build_dataset(
        args.track, root=config.dataset_root,
        transform=TestTransform(config), test_data_dir=args.test_data_dir
    )
    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.val_loader_workers,
    )
    image_key_loader = make_batch(image_key_list, batch_size=args.batch_size)

    def evaluate_model(state_dict, backbone=config.backbone):
        model = build_model(backbone, config.num_classes)

        weights = OrderedDict()
        for k, v in state_dict.items():
            if "module" in k:
                weights[k.replace("module.", "")] = v
        model.load_state_dict(weights)
        del weights

        model = model.cuda()
        model.eval()

        result = {}
        with torch.no_grad():
            pbar = tqdm(zip(dataloader, image_key_loader), total=len(dataloader))
            for inputs, image_keys in pbar:
                inputs = inputs.cuda()
                predict = model(inputs)
                for idx, key in enumerate(image_keys):
                    result[key] = predict[idx]

        return result

    probs = {}
    if args.merged_model is None:
        for model_path in args.models:
            checkpoint = torch.load(model_path, map_location="cpu")
            prob = evaluate_model(checkpoint["model"])
            for key, p in prob.items():
                probs.setdefault(key, []).append(p)
    else:
        state_dict_list = torch.load(args.merged_model, map_location="cpu")
        for backbone, state_dict in state_dict_list:
            prob = evaluate_model(state_dict, backbone)
            for key, p in prob.items():
                probs.setdefault(key, []).append(p)

    prediction = {}
    for key, p in probs.items():
        prediction[key] = torch.stack(p).mean(dim=0).argmax().item()

    result_file = os.path.join('outputs', f'track_{args.track}', 'prediction.csv')
    with open(result_file, "w") as f:
        w = csv.writer(f)
        w.writerows(prediction.items())

    print(f"dump result to {result_file}")

    with open(result_file.replace('csv', 'json'), 'w') as f:
        json.dump(prediction, f)


if __name__ == "__main__":
    main()
