import argparse
from copy import deepcopy
import torch
import json
import os

from domainbed import algorithms, hparams_registry, datasets
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.lib import misc

from tqdm import tqdm
from glob import glob
from refile import smart_glob, smart_open


def simple_moving_average(args):
    hparams = hparams_registry.default_hparams('ERM', 'NICO')
    hparams['steps_per_epoch'] = 5000
    hparams['backbone'] = args.backbone
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    resolution = hparams.get('res', 224)
    algo_class = algorithms.get_algorithm_class('SWA')
    algo = algo_class((3, resolution, resolution), 60, 1, hparams)

    algo.sma.sma_start_iter = args.sma_start_iter
    for model_path in tqdm(sorted(glob(args.input_dir + '/model_step*'), key=lambda p: int(p.split('/')[-1][len('model_step'): -len('.pkl')]))):
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_dict']
        algo.load_state_dict(state_dict, load_sma=False)

        algo.sma.update(algo.network)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    save_model_path = os.path.join(args.output_dir, 'model.pkl')
    torch.save({'model_dict': algo.state_dict()}, save_model_path)
    print('save averaged model to', save_model_path)

def model_soup(args):
    hparams = hparams_registry.default_hparams('ERM', 'NICO')
    hparams['steps_per_epoch'] = 5000
    hparams['backbone'] = args.backbone
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    
    in_splits = []
    out_splits = []
    dataset = vars(datasets)[args.dataset](args.data_dir, [], hparams)
    
    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        in_weights, out_weights = None, None
        # in_splits.append((in_, in_weights))
        if len(out) > 0:
            out_splits.append((out, out_weights))

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]

    num_domains = len(dataset)
    algo_class = algorithms.get_algorithm_class('SWA')
    algo = algo_class(dataset.input_shape, dataset.num_classes, num_domains, hparams)
    algo.to(device)

    def get_model_list():
        model_list = []
        for d in args.input_dirs:
            if d.startswith('s3'):
                model_key = '/epoch_*'
                sort_by_reversed_step = lambda p: -int(p.split('/')[-1][len('epoch_'):])
            else:
                model_key = '/model_step*'
                sort_by_reversed_step = lambda p: -int(p.split('/')[-1][len('model_step'): -len('.pkl')])
            for model_path in sorted(smart_glob(d + model_key), key=sort_by_reversed_step)[:args.topk]:
                model_list.append(model_path)

        return model_list

    def get_algo_val_acc(algorithm):
        val_accs = []
        for name, loader, weights in zip(eval_loader_names, eval_loaders, eval_weights):
            acc, loss = misc.accuracy(algorithm, name, loader, weights, device)
            val_accs.append(acc)

        return sum(val_accs) / len(val_accs)

    def convert_ddp_state_dict(state_dict):
        cvt_state_dict = {}
        cvt_state_dict.update({k.replace('module', 'network.0.network'): v for k, v in state_dict['model'].items() if 'fc' not in k})
        cvt_state_dict.update({k.replace('module.fc', 'network.1'): v for k, v in state_dict['model'].items() if 'fc' in k})
        return {'model_dict': cvt_state_dict}

    def get_state_dict(model_path):
        state_dict = torch.load(smart_open(model_path, 'rb'), map_location=torch.device('cpu'))
        if 'model_dict' not in state_dict:
            state_dict = convert_ddp_state_dict(state_dict)
        return state_dict['model_dict']

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model_val_acc_save_path = os.path.join(args.output_dir, 'model_val_acc.json')
    if os.path.exists(model_val_acc_save_path):
        model_val_acc = json.load(open(model_val_acc_save_path))
    else:
        model_val_acc = {}

    model_list = get_model_list()
    for model_path in tqdm(model_list, desc='eval models'):
        if model_path not in model_val_acc:
            state_dict = get_state_dict(model_path)
            algo.load_state_dict(state_dict)
            val_acc = get_algo_val_acc(algo)
            model_val_acc[model_path] = val_acc
        print(model_path, model_val_acc[model_path])

    json.dump(model_val_acc, open(model_val_acc_save_path, 'w'), indent=4)

    max_val_acc = None
    model_soup = []
    def show_model_soup():
        print('max val acc:', max_val_acc)
        print('model soup:')
        for m in model_soup:
            print(m)

    for model_path in tqdm(sorted(model_val_acc, key=lambda m: -model_val_acc[m]), desc='model soup'):
        if model_path not in model_list: continue
        state_dict = get_state_dict(model_path)
        if max_val_acc is None:
            max_val_acc = model_val_acc[model_path]
            algo.load_state_dict(state_dict)
            algo.sma.update(algo.network)
            model_soup.append(model_path)
            show_model_soup()
        else:
            greedy_algo = deepcopy(algo)
            greedy_algo.load_state_dict(state_dict, load_sma=False)
            greedy_algo.sma.update(greedy_algo.network)
            val_acc = get_algo_val_acc(greedy_algo)
            print('greedy soup val acc:', val_acc)
            if val_acc > max_val_acc - 0.001:
                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                algo = greedy_algo
                model_soup.append(model_path)
                show_model_soup()

    save_model_path = os.path.join(args.output_dir, 'model.pkl')
    torch.save({'model_dict': algo.state_dict()}, save_model_path)
    print('save model soup to', save_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/datasets/DG')
    parser.add_argument('--dataset', type=str, default="NICO2", choices=['NICO', 'NICO2'])
    parser.add_argument('--holdout_fraction', type=float, default=0.05)
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--backbone', type=str, default="resnet50")
    parser.add_argument('--input_dirs', type=str, default=[], nargs='+')
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--sma_start_iter', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default="train_output/sweep")
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')

    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model_soup(args)