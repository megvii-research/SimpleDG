# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
from glob import glob
import json
import os
import random
import sys
import time
from math import ceil

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import torch.nn as nn

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader


torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    # parser.add_argument('--steps', type=int, default=None,
    #     help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=0,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.05)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    def load_checkpoint():
        model_pattern = os.path.join(args.output_dir, 'model_step*')
        models = glob(model_pattern)
        if len(models) == 0:
            return None, 0
        steps = [int(os.path.basename(model).split('.pkl')[0].strip('model_step')) for model in models]
        idx = np.argmax(steps)
        print('Load model', models[idx])
        algorithm_dict = torch.load(models[idx])["model_dict"]
        return algorithm_dict, steps[idx]
    if args.pretrain_path and os.path.exists(args.pretrain_path):
        print('loading pretrained model:', args.pretrain_path)
        algorithm_dict = torch.load(args.pretrain_path)['model_dict']
        restart_step = 0
    else:
        algorithm_dict, restart_step = load_checkpoint()

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
        hparams['pretrain_path'] = args.pretrain_path

    if 'do_mixstyle' in hparams:
        hparams['ms_layers'] = [1]
        hparams['ms_p'] = 0.5
        hparams['ms_alpha'] = 0.1
        hparams['ms_eps'] = 1e-6
        hparams['ms_type'] = 'random'

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        if len(out) > 0:
            out_splits.append((out, out_weights))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=max(64, hparams['batch_size']),
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])
    snapshot_steps = ceil(steps_per_epoch / 10)
    steps_per_epoch = ceil(steps_per_epoch)
    print('Steps_per_epoch:', steps_per_epoch)
    hparams['steps_per_epoch'] = steps_per_epoch
    # n_steps = args.steps or dataset.N_STEPS
    n_steps = steps_per_epoch * hparams['epochs']
    checkpoint_freq = args.checkpoint_freq or steps_per_epoch * 10 # or dataset.CHECKPOINT_FREQ

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)
    if algorithm_dict is not None:
        print('load_state_dict')
        model_dict=algorithm.state_dict()
        pretrained_dict = {k: v for k, v in algorithm_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        # algorithm.load_state_dict(model_dict)
        algorithm.load_state_dict(algorithm_dict)
        algorithm.train()
        print(f'load_state_dict done: {len(pretrained_dict)}/{len(model_dict)}')
    algorithm.to(device)

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    for step in range(start_step, n_steps):
        # pretrain
        if step < restart_step:
            algorithm.optimizer.step()
            algorithm.scheduler_step()
            lr = algorithm.get_lr()[0]
            if (step+1) % steps_per_epoch == 0:
                print("Next lr: %.8f step: %.8f epoch: %.8f" % (lr, step, (step+1)/steps_per_epoch))
            continue

        # train
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        step_vals = algorithm.update(minibatches_device)
        algorithm.scheduler_step()
        lr = algorithm.get_lr()[0]
        checkpoint_vals['step_time'].append(time.time() - step_start_time)
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        # snapshot
        if (step+1) % snapshot_steps == 0:
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
                'lr': lr,
            }
            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val[-snapshot_steps:])
            snap_val_str = ''
            for key, val in results.items():
                snap_val_str = snap_val_str + "%s: %.3f, " % (key, val)
            print(snap_val_str)

        # eval
        if ((step+1) % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
                'lr': lr,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            val_loss = []
            for name, loader, weights in evals:
                acc, loss = misc.accuracy(algorithm, name, loader, weights, device)
                results[name+'_acc'] = acc
                if '_out' in name:
                    val_loss.append(loss)

            if args.algorithm in ['SWAD']:
                algorithm.update_and_evaluate(np.mean(val_loss))
                if algorithm.dead_valley:
                    print('SWAD valley is dead, early stop!')
                    break

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            misc.print_row(results_keys, colwidth=12)
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

            if args.algorithm in ['SWA', 'SWAD', 'SWA_Distill']:
                algorithm.reset()
    
    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
