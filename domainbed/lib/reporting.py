# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections

import json
import os
import glob

import tqdm

from domainbed.lib.query import Q

def load_records(path_pattern, exp_partterns):
    records = []

    exps = []
    for exp_parttern in exp_partterns:
        for exp in glob.glob(exp_parttern):
            if exp in exps:
                continue
            else:
                exps.append(exp)
            exp_path_pattern = os.path.join(exp, path_pattern)
            exp_name = os.path.basename(exp).replace(os.path.basename(exp_parttern).replace('*', ''), '')
            print('exp_name', exp_name)
            for path in glob.glob(exp_path_pattern):
                basename = os.path.basename(path).replace('single_train-', '')
                basename = basename.replace('single_train', '')
                basename = basename.replace('sweep', '')
                
                records_subdir = []
                for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(path))),
                                        ncols=80,
                                        leave=False):
                    results_path = os.path.join(path, subdir, "results.jsonl")
                    try:
                        with open(results_path, "r") as f:
                            for line in f:
                                record = json.loads(line[:-1])
                                record["args"]["algorithm"] = '.'.join([
                                    record["args"]["algorithm"],
                                    exp_name,
                                    basename,
                                    *subdir.split('-')[2:]
                                ])
                                records_subdir.append(record)
                    except IOError:
                        pass
                print('path:', path, 'records:', len(records_subdir))
                records += records_subdir
    records = Q(records)
    return records.sorted(key=lambda r: r["args"]["algorithm"])

def get_grouped_records(records):
    """Group records by (trial_seed, dataset, algorithm, test_envs). Because
    records can have multiple test envs, a given record may appear in more than
    one group."""
    result = collections.defaultdict(lambda: [])
    for r in records:
        group = (r["args"]["trial_seed"],
            r["args"]["dataset"],
            r["args"]["algorithm"],
            str(r["args"]["test_envs"]))
        result[group].append(r)
    return Q([{"trial_seed": t, "dataset": d, "algorithm": a, "test_envs": e,
        "records": Q(r)} for (t,d,a,e),r in result.items()])
