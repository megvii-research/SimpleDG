# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections


import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys

import numpy as np
import tqdm
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed import model_selection
from domainbed.lib.query import Q
import warnings
from clearml import Task


def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "X"
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    if latex:
        return mean, err, "{:.1f} $\\pm$ {:.1f}".format(mean, err)
    else:
        return mean, err, "{:.1f} +/- {:.1f}".format(mean, err)

def print_table(table, header_text, row_labels, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")

def print_results_tables(records, selection_method, args):
    latex = args.latex
    """Given all records, print a results table for each dataset."""
    grouped_records = reporting.get_grouped_records(records).map(lambda group:
        {
            **group,
            **selection_method.every_acc_dict(group["records"]),
        }
    ).filter(lambda g: g["target_in"] is not None)

    # read algorithm names and sort (predefined order)
    # alg_names = Q(records).select("args.algorithm").unique()
    # alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
    #     [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    acc_domains = ['source_in', 'source_out', 'target_in', 'target_out']

    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))
        test_envs = range(datasets.num_environments(dataset))
        alg_names = Q(records).filter_equals('args.dataset', dataset).select('args.algorithm').unique()

        for acc_domain in acc_domains:
            table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
            for i, algorithm in enumerate(alg_names):
                means = []
                exp_counts = []
                checkpoint_counts = []
                for j, test_env in enumerate(test_envs):
                    if acc_domain.startswith('source_'):
                        choosed_grouped_records = grouped_records.filter_equals(
                            "dataset, algorithm",
                            (dataset, algorithm)
                        ).filter(lambda g: test_env not in eval(g['test_envs']))
                        # print(len(choosed_grouped_records), (dataset, algorithm, test_env))
                    else:
                        choosed_grouped_records = grouped_records.filter_equals(
                            "dataset, algorithm",
                            (dataset, algorithm)
                        ).filter(lambda g: test_env in eval(g['test_envs']))
                        # print(len(choosed_grouped_records), (dataset, algorithm, test_env))
                    if acc_domain.endswith('_in'):
                        acc_key = f'env{j}_in_acc'
                    else:
                        acc_key = f'env{j}_out_acc'
                    trial_accs = choosed_grouped_records.select(acc_key)
                    mean, err, table[i][j] = format_mean(trial_accs, latex)
                    means.append(mean)
                    exp_counts.append(len(choosed_grouped_records))
                    checkpoint_counts.append(sum(choosed_grouped_records.map(lambda group: len(group['records']))))
                if None in means:
                    # means = [mean for mean in means if mean is not None]
                    # table[i][-1] = "{:.1f} ({})".format(sum(means) / len(means), len(means))
                    table[i][-1] = "X"
                else:
                    table[i][-1] = "{:.1f}".format(sum(means) / len(means))
                table[i].append("{:.1f}".format(sum(exp_counts)/len(means)))
                table[i].append("{:.1f}".format(sum(checkpoint_counts)/len(means)))
            col_labels = [
                "Algorithm",
                *datasets.get_dataset_class(dataset).ENVIRONMENTS,
                "Avg",
                "Exps",
                "Checkpoints"
            ]
            header_text = (f"Dataset: {dataset}, "
                f"acc_domain: {acc_domain}, "
                f"model selection method: {selection_method.name}")
            print_table(table, header_text, alg_names, list(col_labels),
                colwidth=15, latex=latex)

        # Print an "summary" table
        if latex:
            print()
            print("\\subsubsection{Summary}")

        table = [[None for _ in [*acc_domains]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            choosed_grouped_records = grouped_records.filter_equals(
                "algorithm, dataset",
                (algorithm, dataset)
            ).group("trial_seed")
            for j, acc_domain in enumerate(acc_domains):
                trial_averages = choosed_grouped_records.map(lambda trial_seed, group:
                        group.select(acc_domain).mean()
                )
                mean, err, table[i][j] = format_mean(trial_averages, latex)
                means.append(mean)
            exp_count = sum(choosed_grouped_records.map(lambda trial_seed, group: len(group)))
            checkpoint_counts = sum(choosed_grouped_records.map(lambda trial_seed, groups: 
                sum(groups.map(lambda group: len(group['records'])))
            ))
            table[i].append("{:.1f}".format(len(choosed_grouped_records)))
            table[i].append("{:.1f}".format(exp_count))
            table[i].append("{:.1f}".format(checkpoint_counts))

        col_labels = ["Algorithm", *acc_domains, "Trials", 'Hparams', 'Checkpoints']
        header_text = f"Summary, model selection method: {selection_method.name}"
        print_table(table, header_text, alg_names, col_labels, colwidth=15,
            latex=latex)

    def plot():
        print('plot')
        for dataset in dataset_names:
            project_name = 'DomainGeneralization/domainbed/anylysis/{}'.format(dataset)
            project_id = Task.get_project_id(project_name)
            task_ids = []
            for i, algorithm in enumerate(alg_names):
                task_name = algorithm
                clearml_task = Task.init(
                    project_name=project_name, task_name=task_name,
                    auto_resource_monitoring=False, auto_connect_arg_parser=False,
                    auto_connect_frameworks=False, auto_connect_streams=False)
                clearml_logger = clearml_task.get_logger()

                choosed_grouped_records = grouped_records.filter_equals(
                    "dataset, algorithm",
                    (dataset, algorithm)
                )
                scatter2d_dict = collections.defaultdict(list)
                for group in choosed_grouped_records:
                    print('algorithm', group['algorithm'])
                    for r in group['records']:
                        results = selection_method._step_acc(r)
                        for key, val in results.items():
                            scatter2d_dict[key].append([r['step'], val])
                        scatter2d_dict['lr'].append([r['step'], r['lr']])
                for key, scatter2d in scatter2d_dict.items():
                    if key in ['lr']:
                        clearml_logger.report_scatter2d(key, key,
                            iteration=0, scatter=sorted(scatter2d, key=lambda x: x[0]),
                            xaxis="step", yaxis=key, mode='lines+markers')
                    else:
                        clearml_logger.report_scatter2d("acc", key,
                            iteration=0, scatter=sorted(scatter2d, key=lambda x: x[0]),
                            xaxis="step", yaxis="acc", mode='lines+markers')
                task_ids.append(clearml_task.task_id)
                clearml_task.close()
            html = 'https://clearis.iap.hh-b.brainpp.cn/projects/{}/compare-experiments;ids='.format(project_id)
            html += ','.join(task_ids)
            html += '/scalars/graph?scalars=graph'
            print(project_name, html)
        return html

    def plot_summary():
        print('plot_summary')
        show_keys = ['batch_size', 'lr', 'weight_decay']

        project_name = 'DomainGeneralization/domainbed/sumarray'
        task_name = args.exp_parttern[0]
        if task_name == '.':
            task_name = os.path.basename(os.path.realpath('.'))
        clearml_task = Task.init(
            project_name=project_name, task_name=task_name,
            auto_resource_monitoring=False, auto_connect_arg_parser=False,
            auto_connect_frameworks=False, auto_connect_streams=False)
        clearml_logger = clearml_task.get_logger()

        results = collections.defaultdict(list)
        for dataset in dataset_names:
            for i, algorithm in enumerate(alg_names):
                choosed_grouped_records = grouped_records.filter_equals(
                    "dataset, algorithm",
                    (dataset, algorithm)
                )
                assert len(choosed_grouped_records) == 1
                group = choosed_grouped_records[0]

                results['algorithm'].append(algorithm)
                for key in show_keys:
                    results[key].append(group['records'][0]['hparams'][key])
                for j, acc_domain in enumerate(acc_domains):
                    if len(group['records']) < 1:
                        results[acc_domain].append(None)
                    else:
                        results[acc_domain].append(group[acc_domain])
                results['checkpoints'].append(len(group['records']))
        df = pd.DataFrame(results)
        clearml_logger.report_text(df)

        batch_size_list = sorted(df.batch_size.unique())
        weight_decay_list = sorted(df.weight_decay.unique())
        lr_list = sorted(df.lr.unique())

        # domain acc table
        def fixna(acc):
            if pd.isna(acc):
                return None
            return acc
        for j, acc_domain in enumerate(acc_domains):
            x = []
            for lr in lr_list:
                x.append([])
                for batch_size in batch_size_list:
                    for weight_decay in weight_decay_list:
                        df_tmp = df[(df.batch_size==batch_size) &
                                    (df.weight_decay==weight_decay) &
                                    (df.lr==lr)]
                        acc = fixna(df_tmp[acc_domain].max())
                        x[-1].append(acc)
            fig = px.imshow(x,
                            text_auto='.2f',
                            labels=dict(y="lr", x="bs|wd", color="acc"),
                            y=[str(v) for v in lr_list],
                            x=[f'{bs}|{wd}' for bs in batch_size_list for wd in weight_decay_list])
            clearml_logger.report_plotly(title=f"domain acc matrix",
                                            series=f"{j} {acc_domain}", iteration=0, figure=fig)
        clearml_logger.report_table("domain acc table", "", iteration=0, table_plot=df)

        # # acc scatter
        # fig = go.Figure()
        # for j, domain in enumerate(acc_domains[:]):
        #     fig.add_trace(go.Scatter(x=df['batch_size'],  y=df[domain], name=domain,
        #                              text=df['algorithm'], mode='markers'))
        # clearml_logger.report_plotly(title=f"acc scatter", series=f"", iteration=0, figure=fig)

        # domain acc trace
        dft = df.T
        fig = go.Figure()
        for i in dft:
            fig.add_trace(go.Scatter(x=acc_domains, y=dft[i][acc_domains],
                                mode='lines+markers', name=dft[i]['algorithm'],
                                text='{} ({})'.format(dft[i]['algorithm'], dft[i]['checkpoints']),))
        clearml_logger.report_plotly(f"domain acc trace", "all", iteration=0, figure=fig)
        for batch_size in batch_size_list:
            fig = go.Figure()
            for i in dft:
                if dft[i]['batch_size'] == batch_size:
                    fig.add_trace(go.Scatter(x=acc_domains, y=dft[i][acc_domains],
                                        mode='lines+markers', name=dft[i]['algorithm'],
                                        text='{} ({})'.format(dft[i]['algorithm'], dft[i]['checkpoints'])))
            clearml_logger.report_plotly(f"domain acc trace", f"bs-{batch_size}", iteration=0, figure=fig)

        # generalization error
        for size_name, size in [
            ('source_in', (df['source_in']**7)+0.05),
            ('target_in', df['target_in'])]:
            df['generalization_error1'] = df.source_in-df.source_out
            df['generalization_error2'] = df.source_out-df.target_in
            df['bs'] = df['batch_size'].astype('string')
            fig = px.scatter(df, x="generalization_error1", y="generalization_error2",
                            color='bs', trendline="ols", size=size,
                            hover_name='algorithm', hover_data=acc_domains+['checkpoints'])
            clearml_logger.report_plotly(f"generalization error (size: {size_name})", f"", iteration=0, figure=fig)

        # domain acc scatter
        df['algo'] = df['algorithm'].map(lambda x:x.split('.')[0])
        fig = px.scatter(df, x="source_out", y="target_in",
                        color='algo', trendline="ols", size='source_in',
                        hover_name='algorithm', hover_data=acc_domains+['checkpoints'])
        clearml_logger.report_plotly(f"domain acc scatter", f"", iteration=0, figure=fig)

        clearml_task.close()

    if args.plot:
        plot()
    if args.plot_summary:
        plot_summary()

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--exp_parttern", type=str, nargs='+', default=['./'])
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_summary", action="store_true")
    args = parser.parse_args()

    results_file = "results.tex" if args.latex else "results.txt"

    if not os.path.exists(args.input_dir):
        os.makedirs(args.input_dir)
    sys.stdout = misc.Tee(os.path.join(args.input_dir, results_file), "w")

    records = reporting.load_records(args.input_dir, args.exp_parttern)

    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\section{Full DomainBed results}")
        print("% Total records:", len(records))
    else:
        print("Total records:", len(records))

    SELECTION_METHODS = [
        model_selection.IIDAccuracySelectionMethod,
        # model_selection.LastEpochIIDAccuracySelectionMethod,
        # model_selection.LeaveOneOutSelectionMethod,
        # model_selection.OracleSelectionMethod,
    ]

    for selection_method in SELECTION_METHODS:
        if args.latex:
            print()
            print("\\subsection{{Model selection: {}}}".format(
                selection_method.name))
        print_results_tables(records, selection_method, args)

    if args.latex:
        print("\\end{document}")
