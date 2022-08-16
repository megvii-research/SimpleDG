import argparse
import torch
import json
import os

import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def filter_state_dict(state_dict):
    filtered_state_dict = {k.replace('module.', ''): v for k, v in state_dict['model_dict'].items() if k.startswith('network')}
    return filtered_state_dict

def convert_ddp_state_dict(state_dict):
    cvt_state_dict = {}
    cvt_state_dict.update({k.replace('module', 'network.0.network'): v for k, v in state_dict['model'].items() if 'fc' not in k})
    cvt_state_dict.update({k.replace('module.fc', 'network.1'): v for k, v in state_dict['model'].items() if 'fc' in k})
    
    return cvt_state_dict

def get_state_dict(model_path):
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if 'model_dict' in state_dict:
        state_dict = filter_state_dict(state_dict)
    elif 'model' in state_dict:
        state_dict = convert_ddp_state_dict(state_dict)

    return state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_models', type=str, default=[], nargs='+')
    parser.add_argument('--output_dir', type=str, default='train_output/ensemble')

    args = parser.parse_args()

    mean_prob = {}
    model_params = []
    for model_path in tqdm(args.input_models):
        d = os.path.dirname(model_path)
        prob_path = os.path.join(d, 'probability.json')
        assert os.path.exists(prob_path), f'{prob_path} not exists, run eval first'
        prob = json.load(open(prob_path))
        for filename, logits in prob.items():
            mean_prob.setdefault(filename, []).append(F.softmax(torch.tensor(logits), dim=0).numpy())

        backbone = os.path.realpath(model_path).split('/')[-3]
        state_dict = get_state_dict(model_path)
        if any(['fc' in k for k in state_dict]):
            backbone = backbone + '_mlp'
        model_params.append((backbone, state_dict))

    probability = {}
    prediction = {}
    for filename, logits_list in mean_prob.items():
        probability[filename] = np.mean(logits_list, axis=0).tolist()
        prediction[filename] = int(np.mean(logits_list, axis=0).argmax())

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    prob_path = os.path.join(args.output_dir, 'probability.json')
    json.dump(probability, open(prob_path, 'w'))
    result_path = os.path.join(args.output_dir, 'prediction.json')
    print('dump result to', os.path.realpath(result_path))
    json.dump(prediction, open(result_path, 'w'))
    model_path = os.path.join(args.output_dir, 'model.pkl')
    torch.save(model_params, model_path)
