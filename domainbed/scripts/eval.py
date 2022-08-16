import argparse
import torch
import json
import os

from domainbed import datasets, algorithms
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader

from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from domainbed import networks

def forward1(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    if self.do_mixstyle and (1 in self.hparams["ms_layers"]) and self.training:
        x = self.mixstyle(x)
    x = self.layer2(x)

    return x

def forward2(self, x):
    if self.do_mixstyle and (2 in self.hparams["ms_layers"]) and self.training:
        x = self.mixstyle(x)
    x = self.layer3(x)
    if self.do_mixstyle and (3 in self.hparams["ms_layers"]) and self.training:
        x = self.mixstyle(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return self.fc(x)

def load_state_dict(model, state_dict, key):
    if 'model_dict' in state_dict:
        state_dict = state_dict['model_dict']
    return model.load_state_dict({k[len(key):]: v for k, v in state_dict.items() if k.startswith(key)})

class StyleTransfer(torch.nn.Module):
    def __init__(self, dataset, hparams, state_dict, feature_size):
        super().__init__()
        self.pre_featurizer = networks.Featurizer(dataset.input_shape, hparams)
        load_state_dict(self.pre_featurizer, state_dict, 'featurizer.')
        forward = forward1.__get__(self.pre_featurizer.network, self.pre_featurizer.network.__class__)
        setattr(self.pre_featurizer.network, 'forward', forward)

        self.post_featurizer = networks.Featurizer(dataset.input_shape, hparams)
        load_state_dict(self.post_featurizer, state_dict, 'featurizer.')
        forward = forward2.__get__(self.post_featurizer.network, self.post_featurizer.network.__class__)
        setattr(self.post_featurizer.network, 'forward', forward)

        self.classifier = networks.Classifier(feature_size, 60)
        load_state_dict(self.classifier, state_dict, 'classifier.')

    def predict(self, x):
        feat = self.pre_featurizer.forward(x)
        feat_mean = feat.mean(dim=(2,3), keepdim=True)
        feat_std = feat.std(dim=(2,3), keepdim=True)

        feat_norm = (feat - feat_mean) / (feat_std + 1e-6)
        cross_feat = feat_norm.unsqueeze(1) * feat_std.unsqueeze(0) + feat_mean.unsqueeze(0)
        cross_feat = cross_feat.reshape([-1] + list(cross_feat.shape[2:]))
        cls_feat = self.post_featurizer(cross_feat)
        cross_pred = self.classifier(cls_feat)
        
        n = feat.shape[0]
        cross_pred = cross_pred.reshape(n, n, -1)
        
        return cross_pred.mean(dim=1)

def evaluate(algo_class, loader, backbone, algo_dict):
    algo_class = algorithms.get_algorithm_class(args.algorithm)
    if 'mlp' in backbone:
        backbone = backbone.split('_mlp')[0]
        hparams['mlp'] = 512
    else:
        hparams['mlp'] = None
    hparams['backbone'] = backbone
    algo = algo_class(dataset.input_shape, dataset.num_classes, 1, hparams)

    algo.load_state_dict(algo_dict)
    print(f'loaded model: {model_path}, {len(algo_dict)}/{len(algo.state_dict())}')

    if hparams.get('do_mixstyle', False):
        algo = StyleTransfer(dataset, hparams, algo_dict, algo.feature_size)

    if torch.cuda.is_available():
        device = 'cuda'
        algo.cuda()
    else:
        device = 'cpu'

    algo.eval()
    prediction = {}
    probability = {}
    logits_prob = {}
    labels = {}
    with torch.no_grad():
        for l in loader:
            for x, y, filenames in tqdm(l):
                x = x.to(device)
                if args.TTA:
                    b, n, c, h, w = x.shape
                    x = x.reshape(b*n, c, h, w)
                    p = F.softmax(algo.predict(x), dim=1).cpu().numpy()
                    p = p.reshape(b, n, dataset.num_classes).mean(axis=1)
                else:
                    logits = algo.predict(x)
                    logits_p = logits.cpu().numpy()
                    p = F.softmax(logits, dim=1).cpu().numpy()

                pred_label = p.argmax(axis=1)
                for i in range(p.shape[0]):
                    prediction[filenames[i]] = int(pred_label[i])
                    probability[filenames[i]] = [float(prob) for prob in p[i]]
                    logits_prob[filenames[i]] = [float(prob) for prob in logits_p[i]]
                    labels[filenames[i]] = int(y[i].item())

    # assert len(prediction) == len(dataset)

    # return logits_prob, probability, labels
    return prediction, probability, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/datasets/DG')
    parser.add_argument('--dataset', type=str, default="NICO", choices=['NICO', 'NICO2'])
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--output_dir', type=str, default="train_output/sweep")
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--TTA', action='store_true')

    args = parser.parse_args()

    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    hparams['steps_per_epoch'] = 5000
    hparams['backbone'] = args.model_path.split('/')[-3]
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    
    if 'do_mixstyle' in hparams:
        hparams['ms_layers'] = [1, 2]
        hparams['ms_p'] = 1
        hparams['ms_alpha'] = 0.1
        hparams['ms_eps'] = 1e-6
        hparams['ms_type'] = 'random'

    test_dataset = args.dataset + '_Test'
    test_env = [0, 3]
    resolution = hparams.get('res', 224)
    if args.dataset in vars(datasets):
        # dataset = vars(datasets)[test_dataset](args.data_dir, args.TTA, resolution)
        dataset = vars(datasets)[args.dataset](args.data_dir, test_env, hparams)
    else:
        raise NotImplementedError

    loader = [FastDataLoader(dataset=dataset[i], batch_size=16, num_workers=dataset.N_WORKERS) for i in test_env]
    algo_class = algorithms.get_algorithm_class(args.algorithm)

    if args.model_path and os.path.exists(args.model_path):
        model_path = args.model_path
        args.output_dir = os.path.dirname(model_path)
    else:
        model_path = os.path.join(args.output_dir, 'model.pkl')

    if os.path.exists(model_path):
        model_params = torch.load(model_path, map_location=torch.device('cpu'))
        if isinstance(model_params, dict):
            algo_dict = model_params['model_dict']
            prediction, probability, labels = evaluate(algo_class, loader, hparams['backbone'], algo_dict)
        elif isinstance(model_params, list):
            prob_list, logits_list = {}, {}
            for backbone, algo_dict in model_params:
                logits, prob = evaluate(algo_class, loader, backbone, algo_dict)
                for filename, p in logits.items():
                    logits_list.setdefault(filename, []).append(p)
                for filename, p in prob.items():
                    prob_list.setdefault(filename, []).append(p)
            probability = {filename: np.array(p).mean(axis=0).tolist() for filename, p in prob_list.items()}
            prediction = {filename: int(np.array(p).argmax()) for filename, p in probability.items()}
            logits_probability = {filename: np.array(p).mean(axis=0).tolist() for filename, p in logits_list.items()}
            logits_prediction = {filename: int(np.array(p).argmax()) for filename, p in logits_probability.items()}
        else:
            raise RuntimeError('unsupported model param')
    else:
        raise RuntimeError(f'model not exists: {model_path}')

    result_path = os.path.join(args.output_dir, 'prediction.json')
    prob_path = os.path.join(args.output_dir, 'probability.json')
    label_path = os.path.join(args.output_dir, 'label.json')
    if args.TTA:
        result_path = result_path.replace('prediction', 'prediction_tta')
        prob_path = prob_path.replace('probability', 'probability_tta')

    print('dump result to', os.path.realpath(result_path))
    json.dump(prediction, open(result_path, 'w'))
    json.dump(probability, open(prob_path, 'w'))
    json.dump(labels, open(label_path, 'w'))
    # json.dump(logits_prediction, open(result_path.replace('prediction', 'logits_prediction'), 'w'))
    # json.dump(logits_probability, open(prob_path.replace('prob', 'logits_prob'), 'w'))
