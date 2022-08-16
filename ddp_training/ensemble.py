import os
import glob
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=int, choices=[1, 2])
    args = parser.parse_args()

    model_list = glob.glob(f'outputs/track_{args.track}/resnet101/seed*/finetune/models/best')
    if len(model_list) < 5:
        print('experiment number less than 5, please check')

    state_dict_list = []
    for model_path in model_list:
        backbone = model_path.split('/')[-5]
        checkpoint = torch.load(model_path, map_location='cpu')
        if checkpoint['best_val_acc'] < 0.80:
            print(f'best val acc is supposed to be greater than 0.80, but got {checkpoint["best_val_acc"]}')
        state_dict_list.append((backbone, checkpoint['model']))

    output_dir = f'outputs/track_{args.track}/ensemble'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    merged_model_path = os.path.join(output_dir, 'merged_model')
    print('dump merged model to', merged_model_path)
    torch.save(state_dict_list, merged_model_path)


if __name__ == '__main__':
    main()
