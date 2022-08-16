# NICO challenge code
This is the training and test code for [NICO challenge](https://nicochallenge.com/) track1 and track2.
## preperation
Download the NICO++ dataset [HERE](https://nicochallenge.com/dataset), unzip them to `$data_dir` and fill it to `config/base_config.py` (line 6: dataset_root = `$data_dir`). The data should have the following structure:  
```
$data_dir
├── track_1  
│   ├── dg_label_id_mapping.json  
│   └── public_dg_0416  
│       ├── public_test_flat  
│       └── train  
├── track_2  
│   ├── ood_label_id_mapping.json  
│   └── public_ood_0412_nodomainlabel  
│       ├── public_test_flat  
│       └── train  
```
Create the required environment
```
conda env create -f environment.yaml
conda activate nico
```
## train
1. pretrain for track1/2 with low input resolution for several times with different random seeds  
```bash
make train_scratch track=1
```
2. finetune for track1/2 with high input resolution  
```bash
make train_finetune track=1
```
3. collect the best models according to validation accuracy and ensemble as a merged model  
```bash
make ensemble track=1
```
## test
Specify `$test_data_dir`, the private flat test dir, and `$merged_model`, the result will be dumped in `outputs/{track}/prediction.csv`
```bash
make test test_data_dir=$test_data_dir track=1 merged_model=outputs/track1/ensemble/merged_model
```