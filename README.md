# SimpleDG
This is the training and test code of our method for [NICO challenge](https://nicochallenge.com/) track1 and track2.

## preperation
Download the NICO++ dataset [HERE](https://nicochallenge.com/dataset), unzip them to `$data_dir` and fill it to Makefile (line 1: data_dir = `$data_dir`). The data should have the following structure:  
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

## train previous DG algorithms
- train a single algorithm with specific parameters
```bash
make train algorithm=ERM
```

- train multiple algorithms with hyperparameter searching, change the value of $(algorithms) in Makefile with what you want
```bash
make sweep
```

## show results
show and compare the results of different algorithms
```bash
make collect_results
```

## train with our method using ddp
enter the ddp_training directory and follow the steps in [README.md](ddp_training/README.md)


## Acknowledgment
We conduct experiments on the base of [DomainBed](https://github.com/facebookresearch/DomainBed)
