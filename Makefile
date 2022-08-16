data_dir = /data/datasets/DG


# train hparms
dataset = NICO
algorithm ?= ERM
backbone ?= resnet50
pretrain_path ?= 0
model_path ?= None
checkpoint_freq ?= 0
batch_size ?= 8
epochs ?= 150
reset_epoch ?= 10
res ?= 224
lr ?= 0.001
weight_decay ?= 0.01
optim ?= SGD
lr_scheduler ?= CosineAnnealingLR
hparams = '{\
	"backbone": "$(backbone)", "update_bn": true, "freeze_bn": false, "reset_epoch": $(reset_epoch),\
	"rand_aug": {"M": 10, "N": 4}, "rand_erasing": false, "do_mixstyle1": true,\
	"batch_size": $(batch_size), "epochs": $(epochs), "res": $(res),\
	"lr": $(lr), "weight_decay": $(weight_decay), "resnet_dropout": 0, "nonlinear_classifier": false,\
	"optimizer": "$(optim)", "lr_scheduler": "$(lr_scheduler)",\
	"rho": 0.03\
}'

# sweep h params
output_dir = ./train_output
n_hparams = 1
n_trials = 1
datasets = NICO
algorithms = $(algorithm)


train:
	python -m domainbed.scripts.train\
		--data_dir=$(data_dir)\
        --algorithm $(algorithm)\
        --dataset $(dataset)\
        --test_envs 0 3\
        --output_dir=$(output_dir)/$(backbone)/$(dataset)-$(algorithm)-bs$(batch_size)-lr$(lr)-wd$(weight_decay)\
		--hparams=$(hparams)\
		--save_model_every_checkpoint\
		--pretrain_path=$(pretrain_path)

train_all:
	python -m domainbed.scripts.train\
		--data_dir=$(data_dir)\
        --algorithm $(algorithm)\
        --dataset $(dataset)\
        --output_dir=$(output_dir)/$(backbone)/$(dataset)-$(algorithm)-bs$(batch_size)-lr$(lr)-wd$(weight_decay)-res$(res)-reg-all\
		--hparams=$(hparams)\
		--holdout_fraction 0.05\
		--checkpoint_freq $(checkpoint_freq)\
		--save_model_every_checkpoint\
		--pretrain_path $(pretrain_path)

eval:
	python -m domainbed.scripts.eval\
		--data_dir=$(data_dir)\
        --algorithm $(algorithm)\
        --dataset $(dataset)\
        --output_dir=$(output_dir)/$(backbone)/$(dataset)-$(algorithm)-bs$(batch_size)-lr$(lr)-wd$(weight_decay)-all\
		--model_path $(model_path)\
		--hparams=$(hparams)

table:
	python -m domainbed.scripts.collect_results\
        --input_dir=./train_output/*\
		--exp_parttern .\
		--plot_summary

collect_results:
	python -m domainbed.scripts.collect_results\
        --input_dir=./train_output/*\
		--exp_parttern .\

delete_incomplete:
	python -m domainbed.scripts.sweep delete_incomplete\
        --data_dir=$(data_dir)\
        --output_dir=$(output_dir)/sweep\
        --command_launcher multi_gpu\
        --datasets $(datasets)\
        --n_hparams $(n_hparams)\
        --n_trials $(n_trials)\
        --algorithms $(algorithms)\
        --single_test_envs\
		--hparams $(hparams)

sweep:
	python -m domainbed.scripts.sweep launch\
		--skip_confirmation\
		--data_dir=$(data_dir)\
		--output_dir=$(output_dir)/sweep\
		--command_launcher multi_gpu\
		--datasets $(datasets)\
		--n_hparams $(n_hparams)\
		--n_trials $(n_trials)\
        --algorithms $(algorithms)\
		--single_test_envs\
		--hparams $(hparams)

