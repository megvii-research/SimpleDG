gpu ?= 8
track ?= 1
test_data_dir ?= None
merged_model ?= None

train_scratch:
	torchrun --standalone --nnodes 1 --nproc_per_node ${gpu} main.py -c config/scratch_track$(track)_seed1.py
	torchrun --standalone --nnodes 1 --nproc_per_node ${gpu} main.py -c config/scratch_track$(track)_seed2.py
	torchrun --standalone --nnodes 1 --nproc_per_node ${gpu} main.py -c config/scratch_track$(track)_seed3.py
	torchrun --standalone --nnodes 1 --nproc_per_node ${gpu} main.py -c config/scratch_track$(track)_seed4.py
	torchrun --standalone --nnodes 1 --nproc_per_node ${gpu} main.py -c config/scratch_track$(track)_seed5.py

train_finetune:
	torchrun --standalone --nnodes 1 --nproc_per_node ${gpu} main.py -c config/finetune_track$(track)_seed1.py
	torchrun --standalone --nnodes 1 --nproc_per_node ${gpu} main.py -c config/finetune_track$(track)_seed2.py
	torchrun --standalone --nnodes 1 --nproc_per_node ${gpu} main.py -c config/finetune_track$(track)_seed3.py
	torchrun --standalone --nnodes 1 --nproc_per_node ${gpu} main.py -c config/finetune_track$(track)_seed4.py
	torchrun --standalone --nnodes 1 --nproc_per_node ${gpu} main.py -c config/finetune_track$(track)_seed5.py

ensemble:
	python3 ensemble.py --track $(track)

test:
	python3 test.py --test_data_dir $(test_data_dir) --track $(track) --merged_model $(merged_model)