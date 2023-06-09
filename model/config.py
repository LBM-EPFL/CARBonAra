# import sys
from datetime import datetime


config_data = {
    'dataset_filepath': "datasets/pdb_structures_16384.h5",
    'train_selection_filepath': "datasets/subunits_train_set.txt",
    'test_selection_filepath': "datasets/subunits_test_set.txt",
    'max_ba': 1,
    'max_size': 1024*8,
    'min_num_res': 48,
    'r_noise': 0.75,
    'virt_cb': True,
    'partial': True,
}

config_model = {
    "em": {'N0': 50, 'N1': 32},
    "sum": sum([
        [{'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8}]*8,
        [{'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16}]*8,
        [{'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32}]*8,
        [{'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64}]*8,
    ], []),
    "spl": {'N0': 32, 'N1': 64, 'Nh': 4},
    "dm": {'N0': 64, 'N1': 64, 'N2': 20},
}

# define run name tag
tag = datetime.now().strftime("_%Y-%m-%d_%H-%M")

config_runtime = {
    'run_name': 's_v6_4'+tag,
    'output_dir': 'save',
    'reload': True,
    'device': 'cuda',
    'num_epochs': 100,
    'log_step': 512,
    'eval_step': 512*8,
    'eval_size': 512,
    'learning_rate': 1e-4,
    'pos_weight_factor': 0.9,
    'comment': "",
}
