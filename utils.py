import torch
import random, os
import numpy as np
import wandb
import json
from pathlib import Path

def seed_setting(seed_number):
    random.seed(seed_number)
    os.environ['PYTHONHASHSEED'] = str(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def setup_wandb(args, seed):
    run = wandb.init(
        project=args.project_name,
        name=args.experiment_name + f'_seed_{seed}',
        config=args,
        mode='online',
    )
    return run


def load_raw_data(dataset):
    data_filepath = f'datasets/{dataset}/'
    print('Loading data...')
    train_idx = torch.load(data_filepath+'train_idx.pt')
    valid_idx = torch.load(data_filepath+'valid_idx.pt')
    test_idx = torch.load(data_filepath+'test_idx.pt')
    
    user_text = json.load(open(data_filepath+'user_sequence.json'))
    labels = torch.load(data_filepath+'labels.pt')
   
    edge_index = torch.load(data_filepath+'edge_index.pt')
    edge_type = torch.load(data_filepath+'edge_type.pt')
    return {'train_idx': train_idx, 
            'valid_idx': valid_idx, 
            'test_idx': test_idx, 
            'user_text': user_text, 
            'labels': labels, 
            'edge_index': edge_index, 
            'edge_type': edge_type}
   

def prepare_path(experiment_name):
    experiment_path = Path(experiment_name)
    filepath = experiment_path
    GNN_filepath = filepath / 'GNN'
    LM_prt_filepath = filepath / 'LM_train'
    LM_prt_filepath.mkdir(exist_ok=True, parents=True)
    GNN_filepath.mkdir(exist_ok=True, parents=True)
    
    LM_data_filepath = experiment_path / 'data' / 'LM'
    LM_data_filepath.mkdir(exist_ok=True, parents=True)
    print(LM_prt_filepath, GNN_filepath, LM_data_filepath)
    return LM_prt_filepath, GNN_filepath, LM_data_filepath
    
def reset_split(n_nodes, ratio):
    idx = torch.randperm(n_nodes)  # Randomly shuffle node indices
    # Convert ratio to a list of floats
    split = list(map(float, ratio.split(',')))
    
    # Normalize the ratios to sum to 1
    total = sum(split)
    train_ratio = split[0] / total
    valid_ratio = split[1] / total

    # Calculate the indices for train, validation, and test splits
    train_idx = idx[:int(train_ratio * n_nodes)]
    valid_idx = idx[int(train_ratio * n_nodes):int((train_ratio + valid_ratio) * n_nodes)]
    test_idx = idx[int((train_ratio + valid_ratio) * n_nodes):]
    
    return train_idx, valid_idx, test_idx

