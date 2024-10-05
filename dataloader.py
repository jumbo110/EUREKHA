from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

class LM_dataset(Dataset):
    def __init__(self, user_text: list, labels: torch.Tensor):
        self.user_text = user_text
        self.labels = labels
        
    def __getitem__(self, index):
        text = self.user_text[index]
        label = self.labels[index]
        return text, label
    

    def __len__(self):
        return len(self.user_text)
    

    
def build_LM_dataloader(dataloader_config, idx, user_seq, labels, mode):
    batch_size = dataloader_config['batch_size']
  

    if mode == 'train':
        user_text = []
        for i in idx:
            user_text.append(user_seq[i.item()])
        loader = DataLoader(dataset=LM_dataset(user_text, labels[idx]), shuffle=True, batch_size=batch_size)

    
    elif mode == 'infer':
        loader = DataLoader(dataset=LM_dataset(user_seq, labels), batch_size=batch_size*5)


    elif mode == 'eval':
        user_text = []
        for i in idx:
            user_text.append(user_seq[i.item()])
        loader = DataLoader(dataset=LM_dataset(user_text, labels[idx]), batch_size=batch_size*5)
    
    else:
        raise ValueError('mode should be in ["eval", "infer" ,"train"].')

    return loader


def build_GNN_dataloader(dataloader_config, idx, LM_embedding, labels, edge_index, edge_type, mode):
    batch_size = dataloader_config['batch_size']
    n_layers = dataloader_config['n_layers']
    
    data = Data(x=LM_embedding, edge_index=edge_index, edge_type=edge_type, labels=labels)
    data.num_nodes = LM_embedding.shape[0]
    if mode == 'train':
        loader = NeighborLoader(data=data, num_neighbors=[-1]*n_layers, batch_size=batch_size, input_nodes=idx, shuffle=True)

    elif mode == 'eval':
        loader = NeighborLoader(data=data, num_neighbors=[-1]*n_layers, input_nodes=idx, batch_size=batch_size)
    
    else:
        raise ValueError('mode should be in ["train", "eval"].')

    return loader
