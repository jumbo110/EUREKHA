from LM import LM_Model
from GNNs import RGCN, GCN, GAT, GATv2 
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer

import torch


def build_LM_model(model_config):
    # Build LM model
    LM_model = LM_Model(model_config).to(model_config['device'])
    # Build tokenizer
    LM_model_name = model_config['lm_model'].lower()

    tokenizer_map = {
        'roberta': 'roberta-base',
        'bert': 'bert-base-uncased',
        'albert': 'albert-base-v2',
        'xlnet': 'xlnet-base-cased',
    }

    if LM_model_name in tokenizer_map:
        LM_tokenizer = AutoTokenizer.from_pretrained(tokenizer_map[LM_model_name])
    else:
        raise ValueError(f"Unsupported LM model: {LM_model_name}")

    special_tokens_dict = {'additional_special_tokens': ['METADATA:','REPLY:','THREAD:']}
    LM_tokenizer.add_special_tokens(special_tokens_dict)
    tokens_list = ['CITE', 'CODE', 'URL', 'QUOTE', 'None']
    LM_tokenizer.add_tokens(tokens_list)
    LM_model.LM.resize_token_embeddings(len(LM_tokenizer))
    
    print('Information about LM model:')
    print('total params:', sum(p.numel() for p in LM_model.parameters()))
    return LM_model, LM_tokenizer



def build_GNN_model(model_config):
    # build GNN_model
    GNN_model_name = model_config['GNN_model'].lower()
    if GNN_model_name == 'gcn':
        GNN_model = GCN(model_config).to(model_config['device'])
    elif GNN_model_name == 'rgcn':
        GNN_model = RGCN(model_config).to(model_config['device'])
    elif GNN_model_name == 'gat':
        GNN_model = GAT(model_config).to(model_config['device'])
    elif GNN_model_name == 'gatv2':
        GNN_model = GATv2(model_config).to(model_config['device'])
        
    else:
        raise ValueError('')

    print('Information about GNN model:')
    print(GNN_model)
    print('total params:', sum(p.numel() for p in GNN_model.parameters()))
    

    return GNN_model




