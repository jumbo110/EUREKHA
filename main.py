'''
We adapted the code provided by Zijian Cai et al. (2024), and their publicly available code from the GitHub repository:
https://github.com/czjdsg/LMBot
 
'''

from parser_args import parser_args
from utils import *
from trainer import LM_Trainer, GNN_Trainer
import torch.nn as nn

def main(args):
    
    for seed in list(map(int, args.seeds.strip().split(','))):
        
        seed_setting(seed)
        LM_prt_filepath, GNN_filepath, LM_data_filepath = prepare_path(args.experiment_name + f'_seed_{seed}')
        
        data = load_raw_data(args.dataset)

        run = setup_wandb(args, seed)

        if args.reset_split != '-1':
            train_idx, valid_idx, test_idx = reset_split(len(data['user_text']), args.reset_split)
            data['train_idx'], data['valid_idx'], data['test_idx'] = train_idx, valid_idx, test_idx

        LMTrainer = LM_Trainer(
            model_name=args.LM_model,
            classifier_n_layers=args.LM_classifier_n_layers,
            classifier_hidden_dim=args.LM_classifier_hidden_dim,
            device=args.device,
            train_epochs=args.LM_train_epochs,
            optimizer_name=args.optimizer_LM,
            lr=args.lr_LM,
            weight_decay=args.weight_decay_LM,
            dropout=args.dropout,
            att_dropout=args.LM_att_dropout,
            lm_dropout=args.LM_dropout,
            warmup=args.warmup,
            max_length=args.max_length,
            batch_size=args.batch_size_LM,
            grad_accumulation=args.LM_accumulation,
            pl_ratio=args.pl_ratio_LM,
            LM_data_filepath=LM_data_filepath,
            train_filepath=LM_prt_filepath,
            train_idx=data['train_idx'],
            valid_idx=data['valid_idx'],
            test_idx=data['test_idx'],
            labels=data['labels'],
            user_seq=data['user_text'],
            run=run,
            eval_patience=args.LM_eval_patience,
            activation=args.activation
        )
        
      

        LMTrainer.build_model()
        LMTrainer.train()
        print(f'Best LM is iter {LMTrainer.best_iter} epoch {LMTrainer.best_epoch}!')
        LMTrainer.test()

      
        GNNTrainer = GNN_Trainer(
            model_name=args.GNN_model,
            device=args.device,
            optimizer_name=args.optimizer_GNN,
            lr=args.lr_GNN,
            weight_decay=args.weight_decay_GNN,
            dropout=args.GNN_dropout,
            batch_size=args.batch_size_GNN,
            gnn_n_layers=args.n_layers,
            n_relations=args.n_relations,
            activation=args.activation,
            gnn_epochs_per_iter=args.GNN_epochs_per_iter,
            pl_ratio=args.pl_ratio_GNN,
            GNN_filepath=GNN_filepath,
            train_idx=data['train_idx'],
            valid_idx=data['valid_idx'],
            test_idx=data['test_idx'],
            labels=data['labels'],
            edge_index=data['edge_index'],
            edge_type=data['edge_type'],
            run=run,
            att_heads=args.att_heads,
            gnn_hidden_dim=args.hidden_dim,
            lm_name = args.LM_model
        )

        GNNTrainer.build_model()
        embeddings_LM = LMTrainer.load_embedding()
        GNNTrainer.train(embeddings_LM)
        print(f'Best GNN is iter {GNNTrainer.best_iter} epoch {GNNTrainer.best_epoch}!')
        GNNTrainer.test(embeddings_LM)

        
if __name__ == '__main__':
    args = parser_args()
    main(args)
